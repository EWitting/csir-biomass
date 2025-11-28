"""RealMLP Trainer with all paper-specific training procedures."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.modules.training.mlp_trainer import MLPTrainer
from src.modules.training.models.realmlp_head import RealMLPHead
from src.modules.training.schedulers import FlatCosScheduler


@dataclass
class RealMLPTrainer(MLPTrainer):
    """RealMLP Trainer with all RealMLP-TD paper features.

    Features:
    - Data-dependent weight initialization
    - Parameter groups with different learning rates (scaling layer, biases)
    - Scheduled dropout (flat_cos)
    - Scheduled weight decay (flat_cos)
    - Output bounds for clipping
    """

    # Base values for scheduling
    base_dropout: float = field(default=0.1, init=True, repr=True, compare=False)
    base_weight_decay: float = field(default=0.02, init=True, repr=True, compare=False)

    # Learning rate multipliers for parameter groups
    scaling_layer_lr_mult: float = field(default=6.0, init=True, repr=True, compare=False)
    bias_lr_mult: float = field(default=0.1, init=True, repr=True, compare=False)

    # Schedulers (initialized during training)
    _dropout_scheduler: Optional[FlatCosScheduler] = field(
        default=None, init=False, repr=False, compare=False
    )
    _weight_decay_scheduler: Optional[FlatCosScheduler] = field(
        default=None, init=False, repr=False, compare=False
    )

    def _initialize_model_statistics(
        self, train_loader: DataLoader[tuple[Tensor, ...]]
    ) -> None:
        """Initialize model with data-dependent initialization and output bounds.

        Args:
            train_loader: Training dataloader
        """
        if not isinstance(self.model, RealMLPHead):
            return

        self.log_to_terminal("Initializing RealMLP statistics from training data...")

        # Collect training data (inputs and targets)
        all_inputs = []
        all_targets = []
        for batch in train_loader:
            x_batch, y_batch = batch
            all_inputs.append(x_batch)
            all_targets.append(y_batch)

        all_inputs = torch.cat(all_inputs, dim=0).to(self.device)
        all_targets = torch.cat(all_targets, dim=0).to(self.device)

        # 1. Data-dependent weight initialization
        self.log_to_terminal("Performing data-dependent weight initialization...")
        self.model.initialize_weights_data_dependent(all_inputs, target_std=1.0)
        self.log_to_terminal("Data-dependent initialization complete")

        # 2. Set output bounds for clipping
        self.model.set_output_bounds(all_targets)
        self.log_to_terminal(
            f"Output bounds set - Min: {self.model.output_min.cpu().numpy()}, "
            f"Max: {self.model.output_max.cpu().numpy()}"
        )

    def custom_train(self, x, y, **train_args):
        """Override custom_train to initialize model statistics.

        Args:
            x: The input to the system.
            y: The expected output of the system.
            train_args: The keyword arguments (includes train_indices, validation_indices).

        Returns:
            The input and output of the system.
        """
        # Store validation indices for scorer access (from parent)
        validation_indices = train_args.get("validation_indices")
        if validation_indices is not None:
            self._current_validation_indices = (
                list(validation_indices)
                if not isinstance(validation_indices, list)
                else validation_indices
            )

        # Get train indices
        train_indices = train_args.get("train_indices")

        # Initialize model if it's RealMLPHead
        if isinstance(self.model, RealMLPHead):
            # Create a temporary dataloader with training data to initialize statistics
            from torch.utils.data import TensorDataset

            if train_indices is not None:
                x_train = x[train_indices]
                y_train = y[train_indices]
            else:
                x_train = x
                y_train = y

            train_dataset = TensorDataset(
                torch.from_numpy(x_train), torch.from_numpy(y_train)
            )
            train_loader_temp = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=False
            )

            # Initialize model statistics
            self._initialize_model_statistics(train_loader_temp)

        # Initialize schedulers
        self._dropout_scheduler = FlatCosScheduler(self.base_dropout, self.epochs)
        self._weight_decay_scheduler = FlatCosScheduler(
            self.base_weight_decay, self.epochs
        )

        # Call parent's custom_train (MLPTrainer.custom_train)
        return super().custom_train(x, y, **train_args)

    def _create_param_groups(self):
        """Create parameter groups with different learning rates.

        From paper:
        - Scaling layer parameters: 6x base LR
        - Bias parameters: 0.1x base LR
        - All other parameters: 1x base LR
        """
        if not isinstance(self.model, RealMLPHead):
            return [{"params": self.model.parameters()}]

        base_lr = self.optimizer_partial.keywords.get("lr", 1e-3)

        # Separate parameters into groups
        scaling_params = []
        bias_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "scaling_layer" in name:
                scaling_params.append(param)
            elif "bias" in name:
                bias_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {
                "params": scaling_params,
                "lr": base_lr * self.scaling_layer_lr_mult,
                "name": "scaling_layer",
            },
            {"params": bias_params, "lr": base_lr * self.bias_lr_mult, "name": "biases"},
            {"params": other_params, "lr": base_lr, "name": "other"},
        ]

        return param_groups

    def _update_dropout(self, epoch: int):
        """Update dropout probability based on flat_cos schedule.

        Args:
            epoch: Current epoch
        """
        if not isinstance(self.model, RealMLPHead) or self._dropout_scheduler is None:
            return

        new_dropout = self._dropout_scheduler.get_value(epoch)

        # Update dropout in all dropout layers
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = new_dropout

    def _update_weight_decay(self, epoch: int):
        """Update weight decay based on flat_cos schedule.

        Args:
            epoch: Current epoch
        """
        if self._weight_decay_scheduler is None:
            return

        new_weight_decay = self._weight_decay_scheduler.get_value(epoch)

        # Update weight decay in all parameter groups
        for param_group in self.initialized_optimizer.param_groups:
            param_group["weight_decay"] = new_weight_decay

    def _training_loop(
        self,
        train_loader: DataLoader[tuple[Tensor, ...]],
        validation_loader: DataLoader[tuple[Tensor, ...]],
        train_losses: list[float],
        val_losses: list[float],
        fold: int = -1,
        start_epoch: int = 0,
    ) -> None:
        """Override training loop to add dropout and weight decay scheduling.

        Args:
            train_loader: Dataloader for the training data.
            validation_loader: Dataloader for the validation data.
            train_losses: List of train losses.
            val_losses: List of validation losses.
            fold: Fold number.
            start_epoch: Starting epoch.
        """
        # Call parent to set up metrics and initial scheduler state
        super()._training_loop(
            train_loader, validation_loader, train_losses, val_losses, fold, start_epoch
        )

    def train_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> float:
        """Train one epoch with scheduled dropout and weight decay.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch

        Returns:
            Average training loss
        """
        # Update schedules before training
        self._update_dropout(epoch)
        self._update_weight_decay(epoch)

        # Call parent's train_one_epoch
        return super().train_one_epoch(dataloader, epoch)
