"""MLP trainer for tabular/embedding data."""

import contextlib
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import torch
from epochlib.training import TorchTrainer
from epochlib.training.utils import batch_to_device
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.scoring.scorer import Scorer


@dataclass
class MLPTrainer(TorchTrainer, Logger):
    """Simple MLP trainer for tabular data with scorer support.

    Extends TorchTrainer with:
    - Custom scorer for validation metrics
    - Logging support
    - Gradient clipping
    """

    scorer: Optional[Scorer] = None  # For exact validation metric computation

    # Gradient clipping
    max_grad_norm: Optional[float] = field(
        default=None, init=True, repr=True, compare=False
    )  # Max gradient norm for clipping (None = disabled)

    # For storing validation indices to pass to scorer
    _current_validation_indices: Optional[list[int]] = field(
        default=None, init=False, repr=False, compare=False
    )
    _current_epoch: int = field(default=0, init=False, repr=False, compare=False)

    def train_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> float:
        """Train the model for one epoch with gradient clipping support.

        :param dataloader: Dataloader for the training data.
        :param epoch: Epoch number.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.train()
        pbar = tqdm(
            dataloader,
            unit="batch",
            desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']:0.8f})",
        )
        for batch in pbar:
            X_batch, y_batch = batch

            X_batch = batch_to_device(X_batch, self.x_tensor_type, self.device)
            y_batch = batch_to_device(y_batch, self.y_tensor_type, self.device)

            # Forward pass
            with torch.autocast(self.device.type) if self.use_mixed_precision else contextlib.nullcontext():  # type: ignore[attr-defined]
                y_pred = self.model(X_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch)

            # Backward pass
            self.initialized_optimizer.zero_grad()
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Apply gradient clipping if enabled
            if self.max_grad_norm is not None:
                if self.use_mixed_precision:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.initialized_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            # Optimizer step
            if self.use_mixed_precision:
                self.scaler.step(self.initialized_optimizer)
                self.scaler.update()
            else:
                self.initialized_optimizer.step()

            # Print tqdm
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Remove the cuda cache
        torch.cuda.empty_cache()
        gc.collect()

        return sum(losses) / len(losses)

    def _training_loop(
        self,
        train_loader: DataLoader[tuple[Tensor, ...]],
        validation_loader: DataLoader[tuple[Tensor, ...]],
        train_losses: list[float],
        val_losses: list[float],
        fold: int = -1,
        start_epoch: int = 0,
    ) -> None:
        """Override training loop to add scorer support.

        :param train_loader: Dataloader for the training data.
        :param validation_loader: Dataloader for the validation data.
        :param train_losses: List of train losses.
        :param val_losses: List of validation losses.
        :param fold: Fold number.
        :param start_epoch: Starting epoch.
        """
        fold_no = "" if fold == -1 else f"_{fold}"

        # Define metrics
        self.external_define_metric(
            self.wrap_log(f"Training/Train Loss{fold_no}"), self.wrap_log("epoch")
        )
        self.external_define_metric(
            self.wrap_log(f"Validation/Validation Loss{fold_no}"),
            self.wrap_log("epoch"),
        )

        # Define metrics for scorer if available
        if self.scorer is not None:
            self.external_define_metric(
                self.wrap_log(f"Validation/{self.scorer.name}"), self.wrap_log("epoch")
            )

            # Define verbose metrics if scorer has verbose mode
            if hasattr(self.scorer, "verbose") and self.scorer.verbose:
                # Define old method metric (for comparison)
                if (
                    hasattr(self.scorer, "log_old_metric")
                    and self.scorer.log_old_metric
                ):
                    self.external_define_metric(
                        self.wrap_log("Validation/weighted_r2"), self.wrap_log("epoch")
                    )

                # Define per-target metrics
                target_names = [
                    "Dry_Clover_g",
                    "Dry_Dead_g",
                    "Dry_Green_g",
                    "GDM_g",
                    "Dry_Total_g",
                ]
                for target_name in target_names:
                    self.external_define_metric(
                        self.wrap_log(f"Validation/R2_target/{target_name}"),
                        self.wrap_log("epoch"),
                    )

        # Set the scheduler to the correct epoch
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=start_epoch)

        for epoch in range(start_epoch, self.epochs):
            # Store current epoch for scorer access
            self._current_epoch = epoch

            # Train one epoch
            train_loss = self.train_one_epoch(train_loader, epoch)
            self.log_to_debug(f"Epoch {epoch} Train Loss: {train_loss}")
            train_losses.append(train_loss)

            # Log train loss
            self.log_to_external(
                message={
                    self.wrap_log(f"Training/Train Loss{fold_no}"): train_losses[-1],
                    self.wrap_log("epoch"): epoch,
                },
            )

            # Step the scheduler
            if self.initialized_scheduler is not None:
                self.initialized_scheduler.step()

            # Checkpointing
            if self.checkpointing_enabled:
                # Save checkpoint
                self._save_model(
                    self.get_model_checkpoint_path(epoch),
                    save_to_external=False,
                    quiet=True,
                )

                # Remove old checkpoints
                if (
                    self.checkpointing_keep_every == 0
                    or epoch % self.checkpointing_keep_every != 0
                ) and self.get_model_checkpoint_path(epoch - 1).exists():
                    self.get_model_checkpoint_path(epoch - 1).unlink()

            # Compute validation loss
            if len(validation_loader) > 0:
                self.last_val_loss = self.val_one_epoch(
                    validation_loader,
                    desc=f"Epoch {epoch} Valid",
                )
                self.log_to_debug(f"Epoch {epoch} Valid Loss: {self.last_val_loss}")
                val_losses.append(self.last_val_loss)

                # Log validation loss
                self.log_to_external(
                    message={
                        self.wrap_log(
                            f"Validation/Validation Loss{fold_no}"
                        ): val_losses[-1],
                        self.wrap_log("epoch"): epoch,
                    },
                )

                # Log train/val loss plot
                self.log_to_external(
                    message={
                        "type": "wandb_plot",
                        "plot_type": "line_series",
                        "data": {
                            "xs": list(range(epoch + 1)),
                            "ys": [train_losses, val_losses],
                            "keys": [f"Train{fold_no}", f"Validation{fold_no}"],
                            "title": self.wrap_log(f"Training/Loss{fold_no}"),
                            "xname": "Epoch",
                        },
                    },
                )

                # Early stopping
                if self._early_stopping():
                    self.log_to_external(
                        message={
                            self.wrap_log(f"Epochs{fold_no}"): (epoch + 1)
                            - self.patience
                        }
                    )
                    break

            # Log the trained epochs to wandb if we finished training
            self.log_to_external(message={self.wrap_log(f"Epochs{fold_no}"): epoch + 1})

    def custom_train(
        self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Override custom_train to store validation indices for scorer access.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The keyword arguments (includes train_indices, validation_indices).
        :return: The input and output of the system.
        """
        # Store validation indices for scorer access
        validation_indices = train_args.get("validation_indices")
        if validation_indices is not None:
            self._current_validation_indices = (
                list(validation_indices)
                if not isinstance(validation_indices, list)
                else validation_indices
            )

        # Call parent custom_train method (TorchTrainer.custom_train)
        return super().custom_train(x, y, **train_args)

    def val_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        desc: str,
    ) -> float:
        """Compute validation loss and exact metric for one epoch.

        Accumulates all predictions and labels to compute exact competition metric.

        :param dataloader: Dataloader for the validation data.
        :param desc: Description for the tqdm progress bar.
        :return: Average validation loss
        """
        losses = []
        all_preds = []
        all_labels = []

        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")

        with torch.no_grad():
            for batch in pbar:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                # Accumulate for metric computation
                losses.append(loss.item())
                all_preds.append(y_pred.cpu())
                all_labels.append(y_batch.cpu())

                # Update progress bar with average loss
                avg_loss = sum(losses) / len(losses)
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=avg_loss)

        avg_loss = sum(losses) / len(losses)

        # Compute exact metric if scorer is provided
        if self.scorer is not None:
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()

            # Pass epoch, indices, and logger to scorer for verbose logging
            metric_score = self.scorer(
                all_labels,
                all_preds,
                epoch=self._current_epoch,
                indices=(
                    np.array(self._current_validation_indices)
                    if self._current_validation_indices
                    else None
                ),
                logger=self,
            )

            # Log main metric to wandb with epoch
            self.log_to_external(
                message={
                    self.wrap_log(f"Validation/{self.scorer.name}"): metric_score,
                    self.wrap_log("epoch"): self._current_epoch,
                }
            )

        # Always return the loss for early stopping and tracking
        return avg_loss

    def _load_model(self, path: Path | None = None) -> None:
        """Load the model from the model_directory folder."""
        model_path = path if path is not None else self.get_model_path()

        # Check if the model exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found in {model_path}",
            )

        # Load model
        self.log_to_terminal(
            f"Loading model from {model_path}",
        )
        checkpoint = torch.load(model_path, weights_only=False)

        # Load the weights from the checkpoint
        if isinstance(checkpoint, nn.DataParallel):
            model = checkpoint.module
        else:
            model = checkpoint

        # Set the current model to the loaded model
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())
