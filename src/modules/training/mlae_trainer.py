"""MLAE (Masked LoRA Experts) Trainer for biomass regression.

This trainer implements fine-tuning with Masked LoRA Experts on vision transformers.
Based on MainTrainer but adapted for MLAE-specific features like expert masking.
"""

import contextlib
import copy
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import torch
import wandb
from epochlib.data import Data
from epochlib.training import TorchTrainer
from epochlib.training.utils import batch_to_device
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.modules.training.augmentation import Augmentation
from src.modules.training.dataset import AugmentedDataset
from src.scoring.scorer import Scorer


@dataclass
class MLAETrainer(TorchTrainer, Logger):
    """MLAE training block for biomass regression with vision transformers.

    Extends TorchTrainer with MLAE-specific features:
    - Fixed mask application for permanent expert masking
    - Expert-level dropout during training
    - Support for various masking strategies
    """

    augmentations: Optional[Augmentation] = None
    val_augmentations: Optional[Augmentation] = None
    gradient_accumulation_steps: int = 1
    tta_hflip: bool = field(default=False, init=True, repr=False, compare=False)
    tta_vflip: bool = field(default=False, init=True, repr=False, compare=False)
    tta_rotate: bool = field(default=False, init=True, repr=False, compare=False)
    tta_sliding_window: int = field(
        default=1, init=True, repr=False, compare=False
    )
    nominal_batch_size: int = field(default=16, init=True, repr=False, compare=False)
    scorer: Optional[Scorer] = None
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Gradient clipping
    max_grad_norm: Optional[float] = field(
        default=None, init=True, repr=True, compare=False
    )

    # MLAE-specific: apply fixed mask after initialization
    apply_fixed_mask: bool = field(
        default=False, init=True, repr=True, compare=False
    )

    # Gradual unfreezing: train only head for first N epochs, then unfreeze backbone
    freeze_backbone_epochs: int = field(
        default=0, init=True, repr=True, compare=False
    )  # Number of epochs to keep backbone frozen (0 = train from start)

    # Embedding extraction mode
    return_embeddings: bool = field(
        default=False, init=True, repr=False, compare=False
    )

    # For storing validation indices to pass to scorer
    _current_validation_indices: Optional[list[int]] = field(
        default=None, init=False, repr=False, compare=False
    )
    _current_epoch: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Post-initialization to ensure parent class initialization."""
        super().__post_init__()
        # Initialize early stopping counter (TorchTrainer expects this)
        if not hasattr(self, 'early_stopping_counter'):
            self.early_stopping_counter = 0

    def _post_model_init(self) -> None:
        """Hook called after model initialization.

        Apply fixed mask if using fixed or mixed masking strategy.
        """
        super()._post_model_init()

        # Apply fixed mask if the model supports it
        if self.apply_fixed_mask and hasattr(self.model, 'apply_fixed_mask'):
            self.log_to_terminal("Applying fixed expert mask to MLAE layers")
            if isinstance(self.model, nn.DataParallel):
                self.model.module.apply_fixed_mask()
            else:
                self.model.apply_fixed_mask()

        # Freeze backbone for gradual unfreezing if requested
        if self.freeze_backbone_epochs > 0:
            self._freeze_backbone()
            self.log_to_terminal(
                f"Backbone frozen for first {self.freeze_backbone_epochs} epochs (training head only)"
            )

        # Print trainable parameters for debugging
        if hasattr(self.model, 'print_trainable_parameters'):
            if isinstance(self.model, nn.DataParallel):
                self.model.module.print_trainable_parameters()
            else:
                self.model.print_trainable_parameters()

    def _freeze_backbone(self) -> None:
        """Freeze backbone (LoRA) parameters, keeping only head trainable."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
            self.log_to_terminal("Frozen backbone parameters (LoRA experts)")

    def _unfreeze_backbone(self) -> None:
        """Unfreeze backbone (LoRA) parameters for full training."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        if hasattr(model, 'get_lora_parameters'):
            # Only unfreeze LoRA parameters, not the entire backbone
            for param in model.get_lora_parameters():
                param.requires_grad = True
            self.log_to_terminal(
                f"Unfrozen backbone at epoch {self._current_epoch} - now training LoRA experts + head"
            )
        elif hasattr(model, 'backbone'):
            # Fallback: unfreeze all backbone params
            for param in model.backbone.parameters():
                param.requires_grad = True
            self.log_to_terminal(
                f"Unfrozen entire backbone at epoch {self._current_epoch}"
            )

    def create_datasets(
        self,
        x: npt.NDArray[np.float32] | Data,
        y: npt.NDArray[np.float32] | Data,
        train_indices: list[int],
        validation_indices: list[int],
    ) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        """Create the datasets for training and validation.

        :param x: The input data (preprocessed images as tensors)
        :param y: The target variable
        :param train_indices: The indices to train on
        :param validation_indices: The indices to validate on
        :return: The training and validation datasets
        """
        if self.dataset is None:
            train_dataset = AugmentedDataset(
                torch.from_numpy(x[train_indices]),
                torch.from_numpy(y[train_indices]),
                self.augmentations,
                device=self.device,
                mean=self.mean,
                std=self.std,
            )
            validation_dataset = AugmentedDataset(
                torch.from_numpy(x[validation_indices]),
                torch.from_numpy(y[validation_indices]),
                self.val_augmentations,
                device=self.device,
                mean=self.mean,
                std=self.std,
            )
            return train_dataset, validation_dataset

        return self.dataset(x[train_indices], y[train_indices]), self.dataset(
            x[validation_indices], y[validation_indices]
        )

    def _concat_datasets(
        self,
        train_dataset,
        validation_dataset,
        train_indices: list[int] | npt.NDArray[np.int32],
        validation_indices: list[int] | npt.NDArray[np.int32],
    ) -> Dataset:
        """Concatenate the training and validation datasets.

        :param train_dataset: The training dataset
        :param validation_dataset: The validation dataset
        :param train_indices: The indices for the training data
        :param validation_indices: The indices for the validation data
        :return: A new dataset containing the concatenated data
        """
        return AugmentedDataset(
            torch.cat([train_dataset.x, validation_dataset.x]),
            torch.cat([train_dataset.y, validation_dataset.y]),
            augmentation=None,
            device=self.device,
            mean=self.mean,
            std=self.std,
        )

    def create_prediction_dataset(
        self,
        x: npt.NDArray[np.float32] | Data,
    ) -> Dataset[tuple[Tensor, ...]]:
        """Create the prediction dataset.

        :param x: The input data (images in [0, 1] range)
        :return: The prediction dataset
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x)
        else:
            x_tensor = torch.tensor(x, dtype=self.x_tensor_type)

        dummy_y = torch.zeros((len(x_tensor), 1), dtype=torch.float32)

        return AugmentedDataset(
            x_tensor,
            dummy_y,
            augmentation=None,
            device=self.device,
            mean=self.mean,
            std=self.std,
        )

    def predict_after_train(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        train_dataset: Dataset[Any],
        validation_dataset: Dataset[Any],
        train_indices: list[int],
        validation_indices: list[int],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Predict after training the model.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_dataset: The training dataset.
        :param validation_dataset: The validation dataset.
        :param train_indices: The indices to train on.
        :param validation_indices: The indices to validate on.

        :return: The predictions (or embeddings) and the expected output.
        """
        match self.to_predict:
            case "all":
                concat_dataset: Dataset[Any] = self._concat_datasets(
                    train_dataset,
                    validation_dataset,
                    train_indices,
                    validation_indices,
                )
                pred_dataloader = DataLoader(
                    concat_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=(
                        self.collate_fn if hasattr(concat_dataset, "__getitems__") else None
                    ),
                    **self.dataloader_args,
                )
                return self.predict_on_loader(pred_dataloader), y
            case "validation":
                validation_dataset = self.create_prediction_dataset(x[validation_indices])
                validation_loader = DataLoader(
                    validation_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=(
                        self.collate_fn if hasattr(validation_dataset, "__getitems__") else None
                    ),
                    **self.dataloader_args,
                )
                return self.predict_on_loader(validation_loader), y[validation_indices]
            case "none":
                return x, y
            case _:
                raise ValueError("to_predict should be either 'validation', 'all' or 'none'")

    def train_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> float:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data
        :param epoch: Epoch number
        :return: Average loss for the epoch
        """
        losses = []
        self.model.train()
        pbar = tqdm(
            dataloader,
            unit="batch",
            desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']:0.8f})",
        )
        self.initialized_optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            X_batch, y_batch = batch

            X_batch = batch_to_device(X_batch, self.x_tensor_type, self.device)
            y_batch = batch_to_device(y_batch, self.y_tensor_type, self.device)

            # Forward pass with gradient accumulation
            with torch.autocast(self.device.type) if self.use_mixed_precision else contextlib.nullcontext():  # type: ignore[attr-defined]
                y_pred = self.model(X_batch)
                raw_loss = self.criterion(y_pred, y_batch)

            # Scale loss to nominal batch size
            loss = raw_loss * (self.batch_size / self.nominal_batch_size)

            # Backward pass: accumulate gradients
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            losses.append(raw_loss.item())

            # Perform optimizer step every 'accumulation_steps' batches
            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm is not None:
                    if self.use_mixed_precision:
                        self.scaler.unscale_(self.initialized_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                if self.use_mixed_precision:
                    self.scaler.step(self.initialized_optimizer)
                    self.scaler.update()
                else:
                    self.initialized_optimizer.step()
                self.initialized_optimizer.zero_grad()
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Final step if remaining gradients didn't trigger an update
        if (i + 1) % self.gradient_accumulation_steps != 0:
            if self.max_grad_norm is not None:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.initialized_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            if self.use_mixed_precision:
                self.scaler.step(self.initialized_optimizer)
                self.scaler.update()
            else:
                self.initialized_optimizer.step()
            self.initialized_optimizer.zero_grad()

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
        """Training loop with verbose scoring support.

        :param train_loader: Dataloader for the training data.
        :param validation_loader: Dataloader for the validation data.
        :param train_losses: List of train losses.
        :param val_losses: List of validation losses.
        :param fold: Fold number.
        :param start_epoch: Starting epoch.
        """
        fold_no = ""

        if fold > -1:
            fold_no = f"_{fold}"

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

            if hasattr(self.scorer, "verbose") and self.scorer.verbose:
                if (
                    hasattr(self.scorer, "log_old_metric")
                    and self.scorer.log_old_metric
                ):
                    self.external_define_metric(
                        self.wrap_log("Validation/weighted_r2"), self.wrap_log("epoch")
                    )

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

        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=start_epoch)

        for epoch in range(start_epoch, self.epochs):
            self._current_epoch = epoch

            # Gradual unfreezing: unfreeze backbone after freeze_backbone_epochs
            if epoch == self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0:
                self._unfreeze_backbone()

            train_loss = self.train_one_epoch(train_loader, epoch)
            self.log_to_debug(f"Epoch {epoch} Train Loss: {train_loss}")
            train_losses.append(train_loss)

            self.log_to_external(
                message={
                    self.wrap_log(f"Training/Train Loss{fold_no}"): train_losses[-1],
                    self.wrap_log("epoch"): epoch,
                },
            )

            if self.initialized_scheduler is not None:
                self.initialized_scheduler.step(epoch=epoch + 1)

            # Checkpointing
            if self.checkpointing_enabled:
                self._save_model(
                    self.get_model_checkpoint_path(epoch),
                    save_to_external=False,
                    quiet=True,
                )

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

                self.log_to_external(
                    message={
                        self.wrap_log(
                            f"Validation/Validation Loss{fold_no}"
                        ): val_losses[-1],
                        self.wrap_log("epoch"): epoch,
                    },
                )

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

            self.log_to_external(message={self.wrap_log(f"Epochs{fold_no}"): epoch + 1})

    def train(
        self,
        x: npt.NDArray[np.float32] | Data,
        y: npt.NDArray[np.float32] | Data,
        train_indices: list[int] | npt.NDArray[np.int32],
        validation_indices: list[int] | npt.NDArray[np.int32],
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.float32] | None]:
        """Train with validation indices tracking for scorer.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_indices: The indices to train on.
        :param validation_indices: The indices to validate on.
        :param kwargs: Additional arguments.
        :return: The predictions and the expected output.
        """
        self._current_validation_indices = (
            list(validation_indices)
            if not isinstance(validation_indices, list)
            else validation_indices
        )

        return super().train(
            x,
            y,
            train_indices=train_indices,
            validation_indices=validation_indices,
            **kwargs,
        )

    def val_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        desc: str,
    ) -> float:
        """Compute validation loss and exact metric for one epoch.

        :param dataloader: Dataloader for the validation data.
        :param desc: Description for the tqdm progress bar.
        :return: Competition metric score (or average loss if no scorer provided)
        """

        losses = []
        all_preds = []
        all_labels = []

        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")

        with torch.no_grad():
            for batch in pbar:
                X_batch, y_batch = batch

                X_batch = batch_to_device(X_batch, self.x_tensor_type, self.device)
                y_batch = batch_to_device(y_batch, self.y_tensor_type, self.device)

                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                losses.append(loss.item())
                all_preds.append(y_pred.cpu())
                all_labels.append(y_batch.cpu())

                avg_loss = sum(losses) / len(losses)
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=avg_loss)

        avg_loss = sum(losses) / len(losses)

        # Compute exact metric if scorer is provided
        if self.scorer is not None:
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()

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

            self.log_to_external(
                message={
                    self.wrap_log(f"Validation/{self.scorer.name}"): metric_score,
                    self.wrap_log("epoch"): self._current_epoch,
                }
            )

        return avg_loss

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
        compile_method: str | None = None,
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader with optional TTA.

        :param loader: The loader to predict on
        :param compile_method: Compilation method (unused)
        :return: The predictions (or embeddings if return_embeddings=True)
        """
        if self.return_embeddings:
            self.log_to_terminal(
                "Extracting embeddings with TTA from the given dataloader"
            )
        else:
            self.log_to_terminal("Running inference on the given dataloader")
        self.model.eval()
        predictions = []

        loader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            collate_fn=(
                self.collate_fn if hasattr(loader.dataset, "__getitems__") else None
            ),
            **self.dataloader_args,
        )

        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = batch_to_device(data[0], self.x_tensor_type, self.device)

                if (
                    self.tta_sliding_window > 1
                    or self.tta_hflip
                    or self.tta_vflip
                    or self.tta_rotate
                ):
                    all_preds = []

                    windows = (
                        self._get_sliding_windows(X_batch)
                        if self.tta_sliding_window > 1
                        else [(X_batch, slice(None))]
                    )

                    for window, _ in windows:
                        hflip_options = [False, True] if self.tta_hflip else [False]
                        vflip_options = [False, True] if self.tta_vflip else [False]
                        if self.tta_rotate:
                            rot_options = (
                                [0, 1]
                                if (self.tta_hflip and self.tta_vflip)
                                else [0, 1, 2, 3]
                            )
                        else:
                            rot_options = [0]

                        for hflip in hflip_options:
                            for vflip in vflip_options:
                                for rot in rot_options:
                                    window_transformed = window.clone()

                                    if hflip:
                                        window_transformed = torch.flip(
                                            window_transformed, dims=[3]
                                        )
                                    if vflip:
                                        window_transformed = torch.flip(
                                            window_transformed, dims=[2]
                                        )
                                    if rot != 0:
                                        window_transformed = torch.rot90(
                                            window_transformed, k=rot, dims=[2, 3]
                                        )

                                    if self.return_embeddings:
                                        if hasattr(self.model, "get_embeddings"):
                                            y_pred = self.model.get_embeddings(
                                                window_transformed
                                            )
                                        elif hasattr(self.model, "module") and hasattr(
                                            self.model.module, "get_embeddings"
                                        ):
                                            y_pred = self.model.module.get_embeddings(
                                                window_transformed
                                            )
                                        else:
                                            raise AttributeError(
                                                f"Model {self.model.__class__.__name__} does not have a get_embeddings method."
                                            )
                                    else:
                                        y_pred = self.model(window_transformed)
                                    all_preds.append(y_pred)

                    pred = torch.mean(torch.stack(all_preds), dim=0)
                    predictions.extend(pred.cpu().numpy())
                else:
                    if self.return_embeddings:
                        if hasattr(self.model, "get_embeddings"):
                            y_pred = self.model.get_embeddings(X_batch).cpu().numpy()
                        elif hasattr(self.model, "module") and hasattr(
                            self.model.module, "get_embeddings"
                        ):
                            y_pred = (
                                self.model.module.get_embeddings(X_batch).cpu().numpy()
                            )
                        else:
                            raise AttributeError(
                                f"Model {self.model.__class__.__name__} does not have a get_embeddings method."
                            )
                    else:
                        y_pred = self.model(X_batch).cpu().numpy()
                    predictions.extend(y_pred)

        return np.array(predictions)

    def _get_sliding_windows(self, X_batch: Tensor) -> list[tuple[Tensor, slice]]:
        """Extract sliding windows from horizontally wide images.

        :param X_batch: Input batch (B, C, H, W)
        :return: List of (window_tensor, slice) tuples
        """
        B, C, H, W = X_batch.shape

        if W <= H:
            return [(X_batch, slice(None))]

        window_size = H

        stride = (
            (W - window_size) // (self.tta_sliding_window - 1)
            if self.tta_sliding_window > 1
            else W
        )

        windows = []
        for i in range(self.tta_sliding_window):
            start_w = min(i * stride, W - window_size)
            end_w = start_w + window_size

            window = X_batch[:, :, :, start_w:end_w]
            windows.append((window, slice(start_w, end_w)))

        return windows

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(
                f"{self.trained_models_directory}/{self.get_hash()}.pt"
            )
            wandb.log_artifact(model_artifact)

    def _load_model(self, path: Path | None = None) -> None:
        """Load the model from the model_directory folder."""
        model_path = path if path is not None else self.get_model_path()

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found in {model_path}",
            )

        self.log_to_terminal(
            f"Loading model from {model_path}",
        )
        checkpoint = torch.load(model_path, weights_only=False)

        if isinstance(checkpoint, nn.DataParallel):
            model = checkpoint.module
        else:
            model = checkpoint

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())
