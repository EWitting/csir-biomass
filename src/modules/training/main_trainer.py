"""Module for biomass training block."""
from dataclasses import dataclass, field
from typing import Optional, Any
import contextlib
import copy
import gc

import wandb
import torch
import numpy as np
import numpy.typing as npt
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from epochlib.training import TorchTrainer
from epochlib.training.utils import batch_to_device
from epochlib.data import Data

from src.modules.logging.logger import Logger
from src.modules.training.augmentation import Augmentation
from src.modules.training.dataset import AugmentedDataset
from src.scoring.scorer import Scorer
from pathlib import Path


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block for biomass regression."""

    augmentations: Optional[Augmentation] = None
    val_augmentations: Optional[Augmentation] = None
    gradient_accumulation_steps: int = 1
    tta_hflip: bool = field(default=False, init=True, repr=False, compare=False)
    tta_vflip: bool = field(default=False, init=True, repr=False, compare=False)
    tta_rotate: bool = field(default=False, init=True, repr=False, compare=False)
    tta_sliding_window: int = field(default=1, init=True, repr=False, compare=False)  # Number of horizontal sliding window slices for inference
    nominal_batch_size: int = field(default=16, init=True, repr=False, compare=False)
    scorer: Optional[Scorer] = None  # For exact validation metric computation
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # Normalization mean
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # Normalization std
    
    # EMA (Exponential Moving Average) parameters
    use_ema: bool = field(default=False, init=True, repr=True, compare=False)  # Enable EMA smoothing
    ema_decay: float = field(default=0.99, init=True, repr=True, compare=False)  # EMA decay rate (0.99 = smooth over ~100 steps)
    
    # For storing validation indices to pass to scorer
    _current_validation_indices: Optional[list[int]] = field(default=None, init=False, repr=False, compare=False)
    _current_epoch: int = field(default=0, init=False, repr=False, compare=False)

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
                std=self.std
            )
            validation_dataset = AugmentedDataset(
                torch.from_numpy(x[validation_indices]),
                torch.from_numpy(y[validation_indices]),
                self.val_augmentations,
                device=self.device,
                mean=self.mean,
                std=self.std
            )
            return train_dataset, validation_dataset

        return self.dataset(x[train_indices], y[train_indices]), self.dataset(x[validation_indices], y[validation_indices])

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
            std=self.std
        )

    def create_prediction_dataset(
        self,
        x: npt.NDArray[np.float32] | Data,
    ) -> Dataset[tuple[Tensor, ...]]:
        """Create the prediction dataset.

        Images are in [0, 1] range and need normalization via val_augmentations.

        :param x: The input data (images in [0, 1] range)
        :return: The prediction dataset
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x)
        else:
            x_tensor = torch.tensor(x, dtype=self.x_tensor_type)
        
        # Use AugmentedDataset without augmentations (only normalization)
        # Create dummy targets for the dataset (not used during inference)
        dummy_y = torch.zeros((len(x_tensor), 1), dtype=torch.float32)
        
        return AugmentedDataset(
            x_tensor,
            dummy_y,
            augmentation=None,
            device=self.device,
            mean=self.mean,
            std=self.std
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

        :return: The predictions and the expected output.
        """
        validation_dataset = self.create_prediction_dataset(x[validation_indices])
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=(self.collate_fn if hasattr(validation_dataset, "__getitems__") else None),        )

        return self.predict_on_loader(validation_loader), y[validation_indices]


    def train_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> float:
        """Train the model for one epoch using gradient accumulation.

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
        # Initialize gradients once at the start of the epoch
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

            # Append the original loss for logging
            losses.append(raw_loss.item())

            # Perform optimizer step every 'accumulation_steps' batches
            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.step(self.initialized_optimizer)
                    self.scaler.update()
                else:
                    self.initialized_optimizer.step()
                self.initialized_optimizer.zero_grad()
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Final step if remaining gradients didn't trigger an update
        if (i + 1) % self.gradient_accumulation_steps != 0:
            if self.use_mixed_precision:
                self.scaler.step(self.initialized_optimizer)
                self.scaler.update()
            else:
                self.initialized_optimizer.step()
            self.initialized_optimizer.zero_grad()

        # Remove the CUDA cache
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
        """Override training loop to track validation indices and epoch for verbose scoring.
        
        :param train_loader: Dataloader for the training data.
        :param validation_loader: Dataloader for the validation data. (can be empty)
        :param train_losses: List of train losses.
        :param val_losses: List of validation losses.
        :param fold: Fold number.
        :param start_epoch: Starting epoch.
        """
        fold_no = ""

        if fold > -1:
            fold_no = f"_{fold}"

        self.external_define_metric(self.wrap_log(f"Training/Train Loss{fold_no}"), self.wrap_log("epoch"))
        self.external_define_metric(self.wrap_log(f"Validation/Validation Loss{fold_no}"), self.wrap_log("epoch"))
        
        # Define metrics for scorer if available
        if self.scorer is not None:
            # Define the main scorer metric (new global R²)
            self.external_define_metric(self.wrap_log(f"Validation/{self.scorer.name}"), self.wrap_log("epoch"))
            
            # Define verbose metrics if scorer has verbose mode
            if hasattr(self.scorer, 'verbose') and self.scorer.verbose:
                # Define old method metric (for comparison with historical runs)
                if hasattr(self.scorer, 'log_old_metric') and self.scorer.log_old_metric:
                    self.external_define_metric(self.wrap_log("Validation/weighted_r2"), self.wrap_log("epoch"))
                
                # Define per-target metrics
                target_names = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'GDM_g', 'Dry_Total_g']
                for target_name in target_names:
                    self.external_define_metric(self.wrap_log(f"Validation/R2_target/{target_name}"), self.wrap_log("epoch"))
                
                # Define per-species and per-state metrics (will be created dynamically as encountered)
                # These get defined by wandb automatically when first logged with epoch

        # Set the scheduler to the correct epoch
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=start_epoch)

        for epoch in range(start_epoch, self.epochs):
            # Store current epoch for scorer access
            self._current_epoch = epoch
            
            # Train using train_loader
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
                self.initialized_scheduler.step(epoch=epoch + 1)

            # Checkpointing
            if self.checkpointing_enabled:
                # Save checkpoint
                self._save_model(self.get_model_checkpoint_path(epoch), save_to_external=False, quiet=True)

                # Remove old checkpoints
                if (self.checkpointing_keep_every == 0 or epoch % self.checkpointing_keep_every != 0) and self.get_model_checkpoint_path(epoch - 1).exists():
                    self.get_model_checkpoint_path(epoch - 1).unlink()

            # Compute validation loss
            if len(validation_loader) > 0:
                self.last_val_loss = self.val_one_epoch(
                    validation_loader,
                    desc=f"Epoch {epoch} Valid",
                )
                self.log_to_debug(f"Epoch {epoch} Valid Loss: {self.last_val_loss}")
                val_losses.append(self.last_val_loss)

                # Log validation loss and plot train/val loss against each other
                self.log_to_external(
                    message={
                        self.wrap_log(f"Validation/Validation Loss{fold_no}"): val_losses[-1],
                        self.wrap_log("epoch"): epoch,
                    },
                )

                self.log_to_external(
                    message={
                        "type": "wandb_plot",
                        "plot_type": "line_series",
                        "data": {
                            "xs": list(
                                range(epoch + 1),
                            ),  # Ensure it's a list, not a range object
                            "ys": [train_losses, val_losses],
                            "keys": [f"Train{fold_no}", f"Validation{fold_no}"],
                            "title": self.wrap_log(f"Training/Loss{fold_no}"),
                            "xname": "Epoch",
                        },
                    },
                )

                # Early stopping
                if self._early_stopping():
                    self.log_to_external(message={self.wrap_log(f"Epochs{fold_no}"): (epoch + 1) - self.patience})
                    break

            # Log the trained epochs to wandb if we finished training
            self.log_to_external(message={self.wrap_log(f"Epochs{fold_no}"): epoch + 1})
    
    def train(
        self,
        x: npt.NDArray[np.float32] | Data,
        y: npt.NDArray[np.float32] | Data,
        train_indices: list[int] | npt.NDArray[np.int32],
        validation_indices: list[int] | npt.NDArray[np.int32],
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.float32] | None]:
        """Override train to store validation indices for scorer access.
        
        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_indices: The indices to train on.
        :param validation_indices: The indices to validate on.
        :param kwargs: Additional arguments.
        :return: The predictions and the expected output.
        """
        # Store validation indices for scorer access
        self._current_validation_indices = list(validation_indices) if not isinstance(validation_indices, list) else validation_indices
        
        # Call parent train method - pass indices through kwargs, not as positional args
        return super().train(x, y, train_indices=train_indices, validation_indices=validation_indices, **kwargs)

    def val_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        desc: str,
    ) -> float:
        """Compute validation loss and exact metric for one epoch.

        Accumulates all predictions and labels to compute exact competition metric.
        Uses EMA weights if enabled.

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
                indices=np.array(self._current_validation_indices) if self._current_validation_indices else None,
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

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
        compile_method: str | None = None,
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader with optional TTA.

        :param loader: The loader to predict on
        :param compile_method: Compilation method (unused)
        :return: The predictions
        """
        self.log_to_terminal("Running inference on the given dataloader")
        self.model.eval()
        predictions = []

        # Create a new dataloader from the dataset of the input dataloader with collate_fn
        loader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            collate_fn=(self.collate_fn if hasattr(loader.dataset, "__getitems__") else None),
            **self.dataloader_args,
        )

        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = batch_to_device(data[0], self.x_tensor_type, self.device)
                
                if self.tta_sliding_window > 1 or self.tta_hflip or self.tta_vflip or self.tta_rotate:
                    # Test-time augmentation for regression
                    all_preds = []
                    
                    # Sliding window is the outer loop - for each window, apply TTA
                    windows = self._get_sliding_windows(X_batch) if self.tta_sliding_window > 1 else [(X_batch, slice(None))]
                    
                    for window, _ in windows:
                        # Generate combinations based on enabled TTA options
                        hflip_options = [False, True] if self.tta_hflip else [False]
                        vflip_options = [False, True] if self.tta_vflip else [False]
                        # Optimization: if both hflip and vflip are enabled, only need [0, 1] rotations
                        # because 180° = hflip+vflip and 270° = 90°+hflip+vflip
                        if self.tta_rotate:
                            rot_options = [0, 1] if (self.tta_hflip and self.tta_vflip) else [0, 1, 2, 3]
                        else:
                            rot_options = [0]
                        
                        for hflip in hflip_options:
                            for vflip in vflip_options:
                                for rot in rot_options:
                                    window_transformed = window.clone()
                                    
                                    # Apply transformations to this window
                                    if hflip:
                                        window_transformed = torch.flip(window_transformed, dims=[3])  # Flip width (horizontal)
                                    if vflip:
                                        window_transformed = torch.flip(window_transformed, dims=[2])  # Flip height (vertical)
                                    if rot != 0:
                                        window_transformed = torch.rot90(window_transformed, k=rot, dims=[2, 3])
                                    
                                    # Predict on transformed window
                                    y_pred = self.model(window_transformed)
                                    all_preds.append(y_pred)
                    
                    # Average all predictions (across all windows and TTA variants)
                    pred = torch.mean(torch.stack(all_preds), dim=0)
                    predictions.extend(pred.cpu().numpy())
                else:
                    y_pred = self.model(X_batch).cpu().numpy()
                    predictions.extend(y_pred)
        
        return np.array(predictions)
    
    def _get_sliding_windows(self, X_batch: Tensor) -> list[tuple[Tensor, slice]]:
        """Extract sliding windows from horizontally wide images.
        
        Assumes images are wider than they are tall (2:1 aspect ratio or similar).
        Splits the image into multiple overlapping windows.
        
        :param X_batch: Input batch (B, C, H, W)
        :return: List of (window_tensor, slice) tuples
        """
        B, C, H, W = X_batch.shape
        
        # If image is square or portrait, return the full image
        if W <= H:
            return [(X_batch, slice(None))]
        
        # Calculate window size (square)
        window_size = H
        
        # Calculate stride for overlapping windows
        # Distribute windows evenly across the width with overlap
        stride = (W - window_size) // (self.tta_sliding_window - 1) if self.tta_sliding_window > 1 else W
        
        windows = []
        for i in range(self.tta_sliding_window):
            # Calculate window start position
            start_w = min(i * stride, W - window_size)
            end_w = start_w + window_size
            
            # Extract window
            window = X_batch[:, :, :, start_w:end_w]
            windows.append((window, slice(start_w, end_w)))
        
        return windows

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self.trained_models_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

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