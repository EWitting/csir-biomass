"""Module for biomass training block."""
from dataclasses import dataclass, field
from typing import Optional, Any
import contextlib
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
    nominal_batch_size: int = field(default=16, init=True, repr=False, compare=False)
    scorer: Optional[Scorer] = None  # For exact validation metric computation
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # Normalization mean
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # Normalization std

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

    def val_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        desc: str,
    ) -> float:
        """Compute validation loss and exact metric for one epoch.

        Accumulates all predictions and labels to compute exact competition metric.

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
            
            # Compute the exact competition metric
            metric_score = self.scorer(all_labels, all_preds)
            
            # Log metric to wandb
            self.log_to_external(
                message={
                    self.wrap_log(f"Validation/{self.scorer.name}"): metric_score,
                    self.wrap_log(f"Validation/loss_vs_metric_diff"): avg_loss - (1 - metric_score),
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
                if self.tta_hflip or self.tta_vflip or self.tta_rotate:
                    # Test-time augmentation for regression
                    tta_preds = []
                    
                    # Generate combinations based on enabled TTA options
                    hflip_options = [False, True] if self.tta_hflip else [False]
                    vflip_options = [False, True] if self.tta_vflip else [False]
                    rot_options = [0, 1, 2, 3] if self.tta_rotate else [0]
                    
                    for hflip in hflip_options:
                        for vflip in vflip_options:
                            for rot in rot_options:
                                X_batch = batch_to_device(data[0], self.x_tensor_type, self.device)

                                # Apply transformations
                                if hflip:
                                    X_batch = torch.flip(X_batch, dims=[3])  # Flip width (horizontal)
                                if vflip:
                                    X_batch = torch.flip(X_batch, dims=[2])  # Flip height (vertical)
                                if rot != 0:
                                    X_batch = torch.rot90(X_batch, k=rot, dims=[2, 3])

                                # Predict
                                y_pred = self.model(X_batch)
                                
                                # No need to inverse transform for regression
                                tta_preds.append(y_pred)
                    
                    pred = torch.mean(torch.stack(tta_preds), dim=0)
                    predictions.extend(pred.cpu().numpy())
                else:
                    X_batch = batch_to_device(data[0], self.x_tensor_type, self.device)
                    y_pred = self.model(X_batch).cpu().numpy()
                    predictions.extend(y_pred)
        
        return np.array(predictions)

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