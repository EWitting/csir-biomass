"""Base class for frozen vision foundation models used for feature extraction."""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class FrozenFeatureExtractor(VerboseTransformationBlock):
    """Base class for frozen pretrained vision models that extract embeddings.

    This class handles:
    - Loading pretrained models
    - Batch processing of images
    - GPU acceleration
    - Optional L2 normalization of embeddings

    Subclasses should implement:
    - _load_model(): Load and return the pretrained model
    - _preprocess_batch(): Preprocess images for the specific model
    - _extract_features(): Extract features from model outputs

    Parameters
    ----------
    model_name : str
        Name/identifier of the pretrained model
    device : str
        Device to run inference on ('cuda' or 'cpu')
    batch_size : int
        Batch size for inference
    normalize : bool
        Whether to L2 normalize the extracted embeddings
    """

    model_name: str
    device: str = "cuda"
    batch_size: int = 32
    normalize: bool = True
    target_size: int | None = None  # Optional: resize images to this size before processing
    split_2_to_1_images: bool = True  # Split 2:1 wide images into two squares and average embeddings

    # Internal state (not part of config)
    model: nn.Module | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the model after dataclass initialization."""
        super().__post_init__()

        # Load the pretrained model
        self.log_to_terminal(f"Loading frozen model: {self.model_name}")
        self.model = self._load_model()
        self.model.eval()

        # Move to device
        if torch.cuda.is_available() and self.device == "cuda":
            self.model = self.model.to(self.device)
            self.log_to_terminal(f"Model moved to {self.device}")
        else:
            self.device = "cpu"
            self.model = self.model.to("cpu")
            self.log_to_terminal("CUDA not available, using CPU")

    @abstractmethod
    def _load_model(self) -> nn.Module:
        """Load and return the pretrained model.

        Returns
        -------
        nn.Module
            The loaded pretrained model
        """
        raise NotImplementedError("Subclasses must implement _load_model()")

    @abstractmethod
    def _preprocess_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess a batch of images for the model.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images in shape (B, H, W, C) or (B, C, H, W), values in [0, 1]

        Returns
        -------
        torch.Tensor
            Preprocessed images ready for the model
        """
        raise NotImplementedError("Subclasses must implement _preprocess_batch()")

    @abstractmethod
    def _extract_features(self, model_output: Any) -> torch.Tensor:
        """Extract feature embeddings from model output.

        Parameters
        ----------
        model_output : Any
            Output from the model forward pass

        Returns
        -------
        torch.Tensor
            Feature embeddings in shape (B, embedding_dim)
        """
        raise NotImplementedError("Subclasses must implement _extract_features()")

    def _split_wide_image(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a 2:1 wide image into two square halves.

        Parameters
        ----------
        image : torch.Tensor
            Image tensor in shape (C, H, W) where W is approximately 2*H

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Left and right square crops
        """
        C, H, W = image.shape
        # Split at the midpoint
        mid = W // 2
        left_half = image[:, :, :mid]  # (C, H, W/2)
        right_half = image[:, :, mid:]  # (C, H, W/2)
        return left_half, right_half

    def custom_transform(self, data: npt.NDArray[np.float32], **transform_args: Any) -> npt.NDArray[np.float32]:
        """Extract features from images using the frozen model.

        Parameters
        ----------
        data : np.ndarray
            Input images in shape (N, C, H, W), values in [0, 1] range

        Returns
        -------
        np.ndarray
            Feature embeddings in shape (N, embedding_dim)
        """
        # Check if images are 2:1 aspect ratio
        _, _, H, W = data.shape
        is_2_to_1 = (W / H) > 1.8  # Allow some tolerance

        if self.split_2_to_1_images and is_2_to_1:
            self.log_to_terminal(f"Detected 2:1 images ({W}x{H}). Splitting into two halves and averaging embeddings.")
            return self._extract_features_with_split(data)
        else:
            self.log_to_terminal(f"Extracting features from {len(data)} images using {self.model_name}")
            return self._extract_features_standard(data)

    def _extract_features_standard(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Standard feature extraction without splitting.

        Parameters
        ----------
        data : np.ndarray
            Input images in shape (N, C, H, W), values in [0, 1] range

        Returns
        -------
        np.ndarray
            Feature embeddings in shape (N, embedding_dim)
        """
        all_embeddings = []
        n_batches = (len(data) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for i in tqdm(range(n_batches), desc=f"Extracting features ({self.model_name})"):
                # Get batch
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(data))
                batch_images = data[start_idx:end_idx]

                # Convert to tensor
                # Input is already (B, C, H, W) from ImageLoader
                batch_tensor = torch.from_numpy(batch_images).float()

                # Optional: resize to target size if specified
                if self.target_size is not None:
                    current_size = batch_tensor.shape[2]
                    if current_size != self.target_size:
                        batch_tensor = torch.nn.functional.interpolate(
                            batch_tensor,
                            size=(self.target_size, self.target_size),
                            mode='bilinear',
                            align_corners=False
                        )

                # Preprocess for the specific model
                batch_tensor = self._preprocess_batch(batch_tensor)
                batch_tensor = batch_tensor.to(self.device)

                # Forward pass
                model_output = self.model(batch_tensor)

                # Extract features
                features = self._extract_features(model_output)

                # Optional L2 normalization
                if self.normalize:
                    features = torch.nn.functional.normalize(features, p=2, dim=1)

                # Move to CPU and collect
                all_embeddings.append(features.cpu().numpy())

        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)

        self.log_to_terminal(f"Extracted embeddings with shape {embeddings.shape}")

        return embeddings

    def _extract_features_with_split(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Extract features by splitting 2:1 images and averaging embeddings.

        Parameters
        ----------
        data : np.ndarray
            Input images in shape (N, C, H, W) with W ≈ 2*H, values in [0, 1] range

        Returns
        -------
        np.ndarray
            Feature embeddings in shape (N, embedding_dim), averaged from both halves
        """
        all_embeddings = []
        n_batches = (len(data) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for i in tqdm(range(n_batches), desc=f"Extracting features (split mode, {self.model_name})"):
                # Get batch
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(data))
                batch_images = data[start_idx:end_idx]

                # Convert to tensor
                batch_tensor = torch.from_numpy(batch_images).float()

                # Split each image in the batch into left and right halves
                left_halves = []
                right_halves = []
                for img in batch_tensor:
                    left, right = self._split_wide_image(img)
                    left_halves.append(left)
                    right_halves.append(right)

                # Stack into batches
                left_batch = torch.stack(left_halves)  # (B, C, H, W/2)
                right_batch = torch.stack(right_halves)  # (B, C, H, W/2)

                # Combine both batches for efficient processing
                # Process both halves together in one forward pass
                combined_batch = torch.cat([left_batch, right_batch], dim=0)  # (2B, C, H, W/2)

                # Optional: resize to target size if specified
                if self.target_size is not None:
                    combined_batch = torch.nn.functional.interpolate(
                        combined_batch,
                        size=(self.target_size, self.target_size),
                        mode='bilinear',
                        align_corners=False
                    )

                # Preprocess for the specific model
                combined_batch = self._preprocess_batch(combined_batch)
                combined_batch = combined_batch.to(self.device)

                # Forward pass
                model_output = self.model(combined_batch)

                # Extract features
                features = self._extract_features(model_output)  # (2B, embedding_dim)

                # Optional L2 normalization (before averaging)
                if self.normalize:
                    features = torch.nn.functional.normalize(features, p=2, dim=1)

                # Split features back into left and right
                batch_size = len(batch_images)
                left_features = features[:batch_size]  # (B, embedding_dim)
                right_features = features[batch_size:]  # (B, embedding_dim)

                # Average the embeddings from both halves
                avg_features = (left_features + right_features) / 2.0  # (B, embedding_dim)

                # Move to CPU and collect
                all_embeddings.append(avg_features.cpu().numpy())

        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)

        self.log_to_terminal(f"Extracted and averaged embeddings with shape {embeddings.shape}")

        return embeddings
