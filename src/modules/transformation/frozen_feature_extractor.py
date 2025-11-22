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
        self.log_to_terminal(f"Extracting features from {len(data)} images using {self.model_name}")

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
