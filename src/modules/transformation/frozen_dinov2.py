"""DINOv2 frozen feature extractor implementation."""
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.modules.transformation.frozen_feature_extractor import FrozenFeatureExtractor


@dataclass
class FrozenDinoV2(FrozenFeatureExtractor):
    """Frozen DINOv2 model for feature extraction.

    Uses Meta's DINOv2 vision transformer models pretrained via self-supervision.
    These models are excellent for extracting generic visual features.

    Available models:
    - dinov2_vits14: Small (21M params, 384 dim)
    - dinov2_vitb14: Base (86M params, 768 dim)
    - dinov2_vitl14: Large (300M params, 1024 dim)
    - dinov2_vitg14: Giant (1.1B params, 1536 dim)

    The models expect images normalized with ImageNet statistics.

    Parameters
    ----------
    model_name : str
        Name of the DINOv2 model variant
    device : str
        Device to run inference on
    batch_size : int
        Batch size for inference
    normalize : bool
        Whether to L2 normalize embeddings
    """

    def _load_model(self) -> nn.Module:
        """Load pretrained DINOv2 model from torch hub.

        Returns
        -------
        nn.Module
            Pretrained DINOv2 model
        """
        # Load from torch hub
        model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        return model

    def _preprocess_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for DINOv2.

        DINOv2 expects:
        - Images in (B, C, H, W) format
        - Normalized with ImageNet mean and std
        - Values in range [0, 1] before normalization

        Parameters
        ----------
        images : torch.Tensor
            Batch of images in (B, C, H, W) format, values in [0, 1]

        Returns
        -------
        torch.Tensor
            Preprocessed images
        """
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)

        normalized = (images - mean) / std

        return normalized

    def _extract_features(self, model_output: Any) -> torch.Tensor:
        """Extract CLS token embeddings from DINOv2 output.

        DINOv2's forward pass returns a dictionary with patch tokens and CLS token.
        We use the CLS token as the image-level embedding.

        Parameters
        ----------
        model_output : Any
            Output from DINOv2 forward pass (CLS token for standard forward)

        Returns
        -------
        torch.Tensor
            Feature embeddings in shape (B, embedding_dim)
        """
        # For DINOv2, the standard forward() returns the CLS token directly
        # Shape is already (B, embedding_dim)
        return model_output
