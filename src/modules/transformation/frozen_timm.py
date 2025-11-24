"""Generic frozen feature extractor for any timm model."""
from dataclasses import dataclass
from typing import Any

import timm
import torch
from torch import nn

from src.modules.transformation.frozen_feature_extractor import FrozenFeatureExtractor


@dataclass
class FrozenTimm(FrozenFeatureExtractor):
    """Generic frozen model for feature extraction from timm library.

    Works with any timm model including:
    - EfficientNet variants (efficientnet_b0-b7, efficientnetv2_*)
    - ResNet variants (resnet18, resnet50, resnet152, etc.)
    - Vision Transformers (vit_base_patch16_224, etc.)
    - EVA-02 models (eva02_base_patch14_224, etc.)
    - SigLIP models (vit_base_patch16_siglip_*, etc.)
    - ConvNeXt models (convnext_tiny, convnext_base, etc.)
    - And 1000+ other architectures

    The model is loaded with num_classes=0 to extract features before
    the classification head.

    Parameters
    ----------
    model_name : str
        Name of the timm model (e.g., "efficientnet_b3", "resnet50")
    device : str
        Device to run inference on
    batch_size : int
        Batch size for inference
    normalize : bool
        Whether to L2 normalize embeddings
    target_size : int | None
        Optional target size to resize images to
    pretrained : bool
        Whether to load pretrained weights (default: True)

    Example
    -------
    ```yaml
    - _target_: src.modules.transformation.frozen_timm.FrozenTimm
      model_name: "efficientnet_b3"
      batch_size: 32
      normalize: true
    ```
    """

    pretrained: bool = True  # Load pretrained weights

    def _load_model(self) -> nn.Module:
        """Load pretrained model from timm.

        Returns
        -------
        nn.Module
            Pretrained timm model with feature extraction head
        """
        # Load from timm with pretrained weights
        # num_classes=0 returns features before classification head
        model = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=0  # Return features only
        )
        return model

    def _preprocess_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for the timm model.

        Uses the model's data config for proper normalization.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images in (B, C, H, W) format, values in [0, 1]

        Returns
        -------
        torch.Tensor
            Preprocessed images
        """
        # Get the data config from the model for proper normalization
        data_config = timm.data.resolve_model_data_config(self.model)

        # Extract mean and std for normalization
        mean = torch.tensor(data_config['mean']).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(data_config['std']).view(1, 3, 1, 1).to(images.device)

        # Normalize
        normalized = (images - mean) / std

        return normalized

    def _extract_features(self, model_output: Any) -> torch.Tensor:
        """Extract feature embeddings from model output.

        timm models with num_classes=0 return pooled features directly.

        Parameters
        ----------
        model_output : Any
            Output from model forward pass (pooled features)

        Returns
        -------
        torch.Tensor
            Feature embeddings in shape (B, embedding_dim)
        """
        # timm models with num_classes=0 return features directly
        # Shape is already (B, embedding_dim)
        return model_output
