"""SigLIP frozen feature extractor implementation."""
from dataclasses import dataclass
from typing import Any

import timm
import torch
from torch import nn

from src.modules.transformation.frozen_feature_extractor import FrozenFeatureExtractor


@dataclass
class FrozenSigLIP(FrozenFeatureExtractor):
    """Frozen SigLIP model for feature extraction.

    SigLIP is Google's improved version of CLIP, trained with a sigmoid loss
    instead of softmax. It provides strong visual representations.

    Available models (via timm):
    - vit_base_patch16_siglip_224: Base model (86M params, 768 dim)
    - vit_base_patch16_siglip_256: Base model with 256px images
    - vit_base_patch16_siglip_384: Base model with 384px images
    - vit_base_patch16_siglip_512: Base model with 512px images
    - vit_large_patch16_siglip_256: Large model (304M params, 1024 dim)
    - vit_large_patch16_siglip_384: Large model with 384px images

    The models expect images normalized with their own statistics (handled by timm).

    Parameters
    ----------
    model_name : str
        Name of the SigLIP model variant (timm model name)
    device : str
        Device to run inference on
    batch_size : int
        Batch size for inference
    normalize : bool
        Whether to L2 normalize embeddings
    use_fc_norm : bool
        Whether to use the FC norm layer output (if False, uses pre-norm features)
    """

    use_fc_norm: bool = True  # Use normalized features from the model

    def _load_model(self) -> nn.Module:
        """Load pretrained SigLIP model from timm.

        Returns
        -------
        nn.Module
            Pretrained SigLIP model
        """
        # Load from timm with pretrained weights
        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        return model

    def _preprocess_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SigLIP.

        timm models handle their own normalization via data_config,
        but we'll apply the standard preprocessing here.

        SigLIP models typically use ImageNet normalization.

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
        transforms = timm.data.create_transform(**data_config, is_training=False)

        # timm transforms expect PIL images or need to be applied differently
        # For simplicity, use the mean/std from data_config
        mean = torch.tensor(data_config['mean']).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(data_config['std']).view(1, 3, 1, 1).to(images.device)

        normalized = (images - mean) / std

        return normalized

    def _extract_features(self, model_output: Any) -> torch.Tensor:
        """Extract feature embeddings from SigLIP output.

        When created with num_classes=0, timm models return the pooled features
        directly before the classification head.

        Parameters
        ----------
        model_output : Any
            Output from SigLIP forward pass (pooled features)

        Returns
        -------
        torch.Tensor
            Feature embeddings in shape (B, embedding_dim)
        """
        # timm models with num_classes=0 return features directly
        # Shape is already (B, embedding_dim)
        return model_output
