"""DINOv3 frozen feature extractor implementation."""
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel

from src.modules.transformation.frozen_feature_extractor import FrozenFeatureExtractor


@dataclass
class FrozenDinoV3(FrozenFeatureExtractor):
    """Frozen DINOv3 model for feature extraction.

    Uses Meta's DINOv3 vision transformer models - the latest generation
    pretrained via self-supervision with improved performance over DINOv2.

    DINOv3 models handle flexible image sizes (any multiple of patch size 16)
    and are specifically designed to work well at higher resolutions.

    Available models from Hugging Face:
    - facebook/dinov3-vits16-pretrain-lvd1689m: Small (21M params, 384 dim)
    - facebook/dinov3-vits-plus16-pretrain-lvd1689m: Small+ (29M params, 384 dim)
    - facebook/dinov3-vitb16-pretrain-lvd1689m: Base (86M params, 768 dim)
    - facebook/dinov3-vitl16-pretrain-lvd1689m: Large (300M params, 1024 dim)
    - facebook/dinov3-vith-plus16-pretrain-lvd1689m: Huge+ (840M params, 1280 dim)
    - facebook/dinov3-vit7b16-pretrain-lvd1689m: Giant (6.7B params, 4096 dim)

    The models expect images with flexible sizes (multiples of 16).
    Image preprocessing is handled automatically by AutoImageProcessor.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier for DINOv3 variant
    device : str
        Device to run inference on
    batch_size : int
        Batch size for inference
    normalize : bool
        Whether to L2 normalize embeddings
    """

    def _load_model(self) -> nn.Module:
        """Load pretrained DINOv3 model from Hugging Face.

        Returns
        -------
        nn.Module
            Pretrained DINOv3 model
        """
        # Load processor and model from Hugging Face
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)

        return model

    def _preprocess_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for DINOv3.

        DINOv3 uses AutoImageProcessor which handles normalization.
        Images should be in (B, C, H, W) format with values in [0, 1].

        Parameters
        ----------
        images : torch.Tensor
            Batch of images in (B, C, H, W) format, values in [0, 1]

        Returns
        -------
        torch.Tensor
            Preprocessed images
        """
        # Convert from [0, 1] to [0, 255] as expected by processor
        images_255 = (images * 255).to(torch.uint8)

        # Move to CPU for processor (it expects CPU tensors or PIL images)
        images_cpu = images_255.cpu()

        # Process with AutoImageProcessor
        # Note: processor expects list of images or batched tensor
        inputs = self.processor(images=images_cpu, return_tensors="pt")

        # Return the pixel values (already normalized)
        return inputs['pixel_values']

    def _extract_features(self, model_output: Any) -> torch.Tensor:
        """Extract pooled CLS token embeddings from DINOv3 output.

        DINOv3's AutoModel returns a BaseModelOutputWithPooling object.
        We use the pooler_output which is the CLS token embedding.

        Parameters
        ----------
        model_output : Any
            Output from DINOv3 forward pass (BaseModelOutputWithPooling)

        Returns
        -------
        torch.Tensor
            Feature embeddings in shape (B, embedding_dim)
        """
        # Extract pooled output (CLS token)
        # pooler_output is already (B, embedding_dim)
        return model_output.pooler_output
