"""Data augmentation for training."""
from dataclasses import dataclass
from typing import Optional

import torch
import kornia as K


@dataclass
class Augmentation:
    """Augmentation wrapper supporting Kornia GPU augmentations.
    
    Applies augmentations to batches of images and labels on GPU.
    Images are expected in [0, 1] range.
    """
   
    kornia: Optional[K.augmentation.AugmentationSequential] = None

    def __call__(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment a batch of images.
        
        :param x: Batch of images to augment (B, C, H, W) in [0, 1] range
        :param y: Batch of targets to augment (B, num_targets)
        :param device: Device to use for augmentation
        :return: Augmented tensors of images (B, C, H, W) and targets (B, num_targets)
        """
        with torch.no_grad():
            # Apply Kornia augmentations (image only for regression)
            # Augmentations expect [0, 1] range
            if self.kornia:
                x = self.kornia(x)
            
            return x, y

