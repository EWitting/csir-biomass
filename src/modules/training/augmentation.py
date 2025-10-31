"""Data augmentation for training."""
from dataclasses import dataclass
from typing import Optional

import torch
import kornia as K


@dataclass
class Augmentation:
    """Augmentation wrapper supporting Kornia GPU augmentations.
    
    Applies augmentations to batches of images and labels on GPU.
    """
   
    kornia: Optional[K.augmentation.AugmentationSequential] = None

    def __call__(self, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment a batch of images.
        
        :param x: Batch of images to augment (B, C, H, W)
        :param y: Batch of targets to augment (B, num_targets)
        :param device: Device to use for augmentation
        :return: Augmented tensors of images (B, C, H, W) and targets (B, num_targets)
        """
        with torch.no_grad():
            # Move to device
            x = x.to(device)
            y = y.to(device)

            # Apply Kornia augmentations (image only for regression)
            if self.kornia:
                x = self.kornia(x)
            
            return x, y

