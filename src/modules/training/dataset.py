"""Custom dataset with augmentation support."""
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.modules.training.augmentation import Augmentation


@dataclass
class AugmentedDataset(Dataset):
    """Dataset that applies augmentations and normalization during getitems.
    
    Images are expected in [0, 1] range and will be normalized with mean/std.
    """

    x: torch.Tensor
    y: torch.Tensor
    augmentation: Optional[Augmentation] = None
    device: torch.device = torch.device('cpu')
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet defaults
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __len__(self) -> int:
        """Get length of dataset.

        :return: Length of dataset
        """
        return len(self.x)
    
    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get multiple items from dataset.

        :param indices: List of indices to get
        :return: Tuple of (x, y) tensors for the batch
        """
        x = self.x[indices]
        y = self.y[indices]

        # Move to device first
        x = x.to(self.device)
        y = y.to(self.device)

        # Apply augmentations if provided (expects [0, 1] range)
        if self.augmentation:
            x, y = self.augmentation(x, y, self.device)
        
        # Always normalize with mean/std (after augmentations)
        mean_tensor = torch.tensor(self.mean, device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        std_tensor = torch.tensor(self.std, device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean_tensor) / std_tensor

        return x, y

