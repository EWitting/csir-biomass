"""Custom dataset with augmentation support."""
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.modules.training.augmentation import Augmentation


@dataclass
class AugmentedDataset(Dataset):
    """Dataset that applies augmentations during getitems."""

    x: torch.Tensor
    y: torch.Tensor
    augmentation: Optional[Augmentation] = None
    device: torch.device = torch.device('cpu')

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

        if self.augmentation:
            x, y = self.augmentation(x, y, self.device)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y

