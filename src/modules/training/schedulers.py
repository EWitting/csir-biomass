"""Custom learning rate schedulers for RealMLP-TD and other models."""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosLogKScheduler(_LRScheduler):
    """Multi-cycle cosine learning rate schedule from RealMLP-TD paper.

    Formula: coslog_k(t) = 1/2 * (1 - cos(2π * log2(1 + (2^k - 1) * t)))

    where t ∈ [0, 1] is the normalized epoch (current_epoch / total_epochs).

    This creates k cycles with logarithmic spacing, providing multiple
    opportunities to escape local minima while allowing high learning rates
    in between.

    Args:
        optimizer: Wrapped optimizer
        total_epochs: Total number of training epochs
        num_cycles: Number of cycles (k in the paper, default 4)
        eta_min: Minimum learning rate (default 0)
        last_epoch: The index of last epoch (default -1)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        num_cycles: int = 4,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.num_cycles = num_cycles
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on coslog_k schedule."""
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        # Normalize epoch to [0, 1]
        t = self.last_epoch / self.total_epochs

        # coslog_k(t) = 1/2 * (1 - cos(2π * log2(1 + (2^k - 1) * t)))
        k = self.num_cycles
        log_term = math.log2(1 + (2**k - 1) * t)
        coslog_k = 0.5 * (1 - math.cos(2 * math.pi * log_term))

        return [
            self.eta_min + (base_lr - self.eta_min) * coslog_k
            for base_lr in self.base_lrs
        ]


class FlatCosScheduler:
    """Flat-cos schedule for dropout and weight decay from RealMLP-TD paper.

    Formula: flat_cos(t) = 1/2 * (1 + cos(π * (max(1, 2t) - 1)))

    where t ∈ [0, 1] is the normalized epoch.

    This starts at 1.0, stays flat until t=0.5, then cosine anneals to 0.

    Args:
        base_value: Base value (e.g., 0.1 for dropout, 0.02 for weight decay)
        total_epochs: Total number of training epochs
    """

    def __init__(self, base_value: float, total_epochs: int):
        self.base_value = base_value
        self.total_epochs = total_epochs

    def get_value(self, epoch: int) -> float:
        """Get the scheduled value for the current epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Scheduled value
        """
        # Normalize epoch to [0, 1]
        t = epoch / self.total_epochs

        # flat_cos(t) = 1/2 * (1 + cos(π * (max(1, 2t) - 1)))
        max_term = max(1.0, 2 * t)
        flat_cos = 0.5 * (1 + math.cos(math.pi * (max_term - 1)))

        return self.base_value * flat_cos
