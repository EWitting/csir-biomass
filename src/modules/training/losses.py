"""Custom loss functions for biomass prediction."""
import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    """Weighted loss wrapper for multi-output regression.
    
    Applies per-target weights to any base loss function.
    Useful when some targets are more important than others.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        weights: list[float] | None = None,
    ) -> None:
        """Initialize the weighted loss.
        
        :param base_loss: Base loss function (e.g., MSELoss, HuberLoss)
        :param weights: Per-target weights (defaults to competition weights)
        """
        super().__init__()
        
        self.base_loss = base_loss
        
        if weights is None:
            # Default: competition weights
            # Target order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
            weights = [0.1, 0.1, 0.1, 0.2, 0.5]
        
        # Convert to tensor for efficiency
        self.register_buffer('weight_tensor', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss.

        :param y_pred: Predictions (B, num_targets)
        :param y_true: Ground truth (B, num_targets)
        :return: Weighted loss scalar
        """
        # Compute loss per target
        losses = []
        for i in range(y_pred.shape[1]):
            target_loss = self.base_loss(y_pred[:, i], y_true[:, i])
            losses.append(target_loss)
        
        # Stack and weight
        losses = torch.stack(losses)
        
        # Ensure weight_tensor is on the same device as losses
        weight_tensor = self.weight_tensor.to(losses.device)
        weighted_loss = torch.sum(losses * weight_tensor)
        
        return weighted_loss

