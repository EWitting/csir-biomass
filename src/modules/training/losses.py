"""Custom loss functions for biomass prediction."""
import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    """Weighted loss wrapper for multi-output regression.
    
    Applies per-target weights to any base loss function.
    Optionally scales by target variance to align with R² optimization.
    
    Mathematical justification:
    - Competition metric: weighted_R² = Σ w_i * (1 - SS_res_i/SS_tot_i)
    - To maximize this, minimize: Σ w_i * SS_res_i/SS_tot_i
    - Since SS_tot_i ∝ Var(y_i), this equals: Σ w_i * MSE_i/Var(y_i)
    - With variance scaling, MSE loss → exactly equivalent to R² optimization
    - With Huber loss, maintains quadratic property for small residuals
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        weights: list[float] | None = None,
        scale_by_variance: bool = False,
        variance_momentum: float = 0.1,
    ) -> None:
        """Initialize the weighted loss.
        
        :param base_loss: Base loss function (e.g., MSELoss, HuberLoss)
        :param weights: Per-target weights (defaults to competition weights)
        :param scale_by_variance: If True, scales loss by 1/std² to match R² optimization
        :param variance_momentum: Momentum for EWMA of variance (0.1 = 10% new, 90% old)
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.scale_by_variance = scale_by_variance
        self.variance_momentum = variance_momentum
        
        if weights is None:
            # Default: competition weights
            # Target order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
            weights = [0.1, 0.1, 0.1, 0.2, 0.5]
        
        # Convert to tensor for efficiency
        self.register_buffer('weight_tensor', torch.tensor(weights, dtype=torch.float32))
        
        # For variance scaling - accumulated via EWMA
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)
        self._stats_initialized = False
    
    def _update_stats(self, y_true: torch.Tensor) -> None:
        """Update running statistics using EWMA.
        
        Accumulates mean and variance across batches using exponential moving average.
        This provides more robust estimates than a single batch.
        
        :param y_true: Ground truth (B, num_targets)
        """
        # Compute batch statistics
        batch_mean = torch.mean(y_true, dim=0)
        batch_var = torch.var(y_true, dim=0, unbiased=True)
        
        if not self._stats_initialized:
            # Initialize with first batch
            self.running_mean = batch_mean
            self.running_var = batch_var
            self._stats_initialized = True
        else:
            # Update using EWMA: new = momentum * batch + (1 - momentum) * old
            self.running_mean = (
                self.variance_momentum * batch_mean +
                (1 - self.variance_momentum) * self.running_mean
            )
            self.running_var = (
                self.variance_momentum * batch_var +
                (1 - self.variance_momentum) * self.running_var
            )
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss.

        :param y_pred: Predictions (B, num_targets)
        :param y_true: Ground truth (B, num_targets)
        :return: Weighted loss scalar
        """
        # Update running statistics (only during training)
        if self.scale_by_variance and self.training:
            self._update_stats(y_true)
        
        # Compute loss per target
        losses = []
        for i in range(y_pred.shape[1]):
            if self.scale_by_variance and self._stats_initialized:
                # Normalize predictions and targets by std
                std = torch.sqrt(self.running_var[i] + 1e-8)  # Add epsilon for numerical stability
                y_pred_norm = y_pred[:, i] / std
                y_true_norm = y_true[:, i] / std
                target_loss = self.base_loss(y_pred_norm, y_true_norm)
            else:
                target_loss = self.base_loss(y_pred[:, i], y_true[:, i])
            
            losses.append(target_loss)
        
        # Stack and weight
        losses = torch.stack(losses)
        
        # Ensure weight_tensor is on the same device as losses
        weight_tensor = self.weight_tensor.to(losses.device)
        weighted_loss = torch.sum(losses * weight_tensor)
        
        return weighted_loss
    
    def __repr__(self) -> str:
        """Custom repr that includes all parameters that influence training behavior.
        
        Ugly, but quick fix for to make the model hashing and reproducibility work properly.
        """
        # Get base_loss parameters by inspecting its attributes
        base_loss_params = []
        base_loss_class = self.base_loss.__class__.__name__
        
        # Common attributes to check for various loss functions
        # These are the parameters that typically influence loss behavior
        param_attrs = [
            'delta',  # HuberLoss
            'reduction',  # Most losses
            'weight',  # Some losses
            'size_average',  # Legacy losses
            'ignore_index',  # CrossEntropyLoss
            'label_smoothing',  # CrossEntropyLoss
            'beta',  # SmoothL1Loss
            'alpha',  # FocalLoss (if used)
            'gamma',  # FocalLoss (if used)
        ]
        
        for attr in param_attrs:
            if hasattr(self.base_loss, attr):
                value = getattr(self.base_loss, attr)
                # Only include if not None and not a tensor buffer
                if value is not None and not isinstance(value, torch.Tensor):
                    base_loss_params.append(f"{attr}={repr(value)}")
        
        # Build base_loss repr with its parameters
        if base_loss_params:
            base_loss_repr = f"{base_loss_class}({', '.join(base_loss_params)})"
        else:
            base_loss_repr = f"{base_loss_class}()"
        
        # Convert weight tensor to list for readable repr
        weights_list = self.weight_tensor.tolist()
        
        # Build complete repr
        return (
            f"{self.__class__.__name__}(\n"
            f"  base_loss={base_loss_repr},\n"
            f"  weights={weights_list},\n"
            f"  scale_by_variance={self.scale_by_variance},\n"
            f"  variance_momentum={self.variance_momentum}\n"
            f")"
        )

