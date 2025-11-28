"""Custom loss functions for biomass prediction."""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def _get_base_loss_repr(base_loss: nn.Module) -> str:
    """Get string representation of base loss with its parameters.
    
    Utility function to avoid code duplication in loss __repr__ methods.
    
    :param base_loss: The base loss module
    :return: String representation like "HuberLoss(delta=1.0, reduction='mean')"
    """
    base_loss_class = base_loss.__class__.__name__
    
    # Common attributes to check for various loss functions
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
    
    base_loss_params = []
    for attr in param_attrs:
        if hasattr(base_loss, attr):
            value = getattr(base_loss, attr)
            # Only include if not None and not a tensor buffer
            if value is not None and not isinstance(value, torch.Tensor):
                base_loss_params.append(f"{attr}={repr(value)}")
    
    # Build base_loss repr with its parameters
    if base_loss_params:
        return f"{base_loss_class}({', '.join(base_loss_params)})"
    else:
        return f"{base_loss_class}()"


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
        # Skip single-sample batches to avoid NaN from unbiased variance
        # (unbiased variance divides by N-1, which is 0 when N=1)
        if y_true.shape[0] <= 1:
            return

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
        base_loss_repr = _get_base_loss_repr(self.base_loss)
        weights_list = self.weight_tensor.tolist()
        
        return (
            f"{self.__class__.__name__}(\n"
            f"  base_loss={base_loss_repr},\n"
            f"  weights={weights_list},\n"
            f"  scale_by_variance={self.scale_by_variance},\n"
            f"  variance_momentum={self.variance_momentum}\n"
            f")"
        )


class TransformedLoss(nn.Module):
    """Transformed loss wrapper that applies a transformation before computing loss.
    
    Applies the same transformation to both predictions and labels before computing loss.
    Useful for training in transformed space (e.g., log space) without modifying the model.
    
    Example: Using log1p transformation with MSE for training in log space:
        transform = torch.log1p
        base_loss = torch.nn.MSELoss()
        loss_fn = TransformedLoss(transform, base_loss)
    """
    
    def __init__(
        self,
        transform: callable,
        base_loss: nn.Module,
    ) -> None:
        """Initialize the transformed loss.
        
        :param transform: Transformation function to apply before computing loss
        :param base_loss: Base loss function (e.g., MSELoss, L1Loss)
        """
        super().__init__()
        
        self.transform = transform
        self.base_loss = base_loss
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss in transformed space.
        
        :param y_pred: Predictions (B, num_targets)
        :param y_true: Ground truth (B, num_targets)
        :return: Loss scalar
        """
        # Apply transformation to both predictions and targets
        y_pred_transformed = self.transform(y_pred)
        y_true_transformed = self.transform(y_true)
        
        # Compute loss in transformed space
        return self.base_loss(y_pred_transformed, y_true_transformed)
    
    def __repr__(self) -> str:
        """Custom repr for debugging and reproducibility."""
        base_loss_repr = _get_base_loss_repr(self.base_loss)
        
        # Try to get a nice string for the transform function
        if hasattr(self.transform, '__name__'):
            transform_str = self.transform.__name__
        elif hasattr(self.transform, '__class__'):
            transform_str = self.transform.__class__.__name__
        else:
            transform_str = str(self.transform)
        
        return (
            f"{self.__class__.__name__}(\n"
            f"  transform={transform_str},\n"
            f"  base_loss={base_loss_repr}\n"
            f")"
        )


class GlobalWeightedLoss(nn.Module):
    """Global Weighted Loss for multi-output regression.
    
    Matches the global weighted R² metric where all (sample, target) pairs
    are combined into one list with per-row weights.
    
    This is mathematically different from WeightedLoss:
    - WeightedLoss: computes loss per target, then weights
    - GlobalWeightedLoss: combines all into one list with row weights
    
    For R² optimization, this loss approximates minimizing:
    1 - global_weighted_R² = SS_res / SS_tot (with global weighting)
    
    Optionally scales by global weighted variance to align loss magnitude with R² optimization.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        weights: list[float] | None = None,
        scale_by_variance: bool = False,
        variance_momentum: float = 0.1,
    ) -> None:
        """Initialize the global weighted loss.
        
        :param base_loss: Base loss function (e.g., MSELoss, HuberLoss)
        :param weights: Per-target weights (defaults to competition weights)
        :param scale_by_variance: If True, scales loss by 1/global_var to align magnitude with R²
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
        
        # For variance scaling - accumulated via EWMA (global weighted variance)
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)
        self._stats_initialized = False
    
    def _update_global_stats(self, y_true: torch.Tensor) -> None:
        """Update global running statistics using EWMA.

        Computes global weighted mean and variance across all (sample, target) pairs.

        :param y_true: Ground truth (B, num_targets)
        """
        # Skip single-sample batches to avoid potential numerical issues
        if y_true.shape[0] <= 1:
            return

        batch_size, num_targets = y_true.shape
        
        # Flatten targets
        y_true_flat = y_true.reshape(-1)
        
        # Create per-row weights (repeat for each sample)
        weights_flat = self.weight_tensor.repeat(batch_size).to(y_true.device)
        
        # Compute weighted global mean and variance
        weight_sum = torch.sum(weights_flat)
        batch_mean = torch.sum(weights_flat * y_true_flat) / weight_sum
        batch_var = torch.sum(weights_flat * (y_true_flat - batch_mean) ** 2) / weight_sum
        
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
        """Compute global weighted loss.
        
        All (sample, target) pairs are flattened into one list,
        each weighted by its target type.
        
        :param y_pred: Predictions (B, num_targets)
        :param y_true: Ground truth (B, num_targets)
        :return: Global weighted loss scalar
        """
        # Update running statistics (only during training)
        if self.scale_by_variance and self.training:
            self._update_global_stats(y_true)
        
        batch_size, num_targets = y_pred.shape
        
        # Flatten predictions and targets
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)
        
        # Create per-row weights (repeat for each sample)
        weights_flat = self.weight_tensor.repeat(batch_size)
        
        # Ensure weights are on the same device
        weights_flat = weights_flat.to(y_pred.device)
        
        # Apply variance scaling if enabled
        if self.scale_by_variance and self._stats_initialized:
            # Scale by global std to align loss magnitude with R² optimization
            std = torch.sqrt(self.running_var + 1e-8)  # Add epsilon for numerical stability
            y_pred_flat = y_pred_flat / std
            y_true_flat = y_true_flat / std
        
        # Compute loss element-wise
        # Note: base_loss should have reduction='none' for this to work properly
        if hasattr(self.base_loss, 'reduction'):
            original_reduction = self.base_loss.reduction
            self.base_loss.reduction = 'none'
            losses = self.base_loss(y_pred_flat, y_true_flat)
            self.base_loss.reduction = original_reduction
        else:
            # If base_loss doesn't have reduction, compute manually
            losses = (y_pred_flat - y_true_flat) ** 2  # Default to MSE
        
        # Apply weights and sum
        weighted_loss = torch.sum(losses * weights_flat) / torch.sum(weights_flat)
        
        return weighted_loss
    
    def __repr__(self) -> str:
        """Custom repr for debugging and reproducibility."""
        base_loss_repr = _get_base_loss_repr(self.base_loss)
        weights_list = self.weight_tensor.tolist()
        
        return (
            f"{self.__class__.__name__}(\n"
            f"  base_loss={base_loss_repr},\n"
            f"  weights={weights_list},\n"
            f"  scale_by_variance={self.scale_by_variance},\n"
            f"  variance_momentum={self.variance_momentum}\n"
            f")"
        )

class StableLogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        # Naive log(cosh(x)) overflows for large x.
        # Stable formula: x + softplus(-2x) - log(2)
        # This works because log(cosh(x)) ~= abs(x) - log(2) for large x
        return torch.mean(diff + F.softplus(-2. * diff) - torch.log(torch.tensor(2.0)))