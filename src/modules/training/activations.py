import torch
import torch.nn as nn


class ReLUExpm1(nn.Module):
    """
    Activation function that applies ReLU, then expm1 (i.e., exp(x) - 1).
    """
    def forward(self, x):
        return torch.expm1(torch.relu(x))


class ClippedReLUExpm1(nn.Module):
    """
    Activation function that applies ReLU, then expm1 (i.e., exp(x) - 1), then clips to max_value.

    This prevents extremely large outputs while maintaining differentiability.
    Training data max is ~186, so we clip at 250 to allow some headroom for validation/test
    while preventing numerical instability.
    """
    def __init__(self, max_value: float = 250.0):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        # Apply ReLU then expm1
        out = torch.expm1(torch.relu(x))
        # Clip to max_value (still differentiable via torch.clamp)
        return torch.clamp(out, max=self.max_value)

    def __repr__(self):
        return f"{self.__class__.__name__}(max_value={self.max_value})"
