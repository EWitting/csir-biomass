import torch
import torch.nn as nn

class ReLUExpm1(nn.Module):
    """
    Activation function that applies ReLU, then expm1 (i.e., exp(x) - 1).
    """
    def forward(self, x):
        return torch.expm1(torch.relu(x))
