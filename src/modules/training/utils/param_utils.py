"""Utilities for parameter counting and inspection."""

import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, only count parameters with requires_grad=True

    Returns:
        Total number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_module_parameter_count(module: nn.Module) -> int:
    """Get parameter count for a specific module.

    Args:
        module: PyTorch module

    Returns:
        Number of parameters in the module
    """
    return sum(p.numel() for p in module.parameters())


def get_trainable_param_names(model: nn.Module) -> list[str]:
    """Get names of all trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        List of parameter names that have requires_grad=True
    """
    return [name for name, param in model.named_parameters() if param.requires_grad]
