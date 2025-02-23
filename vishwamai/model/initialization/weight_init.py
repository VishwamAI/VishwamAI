"""Weight initialization utilities for neural network modules."""

import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn


def _calculate_fan_in_and_fan_out(
    shape: Union[Tuple[int, ...], torch.Size]
) -> Tuple[int, int]:
    """Calculate fan-in and fan-out of a weight tensor.
    
    Args:
        shape: Shape of weight tensor
        
    Returns:
        Tuple of (fan_in, fan_out)
    """
    dimensions = len(shape)
    
    if dimensions < 2:
        raise ValueError("Fan in and fan out can't be computed for tensor with fewer than 2 dimensions")
        
    if dimensions == 2:  # Linear
        fan_in, fan_out = shape[1], shape[0]
    else:  # Convolution
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        for s in shape[2:]:
            receptive_field_size *= s
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        
    return fan_in, fan_out


def normal_init_(
    module: nn.Module,
    mean: float = 0.0,
    std: float = 0.02,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> None:
    """Initialize weights using normal distribution.
    
    Args:
        module: Module to initialize
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
        min_val: Minimum value for clamping (optional)
        max_val: Maximum value for clamping (optional)
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
        if min_val is not None or max_val is not None:
            with torch.no_grad():
                module.weight.clamp_(min=min_val, max=max_val)
                
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


def xavier_uniform_init_(
    module: nn.Module,
    gain: float = 1.0
) -> None:
    """Initialize weights using Xavier/Glorot uniform initialization.
    
    Args:
        module: Module to initialize
        gain: Scaling factor
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.xavier_uniform_(module.weight, gain=gain)
        
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


def kaiming_uniform_init_(
    module: nn.Module,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu"
) -> None:
    """Initialize weights using Kaiming/He uniform initialization.
    
    Args:
        module: Module to initialize
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Name of nonlinearity function
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
        
    if hasattr(module, "bias") and module.bias is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(module.weight.shape)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(module.bias, -bound, bound)


def scaled_init_(
    module: nn.Module,
    scale: float = 1.0
) -> None:
    """Initialize weights using scaled initialization.
    
    Args:
        module: Module to initialize
        scale: Scaling factor
    """
    if hasattr(module, "weight") and module.weight is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(module.weight.shape)
        std = math.sqrt(scale / fan_in)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


def initialize_linear_layer(
    layer: nn.Linear,
    init_method: str = "normal",
    **kwargs
) -> None:
    """Initialize linear layer with specified method.
    
    Args:
        layer: Linear layer to initialize
        init_method: Initialization method
        **kwargs: Additional initialization arguments
    """
    initializers = {
        "normal": normal_init_,
        "xavier_uniform": xavier_uniform_init_,
        "kaiming_uniform": kaiming_uniform_init_,
        "scaled": scaled_init_
    }
    
    if init_method not in initializers:
        raise ValueError(f"Unknown initialization method: {init_method}")
        
    initializers[init_method](layer, **kwargs)


def initialize_embedding_layer(
    layer: nn.Embedding,
    init_method: str = "normal",
    **kwargs
) -> None:
    """Initialize embedding layer with specified method.
    
    Args:
        layer: Embedding layer to initialize
        init_method: Initialization method
        **kwargs: Additional initialization arguments
    """
    if init_method == "normal":
        std = kwargs.get("std", 0.02)
        nn.init.normal_(layer.weight, mean=0.0, std=std)
    elif init_method == "xavier_uniform":
        nn.init.xavier_uniform_(layer.weight)
    else:
        raise ValueError(f"Unsupported embedding initialization method: {init_method}")


def initialize_layer_norm(
    layer: nn.LayerNorm,
    epsilon: float = 1e-5
) -> None:
    """Initialize layer normalization.
    
    Args:
        layer: LayerNorm layer to initialize
        epsilon: Small constant for numerical stability
    """
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.ones_(layer.weight)
        
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
        
    layer.eps = epsilon


def reinitialize_model(
    model: nn.Module,
    init_method: str = "normal",
    **kwargs
) -> None:
    """Reinitialize all parameters in a model.
    
    Args:
        model: Model to reinitialize
        init_method: Initialization method
        **kwargs: Additional initialization arguments
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            initialize_linear_layer(module, init_method, **kwargs)
        elif isinstance(module, nn.Embedding):
            initialize_embedding_layer(module, init_method, **kwargs)
        elif isinstance(module, nn.LayerNorm):
            initialize_layer_norm(module, kwargs.get("epsilon", 1e-5))


__all__ = [
    "normal_init_",
    "xavier_uniform_init_",
    "kaiming_uniform_init_",
    "scaled_init_",
    "initialize_linear_layer",
    "initialize_embedding_layer",
    "initialize_layer_norm",
    "reinitialize_model",
    "_calculate_fan_in_and_fan_out",
]
