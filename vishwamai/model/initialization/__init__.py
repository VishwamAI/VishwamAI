"""Initialization utilities for model components."""

import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from .weight_init import (
    normal_init_,
    xavier_uniform_init_,
    kaiming_uniform_init_,
    scaled_init_,
    _calculate_fan_in_and_fan_out
)

from .expert_init import (
    initialize_expert_weights,
    initialize_expert_biases,
    initialize_expert_layer_norm
)

from .router_init import (
    initialize_router_weights,
    initialize_router_gates,
    initialize_router_load_balancing
)

def initialize_weights(
    module: nn.Module,
    method: str = "normal",
    mean: float = 0.0,
    std: float = 0.02,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    **kwargs
) -> None:
    """Initialize weights of a module using specified method.
    
    Args:
        module: Module to initialize
        method: Initialization method ('normal', 'xavier_uniform', 'kaiming_uniform', 'scaled')
        mean: Mean for normal initialization
        std: Standard deviation for normal initialization
        min_val: Minimum value for clamping (optional)
        max_val: Maximum value for clamping (optional)
        **kwargs: Additional arguments passed to initialization functions
    """
    if method == "normal":
        normal_init_(module, mean=mean, std=std, min_val=min_val, max_val=max_val)
    elif method == "xavier_uniform":
        xavier_uniform_init_(module, gain=kwargs.get("gain", 1.0))
    elif method == "kaiming_uniform":
        kaiming_uniform_init_(module, mode=kwargs.get("mode", "fan_in"), nonlinearity=kwargs.get("nonlinearity", "leaky_relu"))
    elif method == "scaled":
        scaled_init_(module, scale=kwargs.get("scale", 1.0))
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def initialize_experts(
    expert_layer: nn.Module,
    num_experts: int,
    hidden_size: int,
    expert_size: int,
    init_method: str = "normal",
    **kwargs
) -> None:
    """Initialize expert layers with specified method.
    
    Args:
        expert_layer: Expert layer module
        num_experts: Number of experts
        hidden_size: Hidden dimension size
        expert_size: Expert FFN dimension size
        init_method: Weight initialization method
        **kwargs: Additional initialization arguments
    """
    # Initialize expert weights
    initialize_expert_weights(
        expert_layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        expert_size=expert_size,
        method=init_method,
        **kwargs
    )
    
    # Initialize expert biases
    initialize_expert_biases(expert_layer)
    
    # Initialize layer normalization
    if hasattr(expert_layer, "layer_norm"):
        initialize_expert_layer_norm(expert_layer.layer_norm)

def initialize_router(
    router: nn.Module,
    num_experts: int,
    hidden_size: int,
    init_method: str = "normal",
    jitter: Optional[float] = 0.1,
    **kwargs
) -> None:
    """Initialize router components.
    
    Args:
        router: Router module
        num_experts: Number of experts
        hidden_size: Hidden dimension size
        init_method: Weight initialization method
        jitter: Noise scale for router weights
        **kwargs: Additional initialization arguments
    """
    # Initialize router weights
    initialize_router_weights(
        router,
        num_experts=num_experts,
        hidden_size=hidden_size,
        method=init_method,
        jitter=jitter,
        **kwargs
    )
    
    # Initialize gating mechanism
    initialize_router_gates(router)
    
    # Initialize load balancing components
    if hasattr(router, "load_balancing"):
        initialize_router_load_balancing(router.load_balancing)

def get_initializer(method: str):
    """Get initialization function by name.
    
    Args:
        method: Name of initialization method
        
    Returns:
        Callable initialization function
    """
    initializers = {
        "normal": normal_init_,
        "xavier_uniform": xavier_uniform_init_,
        "kaiming_uniform": kaiming_uniform_init_,
        "scaled": scaled_init_
    }
    
    if method not in initializers:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return initializers[method]

def compute_fans(
    shape: Union[Tuple[int, ...], torch.Size]
) -> Tuple[int, int]:
    """Compute fan-in and fan-out of a weight tensor.
    
    Args:
        shape: Shape of weight tensor
        
    Returns:
        Tuple of (fan_in, fan_out)
    """
    return _calculate_fan_in_and_fan_out(shape)

__all__ = [
    # Main initialization functions
    "initialize_weights",
    "initialize_experts",
    "initialize_router",
    "get_initializer",
    "compute_fans",
    
    # Weight initialization methods
    "normal_init_",
    "xavier_uniform_init_",
    "kaiming_uniform_init_",
    "scaled_init_",
    
    # Expert initialization
    "initialize_expert_weights",
    "initialize_expert_biases",
    "initialize_expert_layer_norm",
    
    # Router initialization
    "initialize_router_weights",
    "initialize_router_gates",
    "initialize_router_load_balancing",
]
