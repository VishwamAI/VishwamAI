"""Router weight initialization functions for MoE layers."""
from typing import Optional
import math
import torch
import torch.nn as nn

def init_router_weights(
    router: nn.Module,
    num_experts: int,
    hidden_size: int,
    init_method: str = 'scaled_normal',
    init_std: float = 0.02,
    capacity_factor: float = 1.0,
    router_bias: bool = True,
    router_scale: Optional[float] = None
) -> None:
    """Initialize router module weights.
    
    Args:
        router: Router module to initialize
        num_experts: Number of experts
        hidden_size: Hidden dimension size
        init_method: Initialization method ['scaled_normal', 'jax', 'uniform']
        init_std: Standard deviation for normal initialization
        capacity_factor: Expert capacity multiplier
        router_bias: Whether to use router biases
        router_scale: Optional explicit scaling factor
    """
    if router_scale is None:
        # Default scale based on hidden size and number of experts
        router_scale = 1.0 / math.sqrt(hidden_size * num_experts)
    
    if init_method == 'scaled_normal':
        # Initialize router weights with scaled normal distribution
        if hasattr(router, 'weight') and router.weight is not None:
            nn.init.normal_(
                router.weight,
                mean=0.0,
                std=init_std * router_scale
            )
            
        if router_bias and hasattr(router, 'bias') and router.bias is not None:
            nn.init.zeros_(router.bias)
            
    elif init_method == 'jax':
        # JAX-style initialization (used in GShard, Switch Transformer)
        if hasattr(router, 'weight') and router.weight is not None:
            # Initialize close to zeros to start with roughly uniform routing
            std = 1.0 / math.sqrt(hidden_size)
            nn.init.trunc_normal_(
                router.weight,
                mean=0.0,
                std=std,
                a=-2.0 * std,
                b=2.0 * std
            )
            
        if router_bias and hasattr(router, 'bias') and router.bias is not None:
            nn.init.zeros_(router.bias)
            
    elif init_method == 'uniform':
        # Uniform initialization
        if hasattr(router, 'weight') and router.weight is not None:
            bound = router_scale / math.sqrt(3.0)
            nn.init.uniform_(router.weight, -bound, bound)
            
        if router_bias and hasattr(router, 'bias') and router.bias is not None:
            nn.init.zeros_(router.bias)
            
    else:
        raise ValueError(f"Unknown initialization method: {init_method}")
        
def init_router_temperature(
    router: nn.Module,
    init_value: float = 1.0,
    learnable: bool = True
) -> None:
    """Initialize router temperature parameter.
    
    Args:
        router: Router module
        init_value: Initial temperature value
        learnable: Whether temperature is a learnable parameter
    """
    if hasattr(router, 'temperature'):
        if learnable:
            # Initialize as learnable parameter
            router.temperature = nn.Parameter(torch.tensor(init_value))
        else:
            # Initialize as fixed buffer
            router.register_buffer('temperature', torch.tensor(init_value))
            
def init_router_noise(
    router: nn.Module,
    noise_type: str = 'gaussian',
    noise_std: float = 1.0
) -> None:
    """Initialize router noise parameters.
    
    Args:
        router: Router module
        noise_type: Type of noise ['gaussian', 'gumbel']
        noise_std: Standard deviation for Gaussian noise
    """
    if hasattr(router, 'noise_type'):
        router.noise_type = noise_type
        
    if hasattr(router, 'noise_std'):
        if isinstance(router.noise_std, nn.Parameter):
            # Initialize learnable noise std
            nn.init.constant_(router.noise_std, noise_std)
        else:
            # Set fixed noise std
            router.noise_std = noise_std
