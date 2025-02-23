"""Initialization utilities for router networks in MoE layers."""

import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from .weight_init import (
    normal_init_,
    xavier_uniform_init_,
    kaiming_uniform_init_,
    scaled_init_
)

def initialize_router_weights(
    router: nn.Module,
    num_experts: int,
    hidden_size: int,
    method: str = "normal",
    jitter: Optional[float] = 0.1,
    **kwargs
) -> None:
    """Initialize router network weights.
    
    Args:
        router: Router module
        num_experts: Number of experts
        hidden_size: Hidden dimension size
        method: Weight initialization method
        jitter: Noise scale for router weights
        **kwargs: Additional initialization arguments
    """
    if hasattr(router, "weight"):
        # Initialize routing weights
        if method == "normal":
            std = kwargs.get("std", 1.0 / math.sqrt(hidden_size))
            normal_init_(router, std=std)
        elif method == "xavier_uniform":
            gain = kwargs.get("gain", 1.0 / math.sqrt(num_experts))
            xavier_uniform_init_(router, gain=gain)
        elif method == "kaiming_uniform":
            mode = kwargs.get("mode", "fan_in")
            nonlinearity = kwargs.get("nonlinearity", "linear")
            kaiming_uniform_init_(router, mode=mode, nonlinearity=nonlinearity)
        elif method == "scaled":
            scale = kwargs.get("scale", 1.0 / num_experts)
            scaled_init_(router, scale=scale)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
            
        # Add jitter to break symmetry
        if jitter and jitter > 0:
            with torch.no_grad():
                noise = torch.randn_like(router.weight) * jitter
                router.weight.add_(noise)

def initialize_router_gates(
    router: nn.Module,
    init_type: str = "uniform",
    **kwargs
) -> None:
    """Initialize router gating components.
    
    Args:
        router: Router module
        init_type: Type of gate initialization
        **kwargs: Additional initialization arguments
    """
    if hasattr(router, "gates"):
        if init_type == "uniform":
            a = kwargs.get("a", -0.05)
            b = kwargs.get("b", 0.05)
            nn.init.uniform_(router.gates, a=a, b=b)
        elif init_type == "normal":
            std = kwargs.get("std", 0.02)
            nn.init.normal_(router.gates, mean=0.0, std=std)
        elif init_type == "zero":
            nn.init.zeros_(router.gates)
        else:
            raise ValueError(f"Unknown gate initialization type: {init_type}")
            
def initialize_router_load_balancing(
    load_balancing: nn.Module,
    num_experts: int,
    init_value: float = 1.0
) -> None:
    """Initialize router load balancing components.
    
    Args:
        load_balancing: Load balancing module
        num_experts: Number of experts
        init_value: Initial value for importance weights
    """
    if hasattr(load_balancing, "importance"):
        nn.init.constant_(load_balancing.importance, init_value)
        
    if hasattr(load_balancing, "capacity_factor"):
        load_balancing.capacity_factor = 1.0

def initialize_router_temperature(
    router: nn.Module,
    init_temp: float = 1.0,
    min_temp: float = 0.1,
    max_temp: float = 10.0
) -> None:
    """Initialize router temperature parameter.
    
    Args:
        router: Router module
        init_temp: Initial temperature value
        min_temp: Minimum temperature value
        max_temp: Maximum temperature value
    """
    if hasattr(router, "temperature"):
        router.temperature = nn.Parameter(
            torch.tensor(init_temp),
            requires_grad=True
        )
        router.min_temp = min_temp
        router.max_temp = max_temp

def initialize_router_noise(
    router: nn.Module,
    noise_type: str = "gaussian",
    noise_scale: float = 1.0
) -> None:
    """Initialize router noise parameters.
    
    Args:
        router: Router module
        noise_type: Type of noise ('gaussian' or 'gumbel')
        noise_scale: Scale of noise
    """
    if noise_type not in ["gaussian", "gumbel"]:
        raise ValueError(f"Unknown noise type: {noise_type}")
        
    router.noise_type = noise_type
    router.noise_scale = noise_scale

def initialize_auxiliary_loss(
    router: nn.Module,
    z_loss_weight: float = 0.1,
    load_loss_weight: float = 0.1
) -> None:
    """Initialize router auxiliary loss components.
    
    Args:
        router: Router module
        z_loss_weight: Weight for router z-loss
        load_loss_weight: Weight for load balancing loss
    """
    router.z_loss_weight = z_loss_weight
    router.load_loss_weight = load_loss_weight

class RouterInitializer:
    """Helper class for router initialization."""
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        init_method: str = "normal",
        gate_init: str = "uniform",
        jitter: float = 0.1,
        init_temp: float = 1.0,
        noise_type: str = "gaussian",
        noise_scale: float = 1.0,
        z_loss_weight: float = 0.1,
        load_loss_weight: float = 0.1,
        **kwargs
    ):
        """Initialize router initializer.
        
        Args:
            num_experts: Number of experts
            hidden_size: Hidden dimension size
            init_method: Weight initialization method
            gate_init: Gate initialization type
            jitter: Noise scale for router weights
            init_temp: Initial temperature value
            noise_type: Type of noise
            noise_scale: Scale of noise
            z_loss_weight: Weight for router z-loss
            load_loss_weight: Weight for load balancing loss
            **kwargs: Additional initialization arguments
        """
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.gate_init = gate_init
        self.jitter = jitter
        self.init_temp = init_temp
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.z_loss_weight = z_loss_weight
        self.load_loss_weight = load_loss_weight
        self.kwargs = kwargs
        
    def __call__(self, router: nn.Module) -> None:
        """Initialize router components.
        
        Args:
            router: Router to initialize
        """
        # Initialize routing weights
        initialize_router_weights(
            router,
            self.num_experts,
            self.hidden_size,
            self.init_method,
            self.jitter,
            **self.kwargs
        )
        
        # Initialize gates
        initialize_router_gates(router, self.gate_init, **self.kwargs)
        
        # Initialize load balancing
        if hasattr(router, "load_balancing"):
            initialize_router_load_balancing(router.load_balancing, self.num_experts)
            
        # Initialize temperature
        initialize_router_temperature(router, self.init_temp)
        
        # Initialize noise
        initialize_router_noise(router, self.noise_type, self.noise_scale)
        
        # Initialize auxiliary loss components
        initialize_auxiliary_loss(router, self.z_loss_weight, self.load_loss_weight)

__all__ = [
    "initialize_router_weights",
    "initialize_router_gates",
    "initialize_router_load_balancing",
    "initialize_router_temperature",
    "initialize_router_noise",
    "initialize_auxiliary_loss",
    "RouterInitializer",
]
