"""Mixture of Experts module components."""

from typing import Optional, Type, Union, Dict, Any

import torch
import torch.nn as nn

from .expert import (
    ExpertNetwork,
    ParallelExpertNetwork,
    create_experts
)
from .router import TopKRouter, DenseRouter
from .moe_layer import MoELayer

def create_moe_layer(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> MoELayer:
    """Create MoE layer from configuration.
    
    Args:
        config: Dictionary containing MoE configuration
        device: Device to create tensors on
        dtype: Data type for parameters
        
    Returns:
        Configured MoE layer
    """
    # Extract required params
    hidden_size = config["hidden_size"]
    num_experts = config["num_experts"]
    
    # Get expert class
    expert_type = config.get("expert_type", "parallel")
    expert_class = {
        "standard": ExpertNetwork,
        "parallel": ParallelExpertNetwork
    }[expert_type]
    
    # Get router class
    router_type = config.get("router_type", "top_k")
    router_class = {
        "top_k": TopKRouter,
        "dense": DenseRouter
    }[router_type]
    
    # Extract expert params
    expert_kwargs = {
        "intermediate_size": config.get("expert_intermediate_size", 4 * hidden_size),
        "activation": config.get("expert_activation", "gelu"),
        "dropout_prob": config.get("expert_dropout_prob", 0.1),
        "use_layer_norm": config.get("expert_layer_norm", True),
        "layer_norm_eps": config.get("expert_layer_norm_eps", 1e-5),
        "bias": config.get("expert_bias", True),
    }
    
    # Create MoE layer
    return MoELayer(
        hidden_size=hidden_size,
        num_experts=num_experts,
        expert_class=expert_class,
        router_class=router_class,
        num_selected_experts=config.get("num_selected_experts", 2),
        expert_capacity_factor=config.get("expert_capacity_factor", 1.25),
        expert_dropout_prob=config.get("expert_dropout_prob", 0.1),
        router_dropout_prob=config.get("router_dropout_prob", 0.1),
        jitter_noise=config.get("router_jitter_noise", 0.1),
        expert_parallel=config.get("expert_parallel", True),
        use_aux_loss=config.get("use_aux_loss", True),
        router_z_loss_coef=config.get("router_z_loss_coef", 0.001),
        router_aux_loss_coef=config.get("router_aux_loss_coef", 0.001),
        device=device,
        dtype=dtype,
        **expert_kwargs
    )

def create_nested_moe_layers(
    config: Dict[str, Any],
    num_layers: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> nn.ModuleList:
    """Create multiple MoE layers for a transformer block.
    
    Args:
        config: Dictionary containing MoE configuration
        num_layers: Number of MoE layers to create
        device: Device to create tensors on
        dtype: Data type for parameters
        
    Returns:
        ModuleList containing MoE layers
    """
    return nn.ModuleList([
        create_moe_layer(config, device=device, dtype=dtype)
        for _ in range(num_layers)
    ])

def compute_total_expert_params(
    hidden_size: int,
    num_experts: int,
    intermediate_size: Optional[int] = None,
    include_bias: bool = True
) -> int:
    """Compute total number of expert parameters.
    
    Args:
        hidden_size: Size of hidden dimension
        num_experts: Number of experts
        intermediate_size: Size of expert FFN dimension
        include_bias: Whether to include bias parameters
        
    Returns:
        Total number of parameters across all experts
    """
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
        
    # Compute params for single expert
    params_per_expert = hidden_size * intermediate_size  # Up projection
    params_per_expert += intermediate_size * hidden_size  # Down projection
    
    if include_bias:
        params_per_expert += intermediate_size  # Up projection bias
        params_per_expert += hidden_size  # Down projection bias
        
    # Multiply by number of experts
    total_params = params_per_expert * num_experts
    
    return total_params

def get_router_class(name: str) -> Type[Union[TopKRouter, DenseRouter]]:
    """Get router class by name.
    
    Args:
        name: Name of router class ('top_k' or 'dense')
        
    Returns:
        Router class
    """
    routers = {
        "top_k": TopKRouter,
        "dense": DenseRouter
    }
    
    if name not in routers:
        raise ValueError(
            f"Unknown router type: {name}. "
            f"Available options are: {list(routers.keys())}"
        )
        
    return routers[name]

def get_expert_class(name: str) -> Type[ExpertNetwork]:
    """Get expert network class by name.
    
    Args:
        name: Name of expert class ('standard' or 'parallel')
        
    Returns:
        Expert network class
    """
    experts = {
        "standard": ExpertNetwork,
        "parallel": ParallelExpertNetwork
    }
    
    if name not in experts:
        raise ValueError(
            f"Unknown expert type: {name}. "
            f"Available options are: {list(experts.keys())}"
        )
        
    return experts[name]

__all__ = [
    # Main components
    "MoELayer",
    "ExpertNetwork",
    "ParallelExpertNetwork",
    "TopKRouter",
    "DenseRouter",
    
    # Factory functions
    "create_moe_layer",
    "create_nested_moe_layers",
    "create_experts",
    
    # Utility functions
    "compute_total_expert_params",
    "get_router_class",
    "get_expert_class",
]
