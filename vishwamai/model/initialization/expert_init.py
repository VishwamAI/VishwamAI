"""Expert weight initialization functions for MoE layers in JAX."""
from typing import Optional, Union
import math
import jax
import jax.numpy as jnp
from jax import random

def init_expert_weights(
    expert: dict,
    rng_key: jax.random.PRNGKey,
    init_method: str = 'scaled_normal',
    init_std: float = 0.02,
    num_experts: int = 1,
    capacity_factor: float = 1.0,
    expert_scale: Optional[float] = None
) -> dict:
    """Initialize expert module weights in JAX.
    
    Args:
        expert: Dictionary containing expert parameters
        rng_key: JAX random key for initialization
        init_method: Initialization method ['scaled_normal', 'kaiming', 'xavier']
        init_std: Standard deviation for normal initialization
        num_experts: Number of experts in MoE layer
        capacity_factor: Expert capacity multiplier
        expert_scale: Optional explicit scaling factor for expert weights
    
    Returns:
        Dictionary with initialized parameters
    """
    if expert_scale is None:
        expert_scale = math.sqrt(capacity_factor / num_experts)
    
    params = {}
    subkey = rng_key
    
    for name, param in expert.items():
        if 'weight' in name.lower():
            shape = param.shape
            
            if init_method == 'scaled_normal':
                params[name] = random.normal(subkey, shape) * init_std * expert_scale
                
            elif init_method == 'kaiming':
                fan_in = shape[-1]  # Assume last dim is input features
                gain = math.sqrt(2.0)  # ReLU gain
                std = gain / math.sqrt(fan_in)
                params[name] = random.normal(subkey, shape) * std * expert_scale
                
            elif init_method == 'xavier':
                fan_in, fan_out = shape[-1], shape[-2]  # Linear layer convention
                std = expert_scale * math.sqrt(2.0 / (fan_in + fan_out))
                params[name] = random.normal(subkey, shape) * std
                
            else:
                raise ValueError(f"Unknown initialization method: {init_method}")
                
            subkey, _ = random.split(subkey)
            
        elif 'bias' in name.lower():
            params[name] = jnp.zeros_like(param)
            
    return params

def init_expert_biases(
    expert: dict,
    rng_key: jax.random.PRNGKey,
    bias_init: Union[float, str] = 0.0
) -> dict:
    """Initialize expert module biases in JAX.
    
    Args:
        expert: Dictionary containing expert parameters
        rng_key: JAX random key for initialization
        bias_init: Bias initialization value or method
            - float: Use constant initialization
            - 'zero': Initialize to zeros
            - 'normal': Initialize from N(0, 0.02)
    
    Returns:
        Dictionary with initialized biases
    """
    params = {}
    subkey = rng_key
    
    for name, param in expert.items():
        if 'bias' in name.lower():
            if isinstance(bias_init, (int, float)):
                params[name] = jnp.full_like(param, bias_init)
            elif bias_init == 'zero':
                params[name] = jnp.zeros_like(param)
            elif bias_init == 'normal':
                params[name] = random.normal(subkey, param.shape) * 0.02
                subkey, _ = random.split(subkey)
            else:
                raise ValueError(f"Unknown bias initialization: {bias_init}")
        else:
            params[name] = param  # Copy non-bias params unchanged
            
    return params

def reset_failed_experts(
    expert: dict,
    rng_key: jax.random.PRNGKey,
    expert_idx: int,
    init_method: str = 'scaled_normal',
    init_std: float = 0.02,
    num_experts: int = 1,
    capacity_factor: float = 1.0
) -> dict:
    """Re-initialize a failed expert's weights in JAX.
    
    Args:
        expert: Dictionary containing expert parameters
        rng_key: JAX random key for initialization
        expert_idx: Index of expert being reset
        init_method: Initialization method
        init_std: Standard deviation for normal initialization
        num_experts: Total number of experts
        capacity_factor: Expert capacity multiplier
    
    Returns:
        Dictionary with re-initialized parameters
    """
    reset_scale = math.sqrt(capacity_factor / num_experts)
    
    # Re-initialize weights
    params = init_expert_weights(
        expert,
        rng_key,
        init_method=init_method,
        init_std=init_std,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        expert_scale=reset_scale
    )
    
    # Re-initialize biases to zero
    subkey, _ = random.split(rng_key)
    params = init_expert_biases(params, subkey, bias_init='zero')
    
    return params
