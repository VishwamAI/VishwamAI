"""Router weight initialization functions for MoE layers in JAX."""
from typing import Optional
import math
import jax
import jax.numpy as jnp
from jax import random

def init_router_weights(
    router_params: dict,
    rng_key: jax.random.PRNGKey,
    num_experts: int,
    hidden_size: int,
    init_method: str = 'scaled_normal',
    init_std: float = 0.02,
    capacity_factor: float = 1.0,
    router_bias: bool = True,
    router_scale: Optional[float] = None
) -> dict:
    """Initialize router weights in JAX.
    
    Args:
        router_params: Dictionary containing router parameters
        rng_key: JAX random key for initialization
        num_experts: Number of experts
        hidden_size: Hidden dimension size
        init_method: Initialization method ['scaled_normal', 'jax', 'uniform']
        init_std: Standard deviation for normal initialization
        capacity_factor: Expert capacity multiplier
        router_bias: Whether to use router biases
        router_scale: Optional explicit scaling factor
    
    Returns:
        Initialized router parameters dictionary
    """
    if router_scale is None:
        router_scale = 1.0 / math.sqrt(hidden_size * num_experts)
    
    params = {}
    key = rng_key
    
    if 'weight' in router_params:
        weight_shape = (num_experts, hidden_size)  # Typical router shape
        key, subkey = random.split(key)
        
        if init_method == 'scaled_normal':
            params['weight'] = random.normal(subkey, weight_shape) * init_std * router_scale
            
        elif init_method == 'jax':
            std = 1.0 / math.sqrt(hidden_size)
            weights = random.truncated_normal(
                subkey,
                lower=-2.0 * std,
                upper=2.0 * std,
                shape=weight_shape
            )
            params['weight'] = weights
            
        elif init_method == 'uniform':
            bound = router_scale / math.sqrt(3.0)
            params['weight'] = random.uniform(
                subkey,
                shape=weight_shape,
                minval=-bound,
                maxval=bound
            )
            
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
    
    if router_bias and 'bias' in router_params:
        params['bias'] = jnp.zeros((num_experts,))
        
    return params

def init_router_temperature(
    init_value: float = 1.0,
    learnable: bool = True
) -> dict:
    """Initialize router temperature parameter in JAX.
    
    Args:
        init_value: Initial temperature value
        learnable: Whether temperature is a learnable parameter (for JAX, affects optimization)
    
    Returns:
        Dictionary with temperature parameter
    """
    params = {
        'temperature': jnp.array(init_value),
        'temperature_learnable': learnable  # Metadata for optimization handling
    }
    return params

def init_router_noise(
    noise_type: str = 'gaussian',
    noise_std: float = 1.0,
    rng_key: Optional[jax.random.PRNGKey] = None
) -> dict:
    """Initialize router noise parameters in JAX.
    
    Args:
        noise_type: Type of noise ['gaussian', 'gumbel']
        noise_std: Standard deviation for Gaussian noise
        rng_key: Optional JAX random key for initialization if noise_std is learnable
    
    Returns:
        Dictionary with noise parameters
    """
    params = {
        'noise_type': noise_type,
        'noise_std': jnp.array(noise_std)
    }
    
    if rng_key is not None:  # If we want to initialize noise_std as a learnable param
        key, subkey = random.split(rng_key)
        params['noise_std'] = random.normal(subkey, ()) * 0.01 + noise_std
        
    return params
