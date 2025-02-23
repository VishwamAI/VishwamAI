"""General weight initialization functions in JAX."""
from typing import Union, Tuple,Optional
import math
import jax
import jax.numpy as jnp
from jax import random

def init_normal(
    params: dict,
    rng_key: jax.random.PRNGKey,
    mean: float = 0.0,
    std: float = 1.0
) -> dict:
    """Initialize weights using normal distribution.
    
    Args:
        params: Dictionary of parameters
        rng_key: JAX random key
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
    
    Returns:
        Initialized parameters dictionary
    """
    new_params = {}
    key = rng_key
    
    for name, param in params.items():
        if 'weight' in name.lower():
            key, subkey = random.split(key)
            new_params[name] = random.normal(subkey, param.shape) * std + mean
        elif 'bias' in name.lower():
            new_params[name] = jnp.zeros_like(param)
        else:
            new_params[name] = param
    return new_params

def init_uniform(
    params: dict,
    rng_key: jax.random.PRNGKey,
    a: float = 0.0,
    b: float = 1.0
) -> dict:
    """Initialize weights using uniform distribution.
    
    Args:
        params: Dictionary of parameters
        rng_key: JAX random key
        a: Lower bound of uniform distribution
        b: Upper bound of uniform distribution
    
    Returns:
        Initialized parameters dictionary
    """
    new_params = {}
    key = rng_key
    
    for name, param in params.items():
        if 'weight' in name.lower():
            key, subkey = random.split(key)
            new_params[name] = random.uniform(subkey, param.shape, minval=a, maxval=b)
        elif 'bias' in name.lower():
            new_params[name] = jnp.zeros_like(param)
        else:
            new_params[name] = param
    return new_params

def init_xavier_normal(
    params: dict,
    rng_key: jax.random.PRNGKey,
    gain: float = 1.0
) -> dict:
    """Initialize weights using Xavier normal initialization.
    
    Args:
        params: Dictionary of parameters
        rng_key: JAX random key
        gain: Optional scaling factor
    
    Returns:
        Initialized parameters dictionary
    """
    new_params = {}
    key = rng_key
    
    for name, param in params.items():
        if 'weight' in name.lower():
            key, subkey = random.split(key)
            fan_in, fan_out = param.shape[-1], param.shape[-2]
            std = gain * math.sqrt(2.0 / (fan_in + fan_out))
            new_params[name] = random.normal(subkey, param.shape) * std
        elif 'bias' in name.lower():
            new_params[name] = jnp.zeros_like(param)
        else:
            new_params[name] = param
    return new_params

def init_xavier_uniform(
    params: dict,
    rng_key: jax.random.PRNGKey,
    gain: float = 1.0
) -> dict:
    """Initialize weights using Xavier uniform initialization.
    
    Args:
        params: Dictionary of parameters
        rng_key: JAX random key
        gain: Optional scaling factor
    
    Returns:
        Initialized parameters dictionary
    """
    new_params = {}
    key = rng_key
    
    for name, param in params.items():
        if 'weight' in name.lower():
            key, subkey = random.split(key)
            fan_in, fan_out = param.shape[-1], param.shape[-2]
            limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
            new_params[name] = random.uniform(subkey, param.shape, minval=-limit, maxval=limit)
        elif 'bias' in name.lower():
            new_params[name] = jnp.zeros_like(param)
        else:
            new_params[name] = param
    return new_params

def init_kaiming_normal(
    params: dict,
    rng_key: jax.random.PRNGKey,
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> dict:
    """Initialize weights using Kaiming normal initialization.
    
    Args:
        params: Dictionary of parameters
        rng_key: JAX random key
        a: Negative slope for LeakyReLU
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Nonlinearity after this layer
    
    Returns:
        Initialized parameters dictionary
    """
    new_params = {}
    key = rng_key
    
    for name, param in params.items():
        if 'weight' in name.lower():
            key, subkey = random.split(key)
            fan = param.shape[-1] if mode == 'fan_in' else param.shape[-2]
            gain = math.sqrt(2.0 / (1 + a**2)) if nonlinearity == 'leaky_relu' else math.sqrt(2.0)
            std = gain / math.sqrt(fan)
            new_params[name] = random.normal(subkey, param.shape) * std
        elif 'bias' in name.lower():
            new_params[name] = jnp.zeros_like(param)
        else:
            new_params[name] = param
    return new_params

def init_kaiming_uniform(
    params: dict,
    rng_key: jax.random.PRNGKey,
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> dict:
    """Initialize weights using Kaiming uniform initialization.
    
    Args:
        params: Dictionary of parameters
        rng_key: JAX random key
        a: Negative slope for LeakyReLU
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Nonlinearity after this layer
    
    Returns:
        Initialized parameters dictionary
    """
    new_params = {}
    key = rng_key
    
    for name, param in params.items():
        if 'weight' in name.lower():
            key, subkey = random.split(key)
            fan = param.shape[-1] if mode == 'fan_in' else param.shape[-2]
            gain = math.sqrt(2.0 / (1 + a**2)) if nonlinearity == 'leaky_relu' else math.sqrt(2.0)
            limit = gain * math.sqrt(3.0 / fan)
            new_params[name] = random.uniform(subkey, param.shape, minval=-limit, maxval=limit)
        elif 'bias' in name.lower():
            new_params[name] = jnp.zeros_like(param)
        else:
            new_params[name] = param
    return new_params

def scaled_init(
    params: dict,
    rng_key: jax.random.PRNGKey,
    std: float = 0.02,
    bias_const: float = 0.0,
    scale_factor: Union[float, str] = 1.0,
    num_layers: Optional[int] = None
) -> dict:
    """Initialize weights with scaled normal distribution.
    
    Args:
        params: Dictionary of parameters
        rng_key: JAX random key
        std: Base standard deviation
        bias_const: Constant for bias initialization
        scale_factor: Factor to scale std by. If 'auto', uses 1/sqrt(2*num_layers)
        num_layers: Number of layers for auto scaling (required if scale_factor='auto')
    
    Returns:
        Initialized parameters dictionary
    """
    new_params = {}
    key = rng_key
    
    if isinstance(scale_factor, str) and scale_factor == 'auto':
        if num_layers is None:
            raise ValueError("num_layers must be provided when scale_factor='auto'")
        scale = 1.0 / math.sqrt(2.0 * num_layers)
    else:
        scale = float(scale_factor)
    
    for name, param in params.items():
        if 'weight' in name.lower():
            key, subkey = random.split(key)
            new_params[name] = random.normal(subkey, param.shape) * (std * scale)
        elif 'bias' in name.lower():
            new_params[name] = jnp.full_like(param, bias_const)
        else:
            new_params[name] = param
    return new_params
