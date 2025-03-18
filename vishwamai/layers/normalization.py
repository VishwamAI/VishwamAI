"""TPU-optimized normalization layers for VishwamAI."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Dict

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    More efficient than LayerNorm as it doesn't require mean computation.
    Specifically optimized for TPU.
    
    References:
        https://arxiv.org/abs/1910.07467
    """
    dim: Optional[int] = None
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    scale_init: Callable = nn.initializers.ones
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply RMSNorm to input tensor."""
        feat_shape = (self.dim or x.shape[-1],)
        scale = self.param('scale', self.scale_init, feat_shape, self.param_dtype)
        
        # For TPU efficiency: compute norm all at once
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(variance + self.epsilon)
        
        # scale/weight application
        return x_norm * scale
    
class AdaNorm(nn.Module):
    """
    Adaptive Normalization (AdaNorm).
    
    A more flexible variant of LayerNorm that adapts to the 
    statistics of inputs dynamically.
    
    References:
        https://arxiv.org/abs/2004.08095
    """
    dim: Optional[int] = None
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    scale_init: Callable = nn.initializers.ones
    bias_init: Callable = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply AdaNorm to input tensor."""
        feat_shape = (self.dim or x.shape[-1],)
        
        # Learnable parameters
        scale = self.param('scale', self.scale_init, feat_shape, self.param_dtype)
        bias = self.param('bias', self.bias_init, feat_shape, self.param_dtype)
        adapt_scale = self.param('adapt_scale', nn.initializers.ones, (1,), self.param_dtype)
        adapt_bias = self.param('adapt_bias', nn.initializers.zeros, (1,), self.param_dtype)
        
        # Compute statistics
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        std = jnp.sqrt(var + self.epsilon)
        
        # Apply normalization with adaptive parameters
        x_norm = (x - mean) / std
        
        # Apply scaling and adaptive components
        adapt_factor = adapt_scale * jnp.mean(jnp.abs(x_norm), axis=-1, keepdims=True) + adapt_bias
        return adapt_factor * (x_norm * scale + bias)

class GroupNorm(nn.Module):
    """
    Group Normalization optimized for TPU.
    
    Normalizes over groups of channels for stable training.
    
    References:
        https://arxiv.org/abs/1803.08494
    """
    num_groups: int = 32
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    scale_init: Callable = nn.initializers.ones
    bias_init: Callable = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply GroupNorm to input tensor."""
        # Get input dimensions
        batch_size, *spatial_shape, num_channels = x.shape
        
        # Ensure channels divisible by groups
        if num_channels % self.num_groups != 0:
            raise ValueError(f"Number of channels ({num_channels}) must be divisible by num_groups ({self.num_groups})")
        
        # Parameters
        scale = self.param('scale', self.scale_init, (1, 1, 1, num_channels), self.param_dtype)
        bias = self.param('bias', self.bias_init, (1, 1, 1, num_channels), self.param_dtype)
        
        # Reshape for group normalization
        group_size = num_channels // self.num_groups
        x = x.reshape(batch_size, -1, self.num_groups, group_size)
        
        # Compute statistics along spatial & group dim for TPU efficiency
        mean = jnp.mean(x, axis=(1, 3), keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=(1, 3), keepdims=True)
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + self.epsilon)
        
        # Reshape back
        x = x.reshape(batch_size, *spatial_shape, num_channels)
        
        # Apply scale and bias
        return x * scale + bias
