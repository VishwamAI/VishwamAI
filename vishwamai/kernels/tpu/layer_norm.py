"""TPU-optimized Layer Normalization kernel."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Tuple, Dict, Any, NamedTuple
import numpy as np
from functools import partial

from vishwamai.kernels.core.kernel import Kernel, KernelConfig
from vishwamai.kernels.core.kernel_manager import HardwareType
from vishwamai.kernels.tpu.tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

class LayerNormOutput(NamedTuple):
    """Output from layer normalization."""
    output: jnp.ndarray
    mean: jnp.ndarray
    var: jnp.ndarray

class TPULayerNormKernel:
    """TPU-optimized layer normalization kernel.
    
    Provides efficient layer normalization operations optimized for TPU hardware
    with support for various configurations and optimizations.
    """
    
    def __init__(
        self,
        dim: Optional[int] = None,
        epsilon: float = 1e-5,
        use_bias: bool = True,
        use_scale: bool = True,
        use_bfloat16: bool = True,
        fuse_ops: bool = True,
    ):
        """Initialize TPU LayerNorm kernel.
        
        Args:
            dim: Hidden dimension to normalize
            epsilon: Small constant for numerical stability
            use_bias: Whether to use bias
            use_scale: Whether to use scaling
            use_bfloat16: Whether to use bfloat16 for computations
            fuse_ops: Whether to fuse operations for better TPU performance
        """
        self.dim = dim
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.use_bfloat16 = use_bfloat16
        self.fuse_ops = fuse_ops
    
    def __call__(
        self,
        x: jnp.ndarray,
        weight: Optional[jnp.ndarray] = None,
        bias: Optional[jnp.ndarray] = None,
        return_stats: bool = False,
    ) -> jnp.ndarray:
        """Apply layer normalization optimized for TPU.
        
        Args:
            x: Input tensor
            weight: Optional scale parameter
            bias: Optional bias parameter
            return_stats: Whether to return mean and variance
            
        Returns:
            Normalized tensor, with optional statistics if requested
        """
        return self.forward(x, weight, bias, return_stats)
        
    def forward(
        self,
        x: jnp.ndarray,
        weight: Optional[jnp.ndarray] = None,
        bias: Optional[jnp.ndarray] = None,
        return_stats: bool = False,
    ) -> jnp.ndarray:
        """Forward pass computation.
        
        Args:
            x: Input tensor
            weight: Optional scale parameter
            bias: Optional bias parameter
            return_stats: Whether to return mean and variance
            
        Returns:
            Normalized tensor, with optional statistics if requested
        """
        # Cast to bfloat16 for TPU optimization if specified
        orig_dtype = x.dtype
        if self.use_bfloat16:
            x = x.astype(jnp.bfloat16)
            if weight is not None:
                weight = weight.astype(jnp.bfloat16)
            if bias is not None:
                bias = bias.astype(jnp.bfloat16)
            
        # Handle missing parameters
        if weight is None and self.use_scale:
            if self.dim is None:
                self.dim = x.shape[-1]
            weight = jnp.ones(self.dim, dtype=x.dtype)
            
        if bias is None and self.use_bias:
            if self.dim is None:
                self.dim = x.shape[-1]
            bias = jnp.zeros(self.dim, dtype=x.dtype)
            
        # Compute mean and variance along last dimension
        if self.fuse_ops:
            # Fused implementation with better TPU performance
            output, mean, var = self._fused_layernorm(x, weight, bias)
        else:
            # Standard implementation
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
            
            # Normalize
            inv_std = lax.rsqrt(var + self.epsilon)
            x_norm = (x - mean) * inv_std
            
            # Apply scale and bias if provided
            if weight is not None:
                x_norm = x_norm * weight
            if bias is not None:
                x_norm = x_norm + bias
                
            output = x_norm
            
        # Cast back to original dtype if needed
        if self.use_bfloat16 and orig_dtype != jnp.bfloat16:
            output = output.astype(orig_dtype)
            
        if return_stats:
            return LayerNormOutput(output, mean, var)
        else:
            return output
        
    @partial(jax.jit, static_argnums=(0,))
    def _fused_layernorm(
        self,
        x: jnp.ndarray,
        weight: Optional[jnp.ndarray],
        bias: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Fused layer normalization implementation for TPU.
        
        This implementation is optimized for TPU by using a custom fusion pattern
        that avoids multiple scans over the input.
        
        Args:
            x: Input tensor
            weight: Scale parameter
            bias: Bias parameter
            
        Returns:
            Tuple of (normalized output, mean, variance)
        """
        # Optimize layout for TPU
        x = optimize_tpu_layout(x)
        
        # Compute mean
        mean = jnp.mean(x, axis=-1, keepdims=True)
        
        # Center input
        centered_x = x - mean
        
        # Compute variance
        var = jnp.mean(jnp.square(centered_x), axis=-1, keepdims=True)
        
        # Compute scaling factor
        inv_std = lax.rsqrt(var + self.epsilon)
        
        # Normalize
        x_norm = centered_x * inv_std
        
        # Apply scale and bias if provided
        output = x_norm
        if weight is not None:
            output = output * weight
        if bias is not None:
            output = output + bias
            
        return output, mean, var

    def backward(
        self,
        grad_output: jnp.ndarray,
        x: jnp.ndarray,
        mean: jnp.ndarray,
        var: jnp.ndarray,
        weight: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Backward pass computation for gradients.
        
        Args:
            grad_output: Gradient with respect to output
            x: Input tensor from forward pass
            mean: Mean from forward pass
            var: Variance from forward pass
            weight: Scale parameter from forward pass
            
        Returns:
            Gradients for input, weight, and bias
        """
        # Cast to bfloat16 for TPU optimization if specified
        orig_dtype = grad_output.dtype
        if self.use_bfloat16:
            grad_output = grad_output.astype(jnp.bfloat16)
            x = x.astype(jnp.bfloat16)
            mean = mean.astype(jnp.bfloat16)
            var = var.astype(jnp.bfloat16)
            if weight is not None:
                weight = weight.astype(jnp.bfloat16)
        
        # Compute gradients efficiently
        N = x.shape[-1]
        inv_std = lax.rsqrt(var + self.epsilon)
        
        if weight is not None:
            # Gradient with respect to bias (sum over all except last dimension)
            grad_bias = jnp.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))
            
            # Gradient with respect to weight
            x_norm = (x - mean) * inv_std
            grad_weight = jnp.sum(grad_output * x_norm, axis=tuple(range(grad_output.ndim - 1)))
            
            # Apply weight to grad_output for input gradient calculation
            grad_output = grad_output * weight
        else:
            grad_bias = None
            grad_weight = None
        
        # Gradient with respect to input
        # Efficient implementation that minimizes redundant computations
        grad_output_sum = jnp.sum(grad_output, axis=-1, keepdims=True)
        grad_output_x_centered = jnp.sum(grad_output * (x - mean), axis=-1, keepdims=True)
        
        # Combine terms for dx
        dx = (grad_output - grad_output_sum / N - 
              (x - mean) * grad_output_x_centered * inv_std ** 2 / N) * inv_std
        
        # Cast back to original dtype if needed
        if self.use_bfloat16 and orig_dtype != jnp.bfloat16:
            dx = dx.astype(orig_dtype)
            if grad_weight is not None:
                grad_weight = grad_weight.astype(orig_dtype)
            if grad_bias is not None:
                grad_bias = grad_bias.astype(orig_dtype)
        
        return dx, grad_weight, grad_bias
        
    def init_parameters(self, dim: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize parameters for layer normalization.
        
        Args:
            dim: Dimension of the inputs to normalize
            
        Returns:
            Tuple of (weight, bias) parameters
        """
        self.dim = dim
        weight = jnp.ones(dim)
        bias = jnp.zeros(dim)
        return weight, bias