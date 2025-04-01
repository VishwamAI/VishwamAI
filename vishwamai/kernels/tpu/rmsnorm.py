"""TPU-optimized RMSNorm implementation."""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from typing import Optional, Tuple, NamedTuple
from functools import partial

from .tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

class RMSNormOutput(NamedTuple):
    """Output from RMSNorm computation."""
    normalized: jnp.ndarray
    scale: jnp.ndarray
    inv_rms: jnp.ndarray

class TPURMSNorm:
    """
    TPU-optimized Root Mean Square Layer Normalization.
    
    Features:
    - Blocked computation for TPU efficiency
    - Fused operations
    - Mixed precision support
    - Memory-efficient implementation
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        block_size: int = 128,
        use_bias: bool = False,
        dtype: jnp.dtype = jnp.float32
    ):
        """
        Initialize RMSNorm.
        
        Args:
            dim: Hidden dimension size
            eps: Small constant for numerical stability
            block_size: Block size for TPU operations (must be multiple of 128)
            use_bias: Whether to use bias term
            dtype: Data type for computation
        """
        if block_size % 128 != 0:
            raise ValueError("Block size must be multiple of 128 for TPU")
            
        self.dim = dim
        self.eps = eps
        self.block_size = block_size
        self.use_bias = use_bias
        self.dtype = dtype
        
        # Initialize parameters
        self.weight = jnp.ones(dim, dtype=dtype)
        if use_bias:
            self.bias = jnp.zeros(dim, dtype=dtype)
            
    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        x: jnp.ndarray,
        weight: Optional[jnp.ndarray] = None,
        bias: Optional[jnp.ndarray] = None
    ) -> RMSNormOutput:
        """
        Apply RMSNorm to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            weight: Optional weight parameter
            bias: Optional bias parameter
            
        Returns:
            RMSNormOutput with normalized tensor and scale factors
        """
        orig_dtype = x.dtype
        x = x.astype(self.dtype)
        
        # Use provided or default parameters
        if weight is None:
            weight = self.weight
        if bias is None and self.use_bias:
            bias = self.bias
            
        # Optimize memory layout
        x = optimize_tpu_layout(x, self.block_size)
        
        # Compute RMS statistics in blocks
        def compute_block_rms(block):
            square_sum = jnp.mean(block * block, axis=-1, keepdims=True)
            return jnp.sqrt(square_sum + self.eps)
            
        # Process in blocks for TPU efficiency
        block_size = min(self.block_size, x.shape[-1])
        num_blocks = (x.shape[-1] + block_size - 1) // block_size
        
        rms_parts = []
        for i in range(0, x.shape[-1], block_size):
            block = jax.lax.dynamic_slice(
                x,
                tuple([0] * (x.ndim - 1) + [i]),
                tuple(list(x.shape[:-1]) + [min(block_size, x.shape[-1] - i)])
            )
            rms_parts.append(compute_block_rms(block))
            
        # Combine RMS values
        rms = jnp.mean(jnp.stack(rms_parts, axis=0), axis=0)
        inv_rms = 1.0 / rms
        
        # Normalize
        x_norm = x * inv_rms
        
        # Apply scale and bias
        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None and self.use_bias:
            x_norm = x_norm + bias
            
        # Cast back to original dtype
        x_norm = x_norm.astype(orig_dtype)
        
        return RMSNormOutput(
            normalized=x_norm,
            scale=weight if weight is not None else None,
            inv_rms=inv_rms
        )
        
    def fused_rmsnorm(
        self,
        x: jnp.ndarray,
        epsilon: float = 1e-6
    ) -> jnp.ndarray:
        """
        Fused RMSNorm implementation for better TPU performance.
        
        Args:
            x: Input tensor
            epsilon: Small constant for numerical stability
            
        Returns:
            Normalized tensor
        """
        # Pad to TPU-efficient size
        orig_shape = x.shape
        x_padded, padding = pad_to_tpu_multiple(x, self.block_size)
        
        # Fuse mean square computation
        ms = jax.lax.pmean(
            jnp.mean(jnp.square(x_padded), axis=-1, keepdims=True),
            axis_name="batch"
        )
        
        # Fuse normalization
        inv_rms = jax.lax.rsqrt(ms + epsilon)
        x_normed = x_padded * inv_rms
        
        # Apply weight
        if self.weight is not None:
            x_normed = x_normed * self.weight
            
        # Remove padding
        if padding:
            x_normed = x_normed[..., :orig_shape[-1]]
            
        return x_normed
        
    def rmsnorm_backward(
        self,
        grad: jnp.ndarray,
        x: jnp.ndarray,
        inv_rms: jnp.ndarray,
        scale: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Efficient backward pass for RMSNorm.
        
        Args:
            grad: Gradient of loss with respect to output
            x: Input tensor
            inv_rms: Inverse RMS values from forward pass
            scale: Scale parameter if used
            
        Returns:
            Tuple of (input gradient, scale gradient)
        """
        # Optimize memory layout
        grad = optimize_tpu_layout(grad, self.block_size)
        x = optimize_tpu_layout(x, self.block_size)
        
        # Compute gradients in blocks
        def block_grads(g_block, x_block, inv_rms_block, scale_block=None):
            if scale_block is not None:
                g_block = g_block * scale_block
                
            sum_grad = jnp.sum(g_block * x_block, axis=-1, keepdims=True)
            grad_x = inv_rms_block * (g_block - (x_block * sum_grad) / self.dim)
            
            if scale_block is not None:
                grad_scale = jnp.sum(g_block * x_block * inv_rms_block, axis=tuple(range(g_block.ndim-1)))
                return grad_x, grad_scale
            return grad_x, None
            
        # Process in blocks
        block_size = min(self.block_size, x.shape[-1])
        grad_x_parts = []
        grad_scale_parts = []
        
        for i in range(0, x.shape[-1], block_size):
            g_block = jax.lax.dynamic_slice(
                grad,
                tuple([0] * (grad.ndim - 1) + [i]),
                tuple(list(grad.shape[:-1]) + [min(block_size, grad.shape[-1] - i)])
            )
            x_block = jax.lax.dynamic_slice(
                x,
                tuple([0] * (x.ndim - 1) + [i]),
                tuple(list(x.shape[:-1]) + [min(block_size, x.shape[-1] - i)])
            )
            scale_block = None
            if scale is not None:
                scale_block = jax.lax.dynamic_slice(
                    scale,
                    (i,),
                    (min(block_size, scale.shape[0]),)
                )
                
            dx_block, ds_block = block_grads(g_block, x_block, inv_rms, scale_block)
            grad_x_parts.append(dx_block)
            if ds_block is not None:
                grad_scale_parts.append(ds_block)
                
        # Combine gradients
        grad_x = jnp.concatenate(grad_x_parts, axis=-1)
        grad_scale = jnp.concatenate(grad_scale_parts, axis=0) if grad_scale_parts else None
        
        return grad_x, grad_scale