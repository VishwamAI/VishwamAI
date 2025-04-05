"""TPU-optimized neural network layers with conditional computation"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional, Tuple, Dict, List, Callable
from vishwamai.layers.attention import FlashAttention, flash_attention_inference
from vishwamai.kernels.core.kernel import fp8_gemm_optimized, act_quant, optimize_kernel_layout
from vishwamai.layers.mode import DynamicExpertGating

class TPUGEMMLinear(nn.Module):
    """Linear layer with TPU-optimized GEMM operations."""
    features: int
    use_bias: bool = True
    dtype: Any = jnp.bfloat16  # Changed to bfloat16 for TPU
    precision: Any = None
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    use_fp8: bool = True
    block_size: int = 128
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_shape = x.shape
        
        # Flatten input if needed
        x_flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        
        # Initialize kernel with correct shape
        kernel_shape = (x_flat.shape[-1], self.features)
        kernel = self.param(
            'kernel',
            self.kernel_init,
            kernel_shape,
            self.dtype
        )
        
        # Initialize bias if needed
        if self.use_bias:
            bias = self.param(
                'bias',
                self.bias_init,
                (self.features,),
                self.dtype
            )
        
        if self.use_fp8:
            # Quantize inputs and kernel
            x_quant, x_scale = act_quant(x_flat)
            kernel_quant, kernel_scale = act_quant(kernel)
            
            # Perform matrix multiplication with correct shapes
            y = fp8_gemm_optimized(
                x_quant,
                x_scale,
                kernel_quant,
                kernel_scale,
                block_size=self.block_size
            )
        else:
            # Regular matrix multiplication
            y = jax.lax.dot_general(
                x_flat, 
                kernel,
                (((1,), (0,)), ((), ())),
                precision=self.precision or jax.lax.Precision.HIGHEST
            )
        
        # Add bias if needed
        if self.use_bias:
            y = y + bias
            
        # Restore original dimensions if input was higher-dimensional
        if x.ndim > 2:
            y = y.reshape(input_shape[:-1] + (self.features,))
            
        return y

class TPULayerNorm(nn.Module):
    """TPU-optimized Layer Normalization."""
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    scale_init: Any = nn.initializers.ones
    bias_init: Any = nn.initializers.zeros
    use_bias: bool = True
    use_scale: bool = True
    axis: int = -1
    block_size: int = 128  # TPU-optimal block size
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply layer normalization with TPU optimizations."""
        # Get feature shape for parameters
        feature_shape = (x.shape[self.axis],)
        
        # Initialize parameters with correct shapes
        if self.use_scale:
            scale = self.param('scale', self.scale_init, feature_shape)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, feature_shape)
            
        # Cast input to computation dtype (float32 for stability)
        x = x.astype(jnp.float32)
        
        # Normalize along feature dimension
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        variance = jnp.mean(jnp.square(x - mean), axis=self.axis, keepdims=True)
        inv_stddev = jax.lax.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv_stddev
        
        # Apply scale and bias if configured
        if self.use_scale:
            normalized = normalized * scale.astype(jnp.float32)
        if self.use_bias:
            normalized = normalized + bias.astype(jnp.float32)
            
        # Cast back to working precision
        return normalized.astype(self.dtype)

class TPUShardedLinear(nn.Module):
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param('kernel',
                          self.kernel_init,
                          (inputs.shape[-1], self.features),
                          self.dtype)
        
        # Handle input reshaping for proper matrix multiplication
        input_shape = inputs.shape
        inputs_2d = inputs.reshape(-1, input_shape[-1])
        
        y = jax.lax.dot_general(
            inputs_2d, 
            kernel,
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST
        )
        
        # Reshape back to original dimensions
        output_shape = input_shape[:-1] + (self.features,)
        y = y.reshape(output_shape)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
            y = y + bias
        return y

class TPUMultiHeadAttention(nn.Module):
    """TPU-optimized multi-head attention."""
    num_heads: int
    head_dim: int 
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self,
                 inputs_q: jnp.ndarray,
                 inputs_kv: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        
        batch_size, seq_len = inputs_q.shape[0], inputs_q.shape[1]
        
        # Create attention weights
        query = nn.Dense(features=self.num_heads * self.head_dim)(inputs_q)
        key = nn.Dense(features=self.num_heads * self.head_dim)(inputs_kv)
        value = nn.Dense(features=self.num_heads * self.head_dim)(inputs_kv)
        
        # Reshape for attention
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Scale query
        depth = self.head_dim
        query = query / jnp.sqrt(depth).astype(self.dtype)
        
        # Calculate attention scores
        weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)
        
        # Apply mask if provided
        if mask is not None:
            weights = jnp.where(mask[:, None, :, :], weights, -1e10)
        
        # Apply softmax
        weights = jax.nn.softmax(weights, axis=-1)
        
        # Apply dropout inside the compact context
        weights = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=deterministic
        )(weights)
        
        # Calculate attention output
        output = jnp.einsum('bhqk,bkhd->bqhd', weights, value)
        
        # Reshape output
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Final dense projection
        output = nn.Dense(features=inputs_q.shape[-1])(output)
        
        return output

class TPURMSNorm(nn.Module):
    """TPU-optimized Root Mean Square normalization layer."""
    epsilon: float = 1e-6
    dtype: Any = jnp.bfloat16  # Default to bfloat16 for TPU
    scale_init: Any = nn.initializers.ones
    block_size: int = 128  # TPU-optimal block size

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Ensure TPU-friendly dimensions
        orig_shape = x.shape
        if x.shape[-1] % self.block_size != 0:
            pad_size = self.block_size - (x.shape[-1] % self.block_size)
            x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad_size)])
            
        # Cast to float32 for reduction operations
        x_f32 = x.astype(jnp.float32)
        
        # Get feature shape for scale parameter
        feature_shape = (x.shape[-1],)
        
        # Initialize scale parameter
        scale = self.param('scale', self.scale_init, feature_shape)
        scale = scale.astype(jnp.float32)
        
        # Fused RMS computation for TPU efficiency
        variance = jnp.mean(
            jnp.square(x_f32), 
            axis=-1, 
            keepdims=True,
            where=None if x.shape == orig_shape else jnp.arange(orig_shape[-1]) < orig_shape[-1]
        )
        
        # Use rsqrt for better TPU performance
        x_normalized = x_f32 * jax.lax.rsqrt(variance + self.epsilon)
        
        # Apply scale and cast back to working precision
        result = (x_normalized * scale).astype(self.dtype)
        
        # Remove padding if added
        if x.shape != orig_shape:
            result = result[..., :orig_shape[-1]]
            
        return result

class TPUMoELayer(nn.Module):
    """TPU-optimized Mixture of Experts layer."""
    num_experts: int
    expert_dim: int
    capacity_factor: float = 1.0
    dtype: Any = jnp.float32
    use_fp8: bool = True
    block_size: int = 128
    
    def setup(self):
        self.gating = DynamicExpertGating(
            num_experts=self.num_experts,
            expert_dim=self.expert_dim,
            capacity_factor=self.capacity_factor,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size
        )
    
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        return self.gating(x, deterministic)
# Alias for backwards compatibility
MoELayer = TPUMoELayer

class TPUDense(nn.Module):
    """TPU-optimized Dense (fully connected) layer."""
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    use_fp8: bool = True
    block_size: int = 128

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply the Dense layer to inputs.

        Args:
            inputs: Input array of shape (..., input_features)

        Returns:
            Output array of shape (..., features)
        """
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (inputs.shape[-1], self.features),
            self.param_dtype
        )
        kernel = jnp.asarray(kernel, self.dtype)

        if self.use_fp8:
            # Use FP8 GEMM for faster computation
            x_quant, x_scale = act_quant(inputs, block_size=self.block_size)
            kernel_quant, kernel_scale = act_quant(kernel, block_size=self.block_size)
            
            y = fp8_gemm_optimized(
                x_quant,
                x_scale,
                kernel_quant,
                kernel_scale,
                block_size=self.block_size
            )
        else:
            kernel = optimize_kernel_layout(kernel)
            y = jnp.dot(inputs, kernel, precision=self.precision)

        if self.use_bias:
            bias = self.param(
                'bias',
                self.bias_init,
                (self.features,),
                self.param_dtype
            )
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias

        return y

# Alias for compatibility with flax.linen
Dense = TPUDense
MoELayer = TPUMoELayer