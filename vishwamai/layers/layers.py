"""TPU-optimized neural network layers with conditional computation"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional, Tuple, Dict, List, Callable
from vishwamai.layers.attention import FlashAttention, flash_attention_inference
from vishwamai.kernels.core.kernel import fp8_gemm_optimized, act_quant, optimize_kernel_layout
from vishwamai.layers.mode import DynamicExpertGating

class TPUGEMMLinear(nn.Module):
    """TPU-optimized linear layer with FP8 support."""
    features: int
    use_bias: bool = True
    dtype: Any = jnp.bfloat16
    precision: Optional[Any] = None
    kernel_init: Callable = nn.initializers.glorot_uniform()
    bias_init: Callable = nn.initializers.zeros
    transpose: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply linear transformation."""
        inputs = jnp.asarray(inputs, self.dtype)
        
        # Store original shape for reshaping output
        orig_shape = inputs.shape
        
        # Reshape input to 2D for matrix multiplication
        if inputs.ndim > 2:
            inputs = inputs.reshape(-1, inputs.shape[-1])
            
        in_features = inputs.shape[-1]

        # Initialize kernel with correct shape
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (in_features, self.features) if not self.transpose else (self.features, in_features),
            self.dtype
        )

        # Initialize scales for fp8
        input_scale = self.param(
            'input_scale',
            nn.initializers.ones,
            (1,),
            jnp.float32
        )
        kernel_scale = self.param(
            'kernel_scale',
            nn.initializers.ones,
            (1,),
            jnp.float32
        )

        # Apply matrix multiplication with FP8 optimization
        y = fp8_gemm_optimized(
            inputs, 
            input_scale,
            kernel,
            kernel_scale,
            block_size=128
        )

        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
            y = y + bias

        # Restore original dimensions if input was higher dimensional
        if len(orig_shape) > 2:
            y = y.reshape(orig_shape[:-1] + (self.features,))

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
    dtype: Any = jnp.bfloat16
    use_flash_attn: bool = True

    def setup(self):
        """Initialize attention components."""
        self.q_proj = TPUGEMMLinear(self.num_heads * self.head_dim, dtype=self.dtype)
        self.k_proj = TPUGEMMLinear(self.num_heads * self.head_dim, dtype=self.dtype)
        self.v_proj = TPUGEMMLinear(self.num_heads * self.head_dim, dtype=self.dtype)
        self.out_proj = TPUGEMMLinear(self.num_heads * self.head_dim, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
        # Check if flash attention is available
        try:
            if self.use_flash_attn:
                from ..kernels.cuda.flash_mla_cuda import FlashMLA
                self.has_flash_attn = True
            else:
                self.has_flash_attn = False
        except ImportError:
            self.has_flash_attn = False

    def __call__(
        self,
        queries: jnp.ndarray,
        keys: jnp.ndarray,
        values: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        rng: Optional[Any] = None
    ) -> jnp.ndarray:
        """Compute attention with optimizations."""
        # Project inputs
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)
        
        # Reshape for multi-head attention
        batch_size = queries.shape[0]
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Try to use flash attention if available
        if self.has_flash_attn:
            try:
                from ..kernels.cuda.flash_mla_cuda import FlashMLA
                flash_attn = FlashMLA(
                    head_dim=self.head_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate if not deterministic else 0.0,
                    causal=False
                )
                output = flash_attn(q, k, v, mask=mask)
            except Exception:
                # Fallback to standard attention if flash attention fails
                self.has_flash_attn = False
                
        if not self.has_flash_attn:
            # Standard scaled dot-product attention
            scale = 1.0 / jnp.sqrt(self.head_dim)
            scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

            if mask is not None:
                # Broadcast mask to attention shape
                mask = mask[:, None, None, :] if mask.ndim == 2 else mask
                scores = jnp.where(mask, scores, float('-inf'))

            weights = jax.nn.softmax(scores, axis=-1)
            
            if not deterministic and self.dropout_rate > 0.0:
                weights = self.dropout(weights, deterministic=False, rng=rng)
                
            output = jnp.matmul(weights, v)

        # Reshape back and project output
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(output)

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