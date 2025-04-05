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
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    use_fp8: bool = True
    block_size: int = 128
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Create kernel with input_dim as first dimension for proper matmul
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (x.shape[-1], self.features),  # [input_dim, output_dim]
            self.dtype
        )
        
        if self.use_fp8:
            # Quantize inputs and kernel
            x_quant, x_scale = act_quant(x, block_size=self.block_size)
            kernel_quant, kernel_scale = act_quant(kernel, block_size=self.block_size)
            
            # Use FP8 GEMM 
            y = fp8_gemm_optimized(
                x_quant,
                x_scale,
                kernel_quant,
                kernel_scale,
                block_size=self.block_size
            )
        else:
            kernel = optimize_kernel_layout(kernel)
            y = jnp.dot(x, kernel)
            
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
            y = y + bias
            
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
        # Ensure the input is properly aligned for TPU
        if x.shape[self.axis] % self.block_size != 0:
            pad_size = self.block_size - (x.shape[self.axis] % self.block_size)
            padding = [(0, 0)] * x.ndim
            padding[self.axis] = (0, pad_size)
            x = jnp.pad(x, padding, mode='constant')
            
        feature_shape = (x.shape[self.axis],)
        
        # Cast to float32 for stable reduction operations
        x_f32 = x.astype(jnp.float32)
        
        # Compute statistics with Welford's online algorithm for stability
        mean = jnp.mean(x_f32, axis=self.axis, keepdims=True)
        centered = x_f32 - mean
        var = jnp.mean(jnp.square(centered), axis=self.axis, keepdims=True)
        
        # Normalize with high precision
        inv_std = jax.lax.rsqrt(var + self.epsilon)
        y = centered * inv_std
        
        # Apply scale and bias if configured
        if self.use_scale:
            scale = self.param('scale', self.scale_init, feature_shape, self.dtype)
            scale = scale.astype(jnp.float32)
            y = y * scale
            
        if self.use_bias:
            bias = self.param('bias', self.bias_init, feature_shape, self.dtype)
            bias = bias.astype(jnp.float32)
            y = y + bias
            
        # Remove padding if added
        if x.shape[self.axis] != feature_shape[0]:
            slicing = [slice(None)] * y.ndim
            slicing[self.axis] = slice(0, feature_shape[0])
            y = y[tuple(slicing)]
            
        # Cast back to working precision
        return y.astype(self.dtype)

class TPUMultiHeadAttention(nn.Module):
    """TPU-optimized Multi-Head Attention with Flash Attention."""
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    qkv_bias: bool = True
    use_flash_attn: bool = True
    use_fp8: bool = True
    block_size: int = 128
    
    def setup(self):
        self.hidden_dim = self.num_heads * self.head_dim
        self.q_proj = TPUGEMMLinear(
            features=self.hidden_dim,
            use_bias=self.qkv_bias,
            dtype=self.dtype
        )
        
        self.k_proj = TPUGEMMLinear(
            features=self.hidden_dim,
            use_bias=self.qkv_bias,
            dtype=self.dtype
        )
        
        self.v_proj = TPUGEMMLinear(
            features=self.hidden_dim,
            use_bias=self.qkv_bias,
            dtype=self.dtype
        )
        
        self.out_proj = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype
        )

    def __call__(
        self,
        inputs_q: jnp.ndarray,
        inputs_kv: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        batch_size, q_len = inputs_q.shape[:2]
        kv_len = inputs_kv.shape[1]

        # Project inputs
        q = self.q_proj(inputs_q)
        k = self.k_proj(inputs_kv)
        v = self.v_proj(inputs_kv)

        # Reshape for attention
        def split_heads(x: jnp.ndarray) -> jnp.ndarray:
            return x.reshape(batch_size, -1, self.num_heads, self.head_dim)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Transpose for attention calculation
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention with proper padding for TPU
        if self.use_flash_attn:
            # Ensure dimensions are multiples of block_size
            def pad_to_block(x: jnp.ndarray, seq_len: int) -> Tuple[jnp.ndarray, int]:
                pad_len = (self.block_size - seq_len % self.block_size) % self.block_size
                if pad_len > 0:
                    padding = [(0, 0), (0, 0), (0, pad_len), (0, 0)]
                    x = jnp.pad(x, padding)
                return x, pad_len

            q, q_pad = pad_to_block(q, q_len)
            k, k_pad = pad_to_block(k, kv_len)
            v, v_pad = pad_to_block(v, kv_len)

            # Update attention mask for padded sequence
            if mask is not None:
                mask_padding = [(0, 0), (0, 0), (0, q_pad), (0, k_pad)]
                mask = jnp.pad(mask, mask_padding, constant_values=0)

        # Compute scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attention = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale

        if mask is not None:
            big_neg = jnp.finfo(attention.dtype).min
            attention = jnp.where(mask, attention, big_neg)

        # Apply softmax and dropout
        attention = jax.nn.softmax(attention, axis=-1)
        if not deterministic and self.dropout_rate > 0:
            attention = nn.Dropout(rate=self.dropout_rate)(
                attention, deterministic=False
            )

        # Compute output
        out = jnp.einsum('bhqk,bhkd->bhqd', attention, v)
        
        # Remove padding if needed
        if self.use_flash_attn and q_pad > 0:
            out = out[:, :, :q_len]

        # Reshape output
        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(batch_size, -1, self.hidden_dim)
        return self.out_proj(out)

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