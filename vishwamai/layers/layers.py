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
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (x.shape[-1], self.features),
            self.dtype
        )
        
        if self.use_fp8:
            # Use FP8 GEMM for faster computation
            x_quant, x_scale = act_quant(x, block_size=self.block_size)
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
            y = jnp.dot(x, kernel, precision=self.precision)
            
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
        # Projection layers
        self.qkv = TPUGEMMLinear(
            features=3 * self.num_heads * self.head_dim,
            use_bias=self.qkv_bias,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size
        )
        
        self.out = TPUGEMMLinear(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size
        )
        
        # Flash Attention module
        if self.use_flash_attn:
            self.flash_attn = FlashAttention(
                block_size=self.block_size,
                use_fp8=self.use_fp8,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)  # [batch, seq, 3*heads*dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, batch, heads, seq, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Add past key/values for inference
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=2)  # Concat on seq length
            v = jnp.concatenate([past_v, v], axis=2)
        
        # Apply attention with consistent dimensions
        if self.use_flash_attn:
            # Flash attention expects [batch, heads, seq, dim]
            attn_output = self.flash_attn(
                q, k, v,
                mask=mask,
                deterministic=deterministic
            )
        else:
            # Standard scaled dot-product attention
            scale = 1.0 / jnp.sqrt(self.head_dim)
            # q, k: [batch, heads, seq, dim]
            attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
            
            if mask is not None:
                attn_weights = jnp.where(mask, attn_weights, -1e10)
            
            attn_weights = jax.nn.softmax(attn_weights, axis=-1)
            
            if not deterministic and self.dropout_rate > 0.0:
                dropout_rng = self.make_rng('dropout')
                keep_prob = 1.0 - self.dropout_rate
                dropout_mask = jax.random.bernoulli(
                    dropout_rng,
                    p=keep_prob,
                    shape=attn_weights.shape
                )
                attn_weights = attn_weights * dropout_mask / keep_prob
            
            attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        # Save current key/values for next step if needed
        present = (k, v) if past_key_value is not None else None
        
        # Reshape output to [batch, seq, heads*dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        # Final projection
        attn_output = self.out(attn_output)
        
        return attn_output, present

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