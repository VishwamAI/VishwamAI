"""TPU-optimized kernels for VishwamAI."""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, Optional, Dict, Any, Union, List

# Check if TPU is available
TPU_AVAILABLE = jax.local_devices()[0].platform == 'tpu' if jax.local_devices() else False

def get_tpu_version():
    """Get the TPU version information."""
    if not TPU_AVAILABLE:
        return None
    
    # Extract TPU version from device description
    device_kind = jax.devices()[0].device_kind
    if 'TPU v' in device_kind:
        return device_kind
    return "Unknown TPU"

# Optimization flags for TPU
TPU_VERSION = get_tpu_version()
IS_TPU_V4 = TPU_VERSION and 'v4' in TPU_VERSION.lower()
IS_TPU_V5 = TPU_VERSION and 'v5' in TPU_VERSION.lower()

# Optimized matrix multiplication for TPU
@partial(jax.jit, static_argnums=(2, 3))
def matmul_tpu_optimized(
    x: jnp.ndarray, 
    y: jnp.ndarray, 
    transpose_a: bool = False, 
    transpose_b: bool = False
) -> jnp.ndarray:
    """
    Optimized matrix multiplication for TPU.
    
    Args:
        x: Input tensor of shape [..., m, k]
        y: Input tensor of shape [..., k, n]
        transpose_a: Whether to transpose the first input
        transpose_b: Whether to transpose the second input
        
    Returns:
        Output tensor of shape [..., m, n]
    """
    # Use optimal data layout for TPU
    if IS_TPU_V4 or IS_TPU_V5:
        # TPU v4/v5 work well with medium-size matmuls in bfloat16
        x_bf16 = x.astype(jnp.bfloat16)
        y_bf16 = y.astype(jnp.bfloat16)
        if transpose_a:
            x_bf16 = jnp.transpose(x_bf16, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))
        if transpose_b:
            y_bf16 = jnp.transpose(y_bf16, (*range(y.ndim - 2), y.ndim - 1, y.ndim - 2))
        
        # Calculate matrix multiplication
        result = jnp.matmul(x_bf16, y_bf16)
        return result.astype(x.dtype)  # Convert back to original dtype
    else:
        # Fallback for other TPU versions or non-TPU devices
        if transpose_a:
            x = jnp.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))
        if transpose_b:
            y = jnp.transpose(y, (*range(y.ndim - 2), y.ndim - 1, y.ndim - 2))
        
        return jnp.matmul(x, y)

@partial(jax.jit, static_argnums=(3, 4))
def attention_tpu_optimized(
    q: jnp.ndarray,  # [batch, heads, seq_len, dim]
    k: jnp.ndarray,  # [batch, heads, seq_len, dim]
    v: jnp.ndarray,  # [batch, heads, seq_len, dim]
    causal: bool = True,
    scale: Optional[float] = None
) -> jnp.ndarray:
    """
    Optimized scaled dot-product attention for TPU.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        causal: Whether to apply causal mask
        scale: Scaling factor, defaults to 1/sqrt(dim)
        
    Returns:
        Output tensor after attention
    """
    # Optimize shapes for TPU memory layout
    batch, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    
    # Scale query by default
    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)
    q = q * scale
    
    # Attention scores
    # Use bfloat16 for TPU v4/v5 to leverage bfloat16 acceleration
    if IS_TPU_V4 or IS_TPU_V5:
        q_bf16 = q.astype(jnp.bfloat16)
        k_bf16 = k.astype(jnp.bfloat16)
        v_bf16 = v.astype(jnp.bfloat16)
        
        # Compute attention scores
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q_bf16, k_bf16)
    else:
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k)
    
    # Apply causal mask if needed
    if causal:
        # Create causal mask (lower triangular)
        mask = jnp.triu(
            jnp.ones((1, 1, seq_len_q, seq_len_k), dtype=q.dtype) * jnp.finfo(q.dtype).min, 
            k=1
        )
        attn_scores = attn_scores + mask
        
    # Apply softmax to get attention weights
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    
    # Apply attention weights to values
    if IS_TPU_V4 or IS_TPU_V5:
        attn_weights_bf16 = attn_weights.astype(jnp.bfloat16)
        output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights_bf16, v_bf16).astype(v.dtype)
    else:
        output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
    return output

@jax.jit
def rotary_embedding_tpu(
    x: jnp.ndarray,  # [..., seq_len, dim]
    cos: jnp.ndarray,  # [seq_len, dim/2]
    sin: jnp.ndarray   # [seq_len, dim/2]
) -> jnp.ndarray:
    """
    Optimized rotary position embeddings (RoPE) for TPU.
    
    Args:
        x: Input tensor
        cos: Cosine part of the rotary embeddings
        sin: Sine part of the rotary embeddings
        
    Returns:
        Tensor with rotary embeddings applied
    """
    # Get sequence length and dimension
    seq_len = x.shape[-2]
    dim = x.shape[-1]
    half_dim = dim // 2
    
    # Reshape x to split the last dimension in half
    x_reshaped = x.reshape(*x.shape[:-1], 2, half_dim)
    
    # Extract even and odd components
    x_even = x_reshaped[..., 0, :]
    x_odd = x_reshaped[..., 1, :]
    
    # Ensure cos and sin have proper shapes
    if cos.shape[0] < seq_len:
        # Pad if needed
        cos = jnp.pad(cos, ((0, seq_len - cos.shape[0]), (0, 0)))
        sin = jnp.pad(sin, ((0, seq_len - sin.shape[0]), (0, 0)))
    else:
        # Truncate if needed
        cos = cos[:seq_len]
        sin = sin[:seq_len]
    
    # Apply the rotary embeddings with broadcasting for TPU efficiency
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_odd * cos + x_even * sin
    
    # Recombine the even and odd components
    x_rotated = jnp.stack([x_rotated_even, x_rotated_odd], axis=-2)
    
    # Reshape back to original shape
    return x_rotated.reshape(*x.shape)

@partial(jax.jit, static_argnums=(1, 2))
def fp8_cast_tpu(
    x: jnp.ndarray,
    e_bits: int = 4,
    m_bits: int = 3
) -> jnp.ndarray:
    """
    TPU-optimized FP8 casting for e4m3 or e5m2 formats.
    
    For TPU, this creates a simulated FP8 representation since TPUs don't natively support FP8.
    
    Args:
        x: Input tensor
        e_bits: Number of exponent bits (4 for e4m3, 5 for e5m2)
        m_bits: Number of mantissa bits (3 for e4m3, 2 for e5m2)
        
    Returns:
        Tensor quantized to simulate FP8 precision
    """
    # Determine the scaling factor based on FP8 format
    if e_bits == 4 and m_bits == 3:  # e4m3
        max_exp = 7
        min_exp = -8
    elif e_bits == 5 and m_bits == 2:  # e5m2
        max_exp = 15
        min_exp = -16
    else:
        raise ValueError(f"Unsupported FP8 format: e{e_bits}m{m_bits}")
    
    # Scale factor
    scale = jnp.max(jnp.abs(x)) / (2**max_exp - 2**(-m_bits))
    
    # Scale and clamp the values to FP8 range
    x_scaled = x / scale
    x_clamped = jnp.clip(x_scaled, -2**max_exp + 2**(-m_bits), 2**max_exp - 2**(-m_bits))
    
    # Quantize to FP8 precision
    mantissa_bits = m_bits
    total_bits = e_bits + m_bits + 1  # +1 for sign bit
    
    # Compute the step size for the mantissa
    step = 2.0 ** (-mantissa_bits)
    
    # Round to the nearest representable value
    x_quantized = jnp.round(x_clamped / step) * step
    
    # Scale back to the original range
    return x_quantized * scale

@jax.jit
def multi_head_self_attention_tpu(
    q: jnp.ndarray,  # [batch, seq_len, num_heads, head_dim]
    k: jnp.ndarray,  # [batch, seq_len, num_heads, head_dim]
    v: jnp.ndarray,  # [batch, seq_len, num_heads, head_dim]
    mask: Optional[jnp.ndarray] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    TPU-optimized multi-head self-attention.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        mask: Optional attention mask
        dropout_rate: Dropout rate for attention weights
        training: Whether in training mode
        key: PRNG key for dropout
        
    Returns:
        Output tensor after multi-head attention
    """
    # Transpose to shape expected by attention_tpu_optimized
    # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    
    # Compute attention scores
    batch, num_heads, seq_len_q, head_dim = q.shape
    scale = 1.0 / jnp.sqrt(head_dim)
    
    attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    
    # Apply mask if provided
    if mask is not None:
        # Make sure mask is broadcastable [batch, 1, seq_len_q, seq_len_k]
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
            
        # Apply mask
        big_neg = jnp.finfo(q.dtype).min
        attn_scores = jnp.where(mask == 0, big_neg, attn_scores)
    
    # Apply softmax
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    
    # Apply dropout if in training mode
    if training and dropout_rate > 0:
        if key is None:
            key = jax.random.PRNGKey(0)
        keep_prob = 1.0 - dropout_rate
        dropout_shape = (batch, num_heads, seq_len_q, 1)
        keep_mask = jax.random.bernoulli(key, keep_prob, shape=dropout_shape)
        keep_mask = jnp.broadcast_to(keep_mask, attn_weights.shape)
        attn_weights = keep_mask * attn_weights / keep_prob
    
    # Apply attention weights to value
    outputs = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
    
    # Transpose back to original format
    # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    outputs = jnp.transpose(outputs, (0, 2, 1, 3))
    
    return outputs

@jax.jit
def layer_norm_tpu(
    x: jnp.ndarray,
    weight: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    eps: float = 1e-5
) -> jnp.ndarray:
    """
    TPU-optimized Layer Normalization.
    
    Args:
        x: Input tensor
        weight: Scale parameter
        bias: Shift parameter
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor
    """
    # Calculate mean and variance along last dimension
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    
    # Apply scale and shift if provided
    if weight is not None:
        x_norm = x_norm * weight
    
    if bias is not None:
        x_norm = x_norm + bias
        
    return x_norm

@jax.jit
def gelu_tpu(x: jnp.ndarray) -> jnp.ndarray:
    """
    TPU-optimized GELU activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor after GELU activation
    """
    # Use the approximation formula for TPU efficiency
    return x * jax.nn.sigmoid(1.702 * x)

# Kernels collection for easy importing
optimized_kernels = {
    "matmul": matmul_tpu_optimized,
    "attention": attention_tpu_optimized,
    "rotary_embedding": rotary_embedding_tpu,
    "fp8_cast": fp8_cast_tpu,
    "multi_head_attention": multi_head_self_attention_tpu,
    "layer_norm": layer_norm_tpu,
    "gelu": gelu_tpu
}