"""CUDA-optimized kernels for VishwamAI."""

import functools
import math
import os
from typing import Optional, Tuple, Dict, Any, Union, List, Callable

import numpy as np
import jax
import jax.numpy as jnp

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.dlpack import from_dlpack, to_dlpack
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if GPU is available
GPU_AVAILABLE = jax.local_devices()[0].platform == 'gpu' if jax.local_devices() else False
CUDA_VERSION = None

if TORCH_AVAILABLE and torch.cuda.is_available():
    CUDA_VERSION = torch.version.cuda

# JAX-PyTorch interop utilities
def jax_to_torch(x: jnp.ndarray) -> 'torch.Tensor':
    """Convert JAX array to PyTorch tensor on GPU."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for JAX-PyTorch interop")
    return from_dlpack(jax.dlpack.to_dlpack(x))

def torch_to_jax(x: 'torch.Tensor') -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for JAX-PyTorch interop")
    return jax.dlpack.from_dlpack(to_dlpack(x))

# Kernel decorator for ensuring they're only used on GPU
def require_gpu(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not GPU_AVAILABLE:
            raise RuntimeError(f"Function {func.__name__} requires GPU but none is available")
        return func(*args, **kwargs)
    return wrapper

@require_gpu
@functools.partial(jax.jit, static_argnums=(2, 3), backend='cuda')
def matmul_gpu_optimized(
    x: jnp.ndarray, 
    y: jnp.ndarray,
    transpose_a: bool = False,
    transpose_b: bool = False
) -> jnp.ndarray:
    """
    Optimized matrix multiplication for GPU using JAX's CUDA backend.
    
    Args:
        x: Input tensor of shape [..., m, k]
        y: Input tensor of shape [..., k, n]
        transpose_a: Whether to transpose the first input
        transpose_b: Whether to transpose the second input
        
    Returns:
        Output tensor of shape [..., m, n]
    """
    if transpose_a:
        x_axes = list(range(x.ndim))
        x_axes[-2], x_axes[-1] = x_axes[-1], x_axes[-2]
        x = jnp.transpose(x, x_axes)
    
    if transpose_b:
        y_axes = list(range(y.ndim))
        y_axes[-2], y_axes[-1] = y_axes[-1], y_axes[-2]
        y = jnp.transpose(y, y_axes)
    
    # Use JIT-ed version with CUDA backend
    return jnp.matmul(x, y)

@require_gpu
@functools.partial(jax.jit, backend='cuda')
def flash_attention_gpu(
    q: jnp.ndarray,  # [batch, heads, seq_len, dim]
    k: jnp.ndarray,  # [batch, heads, seq_len, dim]
    v: jnp.ndarray,  # [batch, heads, seq_len, dim]
    mask: Optional[jnp.ndarray] = None,
    causal: bool = True,
    scale: Optional[float] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Optimized Flash Attention for GPU.
    
    This implementation follows the Flash Attention algorithm which has O(N) memory
    complexity instead of O(N²).
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        mask: Optional attention mask
        causal: Whether to apply causal mask
        scale: Scaling factor, defaults to 1/sqrt(dim)
        dropout_rate: Dropout rate for attention weights
        training: Whether in training mode
        key: PRNG key for dropout
        
    Returns:
        Output tensor after attention
    """
    # Extract dimensions
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    
    # Scale query by default
    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)
    q = q * scale
    
    # Block size for tiled implementation - optimized for typical GPU memory
    block_size = min(128, seq_len_k)
    
    # Initialize accumulators
    o = jnp.zeros_like(q)
    l = jnp.zeros((batch_size, num_heads, seq_len_q, 1))
    m = jnp.ones((batch_size, num_heads, seq_len_q, 1)) * -float('inf')
    
    # Process key-value pairs in blocks to maintain O(N) memory
    for block_start in range(0, seq_len_k, block_size):
        block_end = min(block_start + block_size, seq_len_k)
        
        # Get the current block of keys and values
        k_block = jax.lax.dynamic_slice(
            k, (0, 0, block_start, 0), 
            (batch_size, num_heads, block_end - block_start, head_dim)
        )
        v_block = jax.lax.dynamic_slice(
            v, (0, 0, block_start, 0), 
            (batch_size, num_heads, block_end - block_start, head_dim)
        )
        
        # Compute attention scores for the current block
        s_block = jnp.einsum('bhqd,bhkd->bhqk', q, k_block)
        
        # Apply mask if provided
        if mask is not None:
            # Extract the block from the mask if mask shape allows
            if mask.ndim == 4 and mask.shape[-1] >= seq_len_k:
                mask_block = jax.lax.dynamic_slice(
                    mask, (0, 0, 0, block_start), 
                    (mask.shape[0], mask.shape[1], mask.shape[2], block_end - block_start)
                )
            else:
                # Broadcast mask appropriately
                mask_block = mask
            
            # Apply mask
            big_neg = jnp.finfo(q.dtype).min
            s_block = jnp.where(mask_block == 0, big_neg, s_block)
        
        # Apply causal mask if needed
        if causal:
            causal_mask = jnp.greater(
                jnp.arange(seq_len_q)[:, None], 
                jnp.arange(block_start, block_end)[None, :]
            )
            s_block = jnp.where(
                causal_mask[None, None, :, :], 
                jnp.finfo(q.dtype).min, 
                s_block
            )
        
        # Update running maximum
        m_block = jnp.max(s_block, axis=-1, keepdims=True)
        m_new = jnp.maximum(m, m_block)
        
        # Update output and softmax denominator
        exp_s = jnp.exp(s_block - m_new)
        l_block = jnp.sum(exp_s, axis=-1, keepdims=True)
        l_new = l * jnp.exp(m - m_new) + l_block
        
        # Update output
        o = o * jnp.exp(m - m_new) + jnp.einsum('bhqk,bhkd->bhqd', exp_s, v_block)
        
        # Update running max and normalization factor
        m = m_new
        l = l_new
    
    # Normalize the output
    o = o / l
    
    return o

@require_gpu
@functools.partial(jax.jit, backend='cuda')
def rotary_embedding_gpu(
    x: jnp.ndarray,  # [..., seq_len, dim]
    cos: jnp.ndarray,  # [seq_len, dim/2]
    sin: jnp.ndarray   # [seq_len, dim/2]
) -> jnp.ndarray:
    """
    Optimized rotary position embeddings (RoPE) for GPU.
    
    Args:
        x: Input tensor
        cos: Cosine part of the rotary embeddings
        sin: Sine part of the rotary embeddings
        
    Returns:
        Tensor with rotary embeddings applied
    """
    # Get dimensions
    ndim = x.ndim
    seq_dim = ndim - 2
    half_dim = x.shape[-1] // 2
    
    # Reshape for easier manipulation
    shape = list(x.shape)
    shape[-1] = half_dim
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:2*half_dim]
    
    # Ensure proper dimensions
    if cos.shape[0] < x.shape[seq_dim]:
        # Pad if needed
        cos = jnp.pad(cos, ((0, x.shape[seq_dim] - cos.shape[0]), (0, 0)))
        sin = jnp.pad(sin, ((0, x.shape[seq_dim] - sin.shape[0]), (0, 0)))
    else:
        # Truncate if needed
        cos = cos[:x.shape[seq_dim]]
        sin = sin[:x.shape[seq_dim]]
    
    # Create broadcast-compatible shapes for better parallelization
    for _ in range(ndim - 2):
        cos = cos[None]
        sin = sin[None]
    
    # Apply rotary embeddings
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin
    
    # Concatenate results
    return jnp.concatenate([x1_rot, x2_rot], axis=-1)

@require_gpu
def fp8_cast_gpu_pytorch(
    x: jnp.ndarray,
    e_bits: int = 4,
    m_bits: int = 3
) -> jnp.ndarray:
    """
    GPU-optimized FP8 casting using PyTorch's quantization.
    
    Args:
        x: Input tensor
        e_bits: Number of exponent bits
        m_bits: Number of mantissa bits
        
    Returns:
        Tensor quantized to simulate FP8 precision
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        raise ImportError("PyTorch with CUDA support is required for fp8_cast_gpu")
    
    # Convert JAX array to PyTorch tensor
    x_torch = jax_to_torch(x)
    
    # Calculate scale factor
    max_exp = (1 << (e_bits - 1)) - 1
    scale = torch.max(torch.abs(x_torch)) / (2**max_exp - 2**(-m_bits))
    
    # Scale and quantize
    x_scaled = x_torch / scale
    
    # Determine number of bits
    num_bits = e_bits + m_bits + 1  # including sign bit
    
    # Simulate quantization
    if num_bits == 8:
        x_quantized = torch.quantize_per_tensor(
            x_scaled, scale=1.0, zero_point=0, dtype=torch.quint8
        )
        x_dequantized = x_quantized.dequantize() * scale
    else:
        # For non-standard bit widths, manually quantize
        qmin, qmax = 0, (1 << num_bits) - 1
        x_int = torch.clamp(torch.round(x_scaled * (1 << m_bits)), qmin, qmax)
        x_dequantized = (x_int / (1 << m_bits)) * scale
    
    # Convert back to JAX
    return torch_to_jax(x_dequantized)

@require_gpu
@functools.partial(jax.jit, backend='cuda')
def fp8_cast_gpu(
    x: jnp.ndarray,
    e_bits: int = 4,
    m_bits: int = 3
) -> jnp.ndarray:
    """
    GPU-optimized FP8 casting for e4m3 or e5m2 formats.
    
    Args:
        x: Input tensor
        e_bits: Number of exponent bits (4 for e4m3, 5 for e5m2)
        m_bits: Number of mantissa bits (3 for e4m3, 2 for e5m2)
        
    Returns:
        Tensor quantized to simulate FP8 precision
    """
    # Use PyTorch implementation if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return fp8_cast_gpu_pytorch(x, e_bits, m_bits)
        except Exception:
            pass  # Fall back to JAX implementation
    
    # Determine the scaling factor based on FP8 format
    if e_bits == 4 and m_bits == 3:  # e4m3
        max_exp = 7
        min_exp = -8
    elif e_bits == 5 and m_bits == 2:  # e5m2
        max_exp = 15
        min_exp = -16
    else:
        raise ValueError(f"Unsupported FP8 format: e{e_bits}m{m_bits}")
    
    # Scale factor - computed per tensor for simplicity
    scale = jnp.max(jnp.abs(x)) / (2**max_exp - 2**(-m_bits))
    
    # Scale and clamp the values to FP8 range
    x_scaled = x / scale
    x_clamped = jnp.clip(x_scaled, -2**max_exp + 2**(-m_bits), 2**max_exp - 2**(-m_bits))
    
    # Quantize to FP8 precision
    step = 2.0 ** (-m_bits)
    
    # Round to the nearest representable value
    x_quantized = jnp.round(x_clamped / step) * step
    
    # Scale back to the original range
    return x_quantized * scale

@require_gpu
@functools.partial(jax.jit, backend='cuda')
def layer_norm_gpu(
    x: jnp.ndarray,
    weight: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    eps: float = 1e-5
) -> jnp.ndarray:
    """
    GPU-optimized Layer Normalization.
    
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

@require_gpu
def fused_attention_pytorch(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Use PyTorch's fused scaled_dot_product_attention for efficiency.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, dim]
        k: Key tensor of shape [batch, heads, seq_len, dim]
        v: Value tensor of shape [batch, heads, seq_len, dim]
        mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal attention
        scale: Scaling factor for attention scores
        
    Returns:
        Tuple of (output tensor, attention weights)
    """
    if not TORCH_AVAILABLE or not hasattr(F, 'scaled_dot_product_attention'):
        raise ImportError("PyTorch with scaled_dot_product_attention is required")
    
    # Convert JAX tensors to PyTorch
    q_pt = jax_to_torch(q)
    k_pt = jax_to_torch(k)
    v_pt = jax_to_torch(v)
    
    mask_pt = None
    if mask is not None:
        mask_pt = jax_to_torch(mask)
    
    # Use PyTorch's fused attention
    attn_output = F.scaled_dot_product_attention(
        q_pt, k_pt, v_pt,
        attn_mask=mask_pt,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale
    )
    
    # Convert back to JAX
    return torch_to_jax(attn_output)

@require_gpu
@functools.partial(jax.jit, backend='cuda')
def gelu_gpu(x: jnp.ndarray) -> jnp.ndarray:
    """
    GPU-optimized GELU activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        GELU activation output
    """
    # Use the approximation formula optimized for GPU
    return 0.5 * x * (1.0 + jnp.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))

# Collection of all GPU-optimized kernels
optimized_kernels = {
    "matmul": matmul_gpu_optimized,
    "flash_attention": flash_attention_gpu,
    "rotary_embedding": rotary_embedding_gpu,
    "fp8_cast": fp8_cast_gpu,
    "layer_norm": layer_norm_gpu,
    "gelu": gelu_gpu
}

# Try to make PyTorch kernels available
if TORCH_AVAILABLE and torch.cuda.is_available() and hasattr(F, 'scaled_dot_product_attention'):
    optimized_kernels["fused_attention"] = fused_attention_pytorch