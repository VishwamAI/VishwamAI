"""TPU-optimized quantization utilities."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional, Dict, Any
from functools import partial

def dynamic_quant(
    x: jnp.ndarray,
    block_size: int = 128,  # Must be multiple of 128 for TPU
    bits: int = 8,
    scale_axis: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Dynamic quantization with TPU optimizations.
    
    Args:
        x: Input tensor
        block_size: Block size for quantization
        bits: Number of bits for quantization
        scale_axis: Axis to compute scales over
        
    Returns:
        Tuple of (quantized tensor, scale factors)
    """
    if block_size % 128 != 0:
        raise ValueError("Block size must be multiple of 128 for TPU")
        
    # Compute quantization parameters
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    
    # Handle different tensor ranks
    if x.ndim >= 2:
        # For matrices and higher-rank tensors, use block-wise quantization
        *batch_dims, rows, cols = x.shape
        
        # Compute number of blocks
        num_row_blocks = (rows + block_size - 1) // block_size
        num_col_blocks = (cols + block_size - 1) // block_size
        
        # Pad if needed
        padded_rows = num_row_blocks * block_size
        padded_cols = num_col_blocks * block_size
        
        if padded_rows > rows or padded_cols > cols:
            padding = [(0, 0)] * len(batch_dims) + [
                (0, padded_rows - rows),
                (0, padded_cols - cols)
            ]
            x = jnp.pad(x, padding)
            
        # Reshape for block-wise processing
        x_blocked = x.reshape(*batch_dims, num_row_blocks, block_size,
                            num_col_blocks, block_size)
        
        # Compute scales per block
        abs_max = jnp.max(jnp.abs(x_blocked), axis=(-3, -1), keepdims=True)
        scales = jnp.where(abs_max > 0, qmax / (abs_max + 1e-6), 1.0)
        
        # Quantize
        x_quant = jnp.clip(jnp.round(x_blocked * scales), qmin, qmax)
        x_quant = x_quant.astype(jnp.int8)
        
        # Restore original shape
        x_quant = x_quant.reshape(x.shape)
        scales = scales.reshape(*batch_dims, num_row_blocks, 1, num_col_blocks, 1)
        
    else:
        # For vectors, use simple quantization
        abs_max = jnp.max(jnp.abs(x), axis=scale_axis, keepdims=True)
        scales = jnp.where(abs_max > 0, qmax / (abs_max + 1e-6), 1.0)
        x_quant = jnp.clip(jnp.round(x * scales), qmin, qmax).astype(jnp.int8)
    
    return x_quant, scales

def dequantize(
    x_quant: jnp.ndarray,
    scale: jnp.ndarray,
    dtype: Any = jnp.float32
) -> jnp.ndarray:
    """
    Dequantize values with TPU optimization.
    
    Args:
        x_quant: Quantized tensor
        scale: Scale factors
        dtype: Output data type
        
    Returns:
        Dequantized tensor
    """
    return (x_quant * (1.0 / scale)).astype(dtype)

@partial(jax.jit, static_argnums=(3, 4))
def fp8_quantize(
    x: jnp.ndarray,
    block_size: int = 128,
    amax_history_x: Optional[jnp.ndarray] = None,
    fp8_format: str = "e4m3",
    amax_compute_algo: str = "most_recent"
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    FP8 quantization optimized for TPU.
    
    Args:
        x: Input tensor
        block_size: Block size for quantization
        amax_history_x: History of amax values
        fp8_format: FP8 format specification
        amax_compute_algo: Algorithm for computing amax
        
    Returns:
        Tuple of (quantized tensor, updated amax history)
    """
    if block_size % 128 != 0:
        raise ValueError("Block size must be multiple of 128 for TPU")
        
    # FP8 format parameters
    if fp8_format == "e4m3":
        exp_bits = 4
        man_bits = 3
    elif fp8_format == "e5m2":
        exp_bits = 5
        man_bits = 2
    else:
        raise ValueError(f"Unsupported FP8 format: {fp8_format}")
        
    bias = 2 ** (exp_bits - 1) - 1
    max_exp = 2 ** exp_bits - 1
    
    # Initialize or update amax history
    if amax_history_x is None:
        amax_history_x = jnp.zeros((8,) + x.shape[:-2], dtype=jnp.float32)
        
    # Compute current amax
    current_amax = jnp.max(jnp.abs(x), axis=(-2, -1), keepdims=True)
    
    # Update amax history
    amax_history_x = jnp.roll(amax_history_x, shift=-1, axis=0)
    amax_history_x = amax_history_x.at[-1].set(current_amax[..., 0, 0])
    
    # Compute scaling factor based on amax history
    if amax_compute_algo == "most_recent":
        scale = current_amax
    elif amax_compute_algo == "max":
        scale = jnp.max(amax_history_x, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unsupported amax compute algorithm: {amax_compute_algo}")
        
    # Reshape for block processing
    *batch_dims, rows, cols = x.shape
    num_row_blocks = (rows + block_size - 1) // block_size
    num_col_blocks = (cols + block_size - 1) // block_size
    
    # Pad if needed
    padded_rows = num_row_blocks * block_size
    padded_cols = num_col_blocks * block_size
    
    if padded_rows > rows or padded_cols > cols:
        padding = [(0, 0)] * len(batch_dims) + [
            (0, padded_rows - rows),
            (0, padded_cols - cols)
        ]
        x = jnp.pad(x, padding)
        
    x = x.reshape(*batch_dims, num_row_blocks, block_size,
                  num_col_blocks, block_size)
                  
    # Quantize to FP8
    abs_x = jnp.abs(x)
    exp = jnp.floor(jnp.log2(abs_x + 1e-30))
    exp = jnp.clip(exp + bias, 0, max_exp)
    man = jnp.round((abs_x / (2 ** (exp - bias))) * (2 ** man_bits))
    
    # Combine exponent and mantissa
    fp8_x = (exp * (2 ** man_bits) + man) * jnp.sign(x)
    fp8_x = fp8_x.astype(jnp.int8)
    
    # Restore original shape
    fp8_x = fp8_x.reshape(x.shape)
    
    return fp8_x, amax_history_x
