"""Quantization utilities for TPU optimization."""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Union

def dynamic_quant(x: jnp.ndarray, num_bits: int = 8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Dynamically quantize tensor to specified bit-width with per-tensor scaling.
    
    Args:
        x: Input tensor to quantize
        num_bits: Number of bits for quantization
    
    Returns:
        Tuple of (quantized_tensor, scale_factor)
    """
    # Compute scale based on absolute max
    abs_max = jnp.max(jnp.abs(x))
    # Avoid division by zero
    abs_max = jnp.maximum(abs_max, 1e-10)
    
    # Compute scale to map to [-2^(num_bits-1), 2^(num_bits-1)-1]
    quant_range = 2**(num_bits - 1) - 1
    scale = quant_range / abs_max
    
    # Quantize and clip
    x_quant = jnp.round(x * scale)
    x_quant = jnp.clip(x_quant, -quant_range-1, quant_range)
    
    # For TPU V2/V3, force to int8 for best performance if num_bits <= 8
    if num_bits <= 8:
        x_quant = x_quant.astype(jnp.int8)
    
    return x_quant, 1.0 / scale

def static_quant(x: jnp.ndarray, scale: float, zero_point: float = 0, num_bits: int = 8) -> jnp.ndarray:
    """
    Statically quantize tensor with provided scale and zero point.
    
    Args:
        x: Input tensor to quantize
        scale: Scale factor (value per quantization step)
        zero_point: Zero point offset
        num_bits: Number of bits for quantization
    
    Returns:
        Quantized tensor
    """
    quant_min = -2**(num_bits - 1)
    quant_max = 2**(num_bits - 1) - 1
    
    # Quantize
    x_quant = jnp.round(x / scale + zero_point)
    x_quant = jnp.clip(x_quant, quant_min, quant_max)
    
    # Force to int8 for TPU optimization if applicable
    if num_bits <= 8:
        x_quant = x_quant.astype(jnp.int8)
    
    return x_quant

def act_quant_static(x: jnp.ndarray, act_stats: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quantize activations using pre-computed statistics.
    
    Args:
        x: Input tensor to quantize
        act_stats: Dictionary with pre-computed statistics
            (e.g., {"absmax": absmax_value})
    
    Returns:
        Tuple of (quantized_tensor, scale_factor)
    """
    absmax = act_stats.get("absmax", jnp.max(jnp.abs(x)))
    absmax = jnp.maximum(absmax, 1e-10)
    
    # Use int8 for TPU v2/v3 optimization
    quant_range = 127.0  # 2^7 - 1 for int8
    scale = quant_range / absmax
    
    x_quant = jnp.round(x * scale)
    x_quant = jnp.clip(x_quant, -128, 127)
    x_quant = x_quant.astype(jnp.int8)
    
    return x_quant, 1.0 / scale

def dequantize(x_quant: jnp.ndarray, scale: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """
    Dequantize a quantized tensor.
    
    Args:
        x_quant: Quantized tensor
        scale: Scale factor for dequantization
    
    Returns:
        Dequantized tensor
    """
    return x_quant.astype(jnp.float32) * scale
