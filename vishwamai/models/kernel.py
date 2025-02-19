"""
Optimized kernel operations and FP8 casting utilities.

This module provides optimized CUDA kernels and quantization utilities
for improved performance and memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

def act_quant_kernel(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    bits: int = 8,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to reduced precision.
    
    Args:
        x: Input tensor
        scale: Optional scaling factor
        bits: Number of bits for quantization
        symmetric: Whether to use symmetric quantization
        
    Returns:
        Tuple of (quantized tensor, scale factor)
    """
    if scale is None:
        if symmetric:
            scale = torch.max(torch.abs(x)).detach()
        else:
            scale = torch.max(x - torch.min(x)).detach()
            
    # Compute scaling
    qmax = 2 ** (bits - 1) - 1
    scale = scale / qmax
    
    # Scale input
    x_scaled = x / scale
    
    # Quantize
    x_quant = torch.round(x_scaled)
    x_quant = torch.clamp(x_quant, -qmax, qmax)
    
    # Scale back
    x_dequant = x_quant * scale
    
    return x_dequant, scale

def weight_dequant_kernel(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bits: int = 8
) -> torch.Tensor:
    """
    Dequantize weights from reduced precision.
    
    Args:
        weight: Quantized weight tensor
        scale: Scale factor
        bits: Number of bits used in quantization
        
    Returns:
        Dequantized tensor
    """
    qmax = 2 ** (bits - 1) - 1
    return weight * scale * qmax

def optimize_kernel_layout(
    weight: torch.Tensor,
    input_size: int,
    output_size: int
) -> torch.Tensor:
    """
    Optimize weight matrix layout for efficient computation.
    
    Args:
        weight: Weight tensor to optimize
        input_size: Size of input dimension
        output_size: Size of output dimension
        
    Returns:
        Optimized weight tensor
    """
    # Reshape to matrix
    weight = weight.view(output_size, input_size)
    
    # Compute optimal tile size
    tile_size = int(math.sqrt(weight.numel() / 32))  # 32 threads per warp
    tile_size = max(1, min(tile_size, 16))  # Keep reasonable bounds
    
    # Pad dimensions to tile boundaries
    pad_rows = (tile_size - (output_size % tile_size)) % tile_size
    pad_cols = (tile_size - (input_size % tile_size)) % tile_size
    
    if pad_rows > 0 or pad_cols > 0:
        weight = F.pad(weight, (0, pad_cols, 0, pad_rows))
        
    # Reshape to 4D with tiles
    weight = weight.view(
        output_size // tile_size,
        tile_size,
        input_size // tile_size,
        tile_size
    )
    
    # Optimize memory layout
    weight = weight.permute(0, 2, 1, 3)
    
    return weight

def fp8_cast(
    x: torch.Tensor,
    e: int = 4,  # Number of exponent bits
    scale: Optional[torch.Tensor] = None,
    amax_history: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cast floating point values to FP8 format.
    
    Args:
        x: Input tensor
        e: Number of exponent bits
        scale: Optional scale factor
        amax_history: History of maximum absolute values
        scale_inv: Inverse of scale factor
        
    Returns:
        Tuple of (FP8 tensor, updated scale)
    """
    m = 7 - e  # Number of mantissa bits
    bias = 2 ** (e - 1) - 1
    
    # Compute max representable value
    max_val = (2 - 2**(-m)) * 2**(2**e - 1 - bias)
    
    if scale is None:
        if amax_history is not None:
            # Use historical maximum
            amax = torch.max(amax_history)
        else:
            # Compute current maximum
            amax = torch.max(torch.abs(x)).detach()
            
        scale = max_val / amax
        scale_inv = 1 / scale
        
    # Scale input
    x_scaled = x * scale
    
    # Clamp to representable range
    x_scaled = torch.clamp(x_scaled, -max_val, max_val)
    
    # Round to nearest representable value
    x_fp8 = torch.round(x_scaled * 2**m) / 2**m
    
    # Scale back
    x_fp8 = x_fp8 * scale_inv
    
    return x_fp8, scale

def weight_quantize_kernel(
    weight: torch.Tensor,
    groups: int = 128,
    bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights with per-group scaling.
    
    Args:
        weight: Weight tensor to quantize
        groups: Number of groups for separate scaling factors
        bits: Number of quantization bits
        
    Returns:
        Tuple of (quantized weights, scale factors)
    """
    # Reshape to 2D
    orig_shape = weight.shape
    weight = weight.view(-1, weight.shape[-1])
    
    # Split into groups
    group_size = weight.shape[0] // groups
    weight = weight.view(groups, group_size, -1)
    
    # Compute scaling factors
    max_abs = torch.amax(torch.abs(weight), dim=(1, 2), keepdim=True)
    scale = max_abs / (2 ** (bits - 1) - 1)
    scale = scale.clamp(min=1e-5)  # Prevent division by zero
    
    # Scale and quantize
    weight_scaled = weight / scale
    weight_quant = torch.round(weight_scaled)
    weight_quant = torch.clamp(
        weight_quant,
        -2 ** (bits - 1) + 1,
        2 ** (bits - 1) - 1
    )
    
    # Reshape back
    weight_quant = weight_quant.view(orig_shape)
    scale = scale.squeeze()
    
    return weight_quant, scale
