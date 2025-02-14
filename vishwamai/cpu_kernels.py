from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to FP8. Returns quantized values and scale factors.
    
    Args:
        x: Input tensor
        block_size: Block size for quantization
    Returns:
        Tuple of (quantized tensor, scale factors)
    """
    scale = x.abs().max(dim=-1, keepdim=True)[0] / 448
    y = (x / scale).round().clip(-448, 448)
    return y.to(torch.int8), scale

def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantize weights from FP8.
    
    Args:
        weight: Quantized weight tensor
        scale: Scale factors
        block_size: Block size for dequantization
    Returns:
        Dequantized tensor in bfloat16
    """
    return weight.to(torch.bfloat16) * scale

def fp8_gemm(x: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using FP8 quantization.
    
    Args:
        x: Input tensor
        x_scale: Scale factors for input
        weight: Weight tensor 
        weight_scale: Scale factors for weights
    Returns:
        Result of matrix multiplication
    """
    return torch.matmul(x.to(torch.bfloat16) * x_scale, weight.to(torch.bfloat16) * weight_scale)
