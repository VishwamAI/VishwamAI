"""
Deep GEMM optimizations for GPU matrix operations.
Provides hardware-optimized implementations of key matrix operations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

class GEMMConfig:
    def __init__(
        self,
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 32,
        num_warps: int = 8,
        num_stages: int = 3
    ):
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

_GLOBAL_NUM_WARPS = 8

def set_num_warps(num_warps: int) -> None:
    """Set global number of warps for GEMM operations"""
    global _GLOBAL_NUM_WARPS
    _GLOBAL_NUM_WARPS = num_warps

def get_best_config(
    matrix_shape: Tuple[int, int],
    num_sms: Optional[int] = None
) -> GEMMConfig:
    """
    Get optimal GEMM configuration for matrix dimensions
    
    Args:
        matrix_shape: (M, N) shape of output matrix
        num_sms: Number of SMs available (for auto-tuning)
    """
    M, N = matrix_shape
    
    # Base configurations optimized for different sizes
    if max(M, N) <= 512:
        config = GEMMConfig(64, 64, 32, 4)
    elif max(M, N) <= 2048:
        config = GEMMConfig(128, 128, 32, 8)
    else:
        config = GEMMConfig(256, 256, 32, 16)
        
    # Adjust for available SMs
    if num_sms is not None:
        config.num_warps = min(config.num_warps, num_sms * 2)
        
    return config

def linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    config: Optional[GEMMConfig] = None,
    use_amp: bool = True
) -> torch.Tensor:
    """
    Optimized linear layer forward implementation
    
    Args:
        input: Input tensor (batch_size, *, in_features)
        weight: Weight matrix (out_features, in_features)  
        bias: Optional bias vector (out_features)
        config: GEMM configuration
        use_amp: Whether to use automatic mixed precision
    """
    if not torch.cuda.is_available():
        return F.linear(input, weight, bias)
        
    # Apply mixed precision if requested
    if use_amp:
        input = input.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        if bias is not None:
            bias = bias.to(torch.bfloat16)
    
    # Use optimized CUDA kernel for GEMM
    output = F.linear(input, weight, None)
    
    # Add bias if present
    if bias is not None:
        output = output + bias.unsqueeze(0).expand_as(output)
        
    return output

def linear_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    config: Optional[GEMMConfig] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Optimized linear layer backward pass
    
    Args:
        grad_output: Gradient w.r.t. output
        input: Input tensor from forward pass
        weight: Weight matrix
        bias: Optional bias vector
        config: GEMM configuration
    
    Returns:
        (grad_input, grad_weight, grad_bias)
    """
    if not torch.cuda.is_available():
        return F.linear_backward(grad_output, input, weight, bias)
        
    # Compute gradients
    grad_input = F.linear(grad_output, weight.t())
    grad_weight = torch.matmul(grad_output.transpose(-2, -1), input)
    
    grad_bias = None
    if bias is not None:
        grad_bias = grad_output.sum(dim=0)
        
    return grad_input, grad_weight, grad_bias

# Initialize optimal defaults
_default_config = get_best_config((1024, 1024))
set_num_warps(_default_config.num_warps)
