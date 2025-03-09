"""
GPU-optimized kernel layers for VishwamAI.
Implements efficient matrix operations using DeepGEMM and optimized kernels.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import math
from typing import Optional, Tuple

from vishwamai.models.gpu.optimizations.deepgemm import (
    DistributedGEMM, GEMMConfig, get_best_configs,
    get_num_sms, layernorm
)
from vishwamai.models.gpu.optimizations.deepgemm.utils import bench, calc_diff
from vishwamai.models.gpu.optimizations.deepgemm.jit_kernels import (
    gemm_fp8_fp8_bf16_nt,
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
    m_grouped_gemm_fp8_fp8_bf16_nt_masked
)

class DeepGEMMLinear(nn.Linear):
    """Linear layer optimized with DeepGEMM"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 use_amp: bool = True, distributed: bool = False,
                 num_executors: Optional[int] = None,
                 cache_dir: Optional[str] = None):
        super().__init__(in_features, out_features, bias)
        self.use_amp = use_amp
        self.distributed = distributed
        
        # Initialize DeepGEMM components
        if distributed:
            self.gemm_engine = DistributedGEMM(
                num_executors=num_executors,
                cache_dir=cache_dir,
                config=get_best_configs(
                    hidden_size=out_features,
                    seq_len=512,  # Default sequence length
                    batch_size=32  # Default batch size
                )
            )
        else:
            self.gemm_engine = None
            
        # Cache warmup state
        self._is_warmed_up = False
        
    def _warmup(self, x: torch.Tensor):
        """Run warmup pass to optimize kernels"""
        if not self._is_warmed_up and torch.cuda.is_available():
            # Run forward pass with dummy data for kernel optimization
            with torch.no_grad():
                out = self.forward(x)
            if self.training:
                # Run backward pass if in training mode
                dummy_grad = torch.randn_like(out)
                out.backward(dummy_grad)
            torch.cuda.synchronize()
            self._is_warmed_up = True
    
    @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass using DeepGEMM"""
        if not self._is_warmed_up:
            self._warmup(x)
            
        if self.distributed and self.gemm_engine:
            # Use distributed GEMM computation
            output = self.gemm_engine.distribute_gemm(
                x, self.weight, self.bias
            )
        else:
            # Use optimized local GEMM
            if x.dtype == torch.float8_e4m3fn:
                output = gemm_fp8_fp8_bf16_nt(
                    (x, None),  # FP8 input with no scaling
                    (self.weight, None),  # FP8 weight with no scaling
                    out=torch.empty(x.size(0), self.out_features, 
                                      dtype=torch.bfloat16,
                                      device=x.device)
                )
            else:
                # Fallback to standard linear
                output = torch.nn.functional.linear(x, self.weight, self.bias)
                
        return output
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if self.gemm_engine:
            self.gemm_engine.cleanup()
            
    def __del__(self):
        self.cleanup()

class DeepGEMMLayerNorm(nn.LayerNorm):
    """Layer normalization optimized for GPU"""
    
    def __init__(self, normalized_shape, eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 use_amp: bool = True):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.use_amp = use_amp
        
    @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optimized implementation"""
        if self.use_amp and torch.cuda.is_available():
            with autocast():
                return layernorm(
                    x,
                    self.weight if self.elementwise_affine else None,
                    self.bias if self.elementwise_affine else None,
                    self.eps
                )
        return super().forward(x)

class DeepGEMMGroupedLinear(nn.Module):
    """Grouped linear layer using optimized GEMM, now supporting masked operations."""
    
    def __init__(self, in_features: int, out_features: int,
                 num_groups: int, bias: bool = True,
                 use_amp: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.use_amp = use_amp
        
        # Initialize parameters with grouped dimensions: [num_groups, out_features, in_features]
        self.weight = nn.Parameter(torch.empty(
            num_groups, out_features, in_features
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights and bias."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    @autocast()
    def forward(self, x: torch.Tensor, group_indices: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with grouped GEMM.
        
        If the input is in FP8 format (torch.float8_e4m3fn):
          - If a mask is provided, use the masked kernel.
          - Otherwise, use the contiguous kernel.
        
        For non-FP8 inputs, falls back to applying a standard linear operation group-wise.
        
        Args:
            x (torch.Tensor): Input tensor.
            group_indices (torch.Tensor): Tensor indicating group membership for each input row.
            mask (Optional[torch.Tensor]): Optional mask tensor for masked GEMM operation.
            
        Returns:
            torch.Tensor: The computed linear transformation output.
        """
        if x.dtype == torch.float8_e4m3fn:
            out_tensor = torch.empty(x.size(0), self.out_features,
                                     dtype=torch.bfloat16,
                                     device=x.device)
            if mask is not None:
                return m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                    (x, None),           # FP8 input with no scaling
                    (self.weight, None), # FP8 weight with no scaling
                    out=out_tensor,
                    m_indices=group_indices,
                    mask=mask
                )
            else:
                return m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (x, None),           # FP8 input with no scaling 
                    (self.weight, None), # FP8 weight with no scaling
                    out=out_tensor,
                    m_indices=group_indices
                )
            
        # Fallback implementation for non-FP8 inputs: process each group separately
        outputs = []
        for i in range(self.num_groups):
            group_mask = group_indices == i
            if group_mask.any():
                group_input = x[group_mask]
                group_output = torch.nn.functional.linear(
                    group_input,
                    self.weight[i],
                    self.bias[i] if self.bias is not None else None
                )
                outputs.append(group_output)
                
        return torch.cat(outputs, dim=0)

# Utility functions
def get_optimal_kernel_config(hidden_size: int, 
                              seq_len: int,
                              batch_size: int,
                              num_groups: int = 1) -> GEMMConfig:
    """Get optimal kernel configuration."""
    return get_best_configs(
        hidden_size=hidden_size,
        seq_len=seq_len,
        batch_size=batch_size
    )

def benchmark_gemm(fn, *args, **kwargs):
    """Benchmark GEMM operation."""
    return bench(fn, *args, **kwargs)

def compute_numerical_error(x: torch.Tensor, 
                            y: torch.Tensor) -> float:
    """Compute numerical error between tensors."""
    return calc_diff(x, y)

# Initialize optimal settings if CUDA is available
if torch.cuda.is_available():
    # Set number of SMs and select the first GPU by default
    num_sms = get_num_sms()
    torch.cuda.set_device(0)
