"""
GPU-optimized kernel layers for VishwamAI.
Implements efficient matrix operations using DeepGEMM and optimized kernels.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import math
import time

# Import optimized GEMM implementations
from vishwamai.models.gpu.optimizations.deepgemm import (
    linear_forward,
    linear_backward,
    get_best_config,
    set_num_warps
)
from vishwamai.models.gpu.optimizations.deep_ep.utils import get_num_sms

class DeepGEMMLinear(nn.Linear):
    """Linear layer optimized with DeepGEMM"""
    
    def __init__(self, in_features, out_features, bias=True, 
                 use_amp=True, cache_dir=None):
        super().__init__(in_features, out_features, bias)
        self.use_amp = use_amp
        self.cache_dir = cache_dir
        
        # Get optimal GEMM configuration for typical sizes
        self.gemm_config = get_best_config(
            (out_features, in_features),
            num_sms=get_num_sms()
        )
        
        # Cache warmup state
        self._is_warmed_up = False
        
    def _warmup(self, x):
        """Run warmup pass to optimize kernels"""
        if not self._is_warmed_up and torch.cuda.is_available():
            # Run forward and backward pass with dummy data
            dummy_out = self.forward(x)
            if self.training:
                dummy_out.mean().backward()
            torch.cuda.synchronize()
            self._is_warmed_up = True
            
    @autocast()
    def forward(self, x):
        """Optimized forward pass using DeepGEMM"""
        if not self._is_warmed_up:
            self._warmup(x)
            
        return linear_forward(
            x, self.weight, self.bias,
            config=self.gemm_config,
            use_amp=self.use_amp
        )
        
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, use_amp={self.use_amp}'

class DeepGEMMLayerNorm(nn.LayerNorm):
    """Layer normalization optimized for GPU"""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 use_amp=True, cache_dir=None):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.use_amp = use_amp
        self.cache_dir = cache_dir
        
    @autocast()
    def forward(self, x):
        """Forward pass with mixed precision"""
        if self.use_amp and torch.cuda.is_available():
            with autocast():
                return super().forward(x)
        return super().forward(x)

class DeepGEMMEmbedding(nn.Embedding):
    """Embedding layer with optimized memory access"""
    
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, use_amp=True, cache_dir=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                        norm_type, scale_grad_by_freq, sparse, _weight)
        self.use_amp = use_amp
        self.cache_dir = cache_dir
        
    @autocast()
    def forward(self, x):
        """Forward pass with mixed precision"""
        if self.use_amp and torch.cuda.is_available():
            with autocast():
                return super().forward(x)
        return super().forward(x)

# Utility functions
def get_optimal_warps(matrix_shape):
    """Get optimal number of warps for matrix shape"""
    if not torch.cuda.is_available():
        return 8  # Default for CPU
        
    # Scale warps based on matrix size and GPU capability
    m, n = matrix_shape
    num_sms = get_num_sms()
    
    if max(m, n) <= 512:
        warps = 4
    elif max(m, n) <= 2048:
        warps = 8
    else:
        warps = 16
        
    # Adjust for available SMs
    warps = min(warps, num_sms * 2)
    return warps

def init_kernels():
    """Initialize GEMM kernels with optimal settings"""
    if torch.cuda.is_available():
        num_sms = get_num_sms()
        set_num_warps(num_sms * 2)  # 2 warps per SM by default
        
# Initialize kernels on import
init_kernels()
