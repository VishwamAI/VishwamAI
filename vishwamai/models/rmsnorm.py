"""
Root Mean Square Layer Normalization implementation.

This module provides an efficient implementation of RMSNorm,
which is a simplified variant of Layer Normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Paper: https://arxiv.org/abs/1910.07467
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        partial: float = -1,
        bias: bool = False
    ):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.bias = bias
        self.partial = partial
        
        self.scale = nn.Parameter(torch.ones(hidden_size))
        if bias:
            self.offset = nn.Parameter(torch.zeros(hidden_size))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        torch.nn.init.ones_(self.scale)
        if self.bias:
            torch.nn.init.zeros_(self.offset)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Select normalization dims
        if self.partial > 0:
            norm_dims = tuple(range(x.ndim - 1)) + (-1,)
            d = self.partial
        else:
            norm_dims = -1
            d = x.shape[-1]
            
        # Compute RMS
        rms = torch.sqrt(
            torch.mean(x * x, dim=norm_dims, keepdim=True) + self.eps
        )
        
        # Normalize and scale
        x_norm = x / rms * self.scale
        
        # Add bias if using
        if self.bias:
            x_norm = x_norm + self.offset
            
        return x_norm
        
    def extra_repr(self) -> str:
        """String representation."""
        return f'hidden_size={self.hidden_size}, eps={self.eps}'
