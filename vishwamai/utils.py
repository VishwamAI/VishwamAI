"""Utility functions for the Transformer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Any, Optional

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute frequencies for positional embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

class Linear(nn.Module):
    """
    A custom Linear layer implementation with QK optimization and optional quantization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize: bool = False,
        qconfig: Optional[dict] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize = quantize
        self.qconfig = qconfig or {"dtype": "fp32"}

        # Initialize weights and optional bias
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        
        # Initialize parameters
        self.reset_parameters()
        
        # Setup quantization if enabled
        if quantize:
            self.scale = nn.Parameter(torch.ones(1))
            self.zero_point = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('scale', None)
            self.register_parameter('zero_point', None)

    def reset_parameters(self):
        """Initialize or reset layer parameters."""
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quantization."""
        # Quantize weights if enabled
        if self.quantize and self.training:
            weight = torch.quantize_per_tensor(
                self.weight,
                scale=self.scale,
                zero_point=self.zero_point.int(),
                dtype=torch.qint8
            )
            weight = weight.dequantize()
        else:
            weight = self.weight

        # Compute output
        output = F.linear(x, weight, self.bias)
        
        return output

    def extra_repr(self) -> str:
        """String representation with layer configuration."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'quantize={self.quantize}')

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors using the given frequency tensor."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat keys and values."""
    if n_rep == 1:
        return x
    bs, seqlen, n_kv_heads, head_dim = x.shape
    return (x[:, :, :, None, :]
            .expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, seqlen, n_kv_heads * n_rep, head_dim))
