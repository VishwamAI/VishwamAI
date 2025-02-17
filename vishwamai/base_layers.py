"""Base layer implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class Linear(nn.Module):
    """Linear layer with optional quantization support."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize or reset layer parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        """String representation."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}')
