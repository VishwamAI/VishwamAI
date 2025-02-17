"""Base layer implementations."""

import torch
import torch.nn as nn

class Linear(nn.Linear):
    """Custom Linear layer with dtype support."""
    dtype = torch.bfloat16  # Default dtype

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = None):
        super().__init__(in_features, out_features, bias)
        self._dtype = dtype or self.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight.to(dtype=self._dtype), 
                                 self.bias.to(dtype=self._dtype) if self.bias is not None else None)
