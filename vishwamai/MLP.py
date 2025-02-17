import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel import ColumnParallelLinear, RowParallelLinear

class MLP(nn.Module):
    """Multi-Layer Perceptron used as a feed-forward layer."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
