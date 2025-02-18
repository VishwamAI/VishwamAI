import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Base Linear layer with proper handling of device and dtype.
class Linear(nn.Module):
    """Linear layer with optional quantization support."""
    dtype = None  # Class-level dtype

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Use class-level dtype if no dtype is provided
        if dtype is None:
            dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()

        # Initialize weights and bias using proper keyword arguments
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}')

# ColumnParallelLinear subclass that uses keyword arguments in the super() call.
class ColumnParallelLinear(Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        # For demonstration, we assume world_size is 1. In a real distributed setup, set world_size appropriately.
        world_size = 1  
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        
        # Use keyword arguments to ensure device and dtype are correctly passed
        super().__init__(in_features, self.part_out_features, bias=bias, device=device, dtype=dtype)
        
        # Dummy quantization initialization function (replace with your actual implementation)
        self.init_quantization(self.part_out_features, in_features)
    
    def init_quantization(self, out_features: int, in_features: int):
        # Placeholder for any quantization-related initialization
        pass

# Example usage:
if __name__ == "__main__":
    # Create an instance of ColumnParallelLinear with specified device and dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = ColumnParallelLinear(512, 1024, bias=True, device=device, dtype=torch.float32)
    
    # Perform a forward pass
    x = torch.randn(10, 512, device=device)
    y = layer(x)
    print("Output shape:", y.shape)
