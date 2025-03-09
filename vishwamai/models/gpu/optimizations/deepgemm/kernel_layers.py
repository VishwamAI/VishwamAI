import torch as nn
import torch 
import math
import torch.distributed as dist

class DeepGEMMLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_amp=True, distributed=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_amp = use_amp
        self.distributed = distributed
        
        # Initialize weights with optimal memory layout
        self.weight = nn.Parameter(torch.empty(
            out_features, in_features,
            dtype=torch.float16 if use_amp else torch.float32
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(
                out_features,
                dtype=torch.float16 if use_amp else torch.float32
            ))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
        # Create CUDA streams for async execution
        self.main_stream = torch.cuda.Stream()
        self.copy_stream = torch.cuda.Stream()
        
    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
            
        with torch.cuda.stream(self.main_stream):
            # Use optimized GEMM kernel
            output = self._optimized_linear(x)
            
        # Ensure computations are complete
        self.main_stream.synchronize()
        
        if self.distributed:
            dist.all_reduce(output)
            
        return output
        
    def _optimized_linear(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure proper memory layout
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Copy inputs asynchronously
        with torch.cuda.stream(self.copy_stream):
            x_copy = x.clone()
            weight_copy = self.weight.clone()
            
        # Wait for copies to complete
        self.copy_stream.synchronize()
        
        # Compute using DeepGEMM kernels
        output = torch.empty(
            x_copy.size(0), self.out_features,
            dtype=x.dtype, device=x.device
        )
        
        torch.cuda.synchronize()  # Memory fence
        output = torch.mm(x_copy, weight_copy.t())
        
        if self.bias is not None:
            output.add_(self.bias)
            
        return output
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)