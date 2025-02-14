import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel_cpu(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Compute matrix multiplication C = A @ B using CPU-optimized Triton kernel
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Get program ID for each dimension
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Compute offsets for each dimension
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks from A and B
        a = tl.load(a_ptr + (offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak))
        b = tl.load(b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn))
        
        # Compute matrix multiplication for current block
        acc += tl.dot(a, b)
    
    # Write output
    c = acc.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c)

class TritonLinearCPU(torch.nn.Module):
    """
    CPU-optimized linear layer using Triton kernels
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reshape input for matmul
        batch_size = input.size(0)
        input_reshaped = input.view(-1, self.in_features)
        
        # Compute output shape
        output_shape = list(input.size())
        output_shape[-1] = self.out_features
        
        # Initialize output tensor
        output = torch.empty(
            (input_reshaped.size(0), self.out_features),
            device=input.device,
            dtype=input.dtype
        )
        
        # Launch Triton kernel for matrix multiplication
        grid = lambda META: (
            triton.cdiv(input_reshaped.size(0), META['BLOCK_SIZE_M']) * \
            triton.cdiv(self.out_features, META['BLOCK_SIZE_N']),
        )
        
        matmul_kernel_cpu[grid](
            input_reshaped, self.weight, output,
            input_reshaped.size(0), self.out_features, self.in_features,
            input_reshaped.stride(0), input_reshaped.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias
        
        # Reshape output back to original dimensions
        return output.view(output_shape)

# Helper functions for common operations
@triton.jit
def layer_norm_kernel_cpu(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    stride_x, stride_mean, stride_var,
    n_cols,
    eps: tl.constexpr
):
    """
    Optimized LayerNorm implementation for CPU
    """
    row_idx = tl.program_id(0)
    col_block = tl.arange(0, n_cols)
    
    # Load data
    x = tl.load(x_ptr + row_idx * stride_x + col_block)
    mean = tl.load(mean_ptr + row_idx * stride_mean)
    var = tl.load(var_ptr + row_idx * stride_var)
    weight = tl.load(weight_ptr + col_block)
    bias = tl.load(bias_ptr + col_block)
    
    # Normalize
    x_hat = (x - mean) / tl.sqrt(var + eps)
    out = x_hat * weight + bias
    
    # Store result
    tl.store(out_ptr + row_idx * stride_x + col_block, out)

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    CPU-optimized LayerNorm using Triton
    """
    means = x.mean(dim=-1, keepdim=True)
    vars = x.var(dim=-1, keepdim=True, unbiased=False)
    
    grid = (x.size(0),)
    
    # Launch kernel
    layer_norm_kernel_cpu[grid](
        x, means, vars, weight, bias, x,
        x.stride(0), means.stride(0), vars.stride(0),
        x.size(-1),
        eps
    )
    return x
