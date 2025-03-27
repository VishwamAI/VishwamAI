"""Common CUDA kernel templates."""

from typing import Dict
from .compiler import KernelTemplate

MATMUL_TEMPLATE = KernelTemplate("""
// Matrix multiplication kernel template
// {block_size} is the block size
// {dtype} is the data type (float, half, etc)
extern "C" __global__ void {name}({dtype}* A, {dtype}* B, {dtype}* C, 
                                 int M, int N, int K) {{
    int row = blockIdx.y * {block_size} + threadIdx.y;
    int col = blockIdx.x * {block_size} + threadIdx.x;
    {dtype} sum = 0;
    
    for (int i = 0; i < K; i++) {{
        sum += A[row * K + i] * B[i * N + col];
    }}
    
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
""")

ELEMENTWISE_TEMPLATE = KernelTemplate("""
extern "C" __global__ void {name}({dtype}* input, {dtype}* output, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        output[idx] = {operation}(input[idx]);
    }}
}}
""")

REDUCTION_TEMPLATE = KernelTemplate("""
extern "C" __global__ void {name}({dtype}* input, {dtype}* output, 
                                 int n, {dtype} init_val) {{
    __shared__ {dtype} sdata[{block_size}];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * {block_size} * 2 + tid;
    
    {dtype} sum = init_val;
    if(idx < n) {{
        sum = input[idx];
        if(idx + {block_size} < n) {{
            sum = {operation}(sum, input[idx + {block_size}]);
        }}
    }}
    
    sdata[tid] = sum;
    __syncthreads();
    
    for(int s = {block_size}/2; s > 0; s >>= 1) {{
        if(tid < s) {{
            sdata[tid] = {operation}(sdata[tid], sdata[tid + s]);
        }}
        __syncthreads();
    }}
    
    if(tid == 0) output[blockIdx.x] = sdata[0];
}}
""")

# Map of template names to template objects
TEMPLATES: Dict[str, KernelTemplate] = {
    "matmul": MATMUL_TEMPLATE,
    "elementwise": ELEMENTWISE_TEMPLATE,
    "reduction": REDUCTION_TEMPLATE
}

def get_template(name: str) -> KernelTemplate:
    """Get a kernel template by name."""
    if name not in TEMPLATES:
        raise KeyError(f"Template {name} not found")
    return TEMPLATES[name]