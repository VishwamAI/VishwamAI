from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config
import torch.nn.functional as F
import math


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


def optimize_kernel_layout(
    weight: torch.Tensor,
    block_size: int = 64,
    transpose: bool = False
) -> torch.Tensor:
    """
    Optimize kernel memory layout for efficient computation.
    
    Args:
        weight: Input weight tensor
        block_size: Block size for tiling
        transpose: Whether to transpose the weight matrix
        
    Returns:
        Optimized weight tensor
    """
    if transpose:
        weight = weight.transpose(-2, -1)
        
    if weight.dim() < 2:
        return weight
        
    original_shape = weight.shape
    in_features = original_shape[-2]
    out_features = original_shape[-1]
    
    # Ensure dimensions are multiples of block_size
    pad_in = (block_size - in_features % block_size) % block_size
    pad_out = (block_size - out_features % block_size) % block_size
    
    if pad_in > 0 or pad_out > 0:
        weight = F.pad(weight, (0, pad_out, 0, pad_in))
    
    # Reshape into blocks
    blocked_shape = weight.shape[:-2] + (
        weight.shape[-2] // block_size,
        block_size,
        weight.shape[-1] // block_size,
        block_size
    )
    weight = weight.reshape(blocked_shape)
    
    # Optimize layout
    weight = weight.permute(
        *range(weight.dim() - 4),
        0, 2, 1, 3  # Reorder block dimensions
    )
    
    # Restore original dimensions
    final_shape = original_shape[:-2] + (
        math.ceil(in_features / block_size) * block_size,
        math.ceil(out_features / block_size) * block_size
    )
    weight = weight.reshape(final_shape)
    
    # Remove padding if added
    if pad_in > 0 or pad_out > 0:
        weight = weight[..., :in_features, :out_features]
        
    return weight

def prepare_kernel(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
    quantize: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepare kernel for optimized computation.
    
    Args:
        weight: Input weight tensor
        scale: Optional quantization scale
        zero_point: Optional quantization zero point
        quantize: Whether to quantize the weights
        
    Returns:
        Tuple of (prepared weight tensor, optional quantization params)
    """
    # Optimize memory layout
    weight = optimize_kernel_layout(weight)
    
    if not quantize:
        return weight, None
        
    if scale is None:
        # Calculate dynamic quantization parameters
        weight_abs_max = weight.abs().max()
        scale = weight_abs_max / 127.0  # For int8 quantization
        zero_point = 0
    
    # Quantize weights
    weight_quant = torch.quantize_per_tensor(
        weight,
        scale.item(),
        zero_point,
        torch.qint8
    )
    
    return weight_quant, {"scale": scale, "zero_point": zero_point}

def fuse_kernels(kernels: list) -> torch.Tensor:
    """
    Fuse multiple kernels into a single optimized kernel.
    
    Args:
        kernels: List of kernel tensors to fuse
        
    Returns:
        Fused kernel tensor
    """
    if not kernels:
        raise ValueError("No kernels provided for fusion")
        
    # Ensure all kernels have compatible shapes
    out_features = kernels[0].shape[-1]
    in_features_total = sum(k.shape[-2] for k in kernels)
    
    # Optimize each kernel
    optimized_kernels = [optimize_kernel_layout(k) for k in kernels]
    
    # Concatenate along input dimension
    fused = torch.cat(optimized_kernels, dim=-2)
    
    # Final layout optimization
    return optimize_kernel_layout(fused)


fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c