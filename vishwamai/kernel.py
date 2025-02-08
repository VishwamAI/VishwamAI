from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Quantizes activations to FP8"""
    pid = tl.program_id(axis=0)
    row_idx = pid
    offs = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_elements

    x = tl.load(x_ptr + row_idx * n_elements + tl.arange(0, BLOCK_SIZE), mask=mask).to(tl.float32)
    row_max = tl.max(tl.abs(tl.where(mask, x, 0.0)))
    s = row_max / 448.

    y = tl.where(s > 0, x / s, x)
    y = tl.where(y > 127, 127.0, tl.where(y < -127, -127.0, y))

    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + row_idx, s)

def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    if not x.is_cuda:
        x = x.cuda()
    y = torch.empty_like(x, dtype=torch.float16, device='cuda')
    s = torch.empty(x.size(0), dtype=torch.float32, device='cuda')

    grid = (x.size(0),)
    act_quant_kernel[grid](x, y, s, x.size(1), BLOCK_SIZE=x.size(1))
    return y, s

@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + offs)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)

def weight_dequant(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    s = s.contiguous()
    if not x.is_cuda:
        x = x.cuda()
    if not s.is_cuda:
        s = s.cuda()

    assert x.dim() == 2 and s.dim() == 2, "Both tensors must be 2-dimensional"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float32, device='cuda')

    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return y

@triton.jit
def fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_s_ptr, b_s_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, K)

    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        a_scale = tl.load(a_s_ptr + offs_m)[:, None]
        b_scale = tl.load(b_s_ptr + offs_n)[None, :]
        a = a.to(tl.float32) * a_scale
        b = b.to(tl.float32) * b_scale
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_SIZE
        b_ptrs += BLOCK_SIZE * N

    offs_cm = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_cn = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

def fp8_gemm(input_tensor: torch.Tensor, scale_tensor: torch.Tensor,
             b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    a = input_tensor.contiguous().to(torch.float16).cuda()
    b = b.contiguous().to(torch.float16).cuda()
    a_s = scale_tensor.contiguous().cuda()
    b_s = b_s.contiguous().cuda()

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=torch.float16, device='cuda')

    BLOCK_SIZE = 32
    grid = (triton.cdiv(M, BLOCK_SIZE) * triton.cdiv(N, BLOCK_SIZE),)
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K, BLOCK_SIZE=BLOCK_SIZE)
    
    return c.to(torch.float32)
