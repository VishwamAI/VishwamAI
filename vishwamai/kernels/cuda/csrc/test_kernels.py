"""Test suite for VishwamAI optimized kernels."""

import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import pytest

from vishwamai.kernels.cuda.flashmla_cuda import (
    get_mla_metadata,
    flash_mla_forward
)

def generate_test_inputs(
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 8,
    head_dim: int = 64,
    device: str = "cuda"
) -> Tuple[torch.Tensor, ...]:
    """Generate test tensors for attention computation."""
    # Create random tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads//2, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads//2, head_dim, device=device)
    
    # Generate block table for KV cache
    block_table = torch.arange(seq_len, device=device).reshape(batch_size, -1)
    
    # Sequence lengths
    cu_seqlens = torch.arange(batch_size + 1, device=device) * seq_len
    
    # Other attention parameters
    scale = 1.0 / np.sqrt(head_dim)
    
    return q, k, v, block_table, cu_seqlens, scale

def test_flash_mla_metadata():
    """Test MLA metadata computation."""
    batch_size = 2
    seq_len = 128
    device = "cuda"
    
    # Create sequence lengths tensor
    seqlens = torch.ones(batch_size, device=device, dtype=torch.int32) * seq_len
    
    # Get metadata
    metadata = get_mla_metadata(
        seqlens,
        batch_size=batch_size,
        block_size_n=64,
        fixed_overhead_num_blocks=2,
        num_sm_parts=6
    )
    
    # Validate metadata shape
    assert metadata.size(0) == 2  # Should return [tile_metadata, num_splits]
    assert metadata[0].size(0) == 8  # tile_metadata should have 8 elements
    assert metadata[1].size(0) == batch_size  # num_splits should have batch_size elements

def test_flash_mla_correctness():
    """Test Flash MLA implementation against naive implementation."""
    # Generate inputs
    q, k, v, block_table, cu_seqlens, scale = generate_test_inputs()
    
    # Get metadata
    seqlens = torch.ones(q.size(0), device="cuda", dtype=torch.int32) * q.size(1)
    metadata = get_mla_metadata(
        seqlens,
        batch_size=q.size(0),
        block_size_n=64,
        fixed_overhead_num_blocks=2,
        num_sm_parts=6
    )
    
    # Run Flash MLA implementation
    flash_out, flash_lse = flash_mla_forward(
        q, k, v,
        block_table,
        cu_seqlens,
        scale,
        is_causal=True,
        tile_scheduler_metadata=metadata[0],
        num_splits=metadata[1]
    )
    
    # Compute reference implementation
    def naive_attention(q, k, v, scale):
        # [b, h, s, d] x [b, h, d, s] -> [b, h, s, s]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        mask = torch.triu(torch.ones_like(scores), diagonal=1) * -1e10
        scores = scores + mask
        
        # Apply softmax
        attn = torch.softmax(scores, dim=-1)
        
        # Compute weighted sum
        return torch.matmul(attn, v)
    
    # Reshape for naive implementation
    q_ref = q.transpose(1, 2)  # [b, s, h, d] -> [b, h, s, d]
    k_ref = k.transpose(1, 2).repeat(1, 2, 1, 1)  # Handle grouped QK ratio
    v_ref = v.transpose(1, 2).repeat(1, 2, 1, 1)
    
    # Run reference implementation
    ref_out = naive_attention(q_ref, k_ref, v_ref, scale)
    ref_out = ref_out.transpose(1, 2)  # [b, h, s, d] -> [b, s, h, d]
    
    # Compare results
    torch.testing.assert_close(
        flash_out, 
        ref_out,
        rtol=1e-3,
        atol=1e-3,
        msg="Flash MLA output differs from reference implementation"
    )

def test_flash_mla_performance():
    """Test Flash MLA performance against naive implementation."""
    import time
    
    # Generate larger inputs for timing
    batch_size = 4
    seq_len = 1024
    num_heads = 16
    head_dim = 64
    
    q, k, v, block_table, cu_seqlens, scale = generate_test_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim
    )
    
    # Get metadata
    seqlens = torch.ones(batch_size, device="cuda", dtype=torch.int32) * seq_len
    metadata = get_mla_metadata(
        seqlens,
        batch_size=batch_size,
        block_size_n=64,
        fixed_overhead_num_blocks=2,
        num_sm_parts=6
    )
    
    # Time Flash MLA implementation
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        flash_out, _ = flash_mla_forward(
            q, k, v,
            block_table,
            cu_seqlens,
            scale,
            is_causal=True,
            tile_scheduler_metadata=metadata[0],
            num_splits=metadata[1]
        )
    
    torch.cuda.synchronize()
    flash_time = time.time() - start
    
    # Time naive implementation
    def naive_attention(q, k, v, scale):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones_like(scores), diagonal=1) * -1e10
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
    
    q_ref = q.transpose(1, 2)
    k_ref = k.transpose(1, 2).repeat(1, 2, 1, 1)
    v_ref = v.transpose(1, 2).repeat(1, 2, 1, 1)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        ref_out = naive_attention(q_ref, k_ref, v_ref, scale)
    
    torch.cuda.synchronize()
    naive_time = time.time() - start
    
    # Flash MLA should be significantly faster
    assert flash_time < naive_time * 0.5, \
        f"Flash MLA ({flash_time:.3f}s) not significantly faster than naive ({naive_time:.3f}s)"

if __name__ == "__main__":
    # Run all tests
    test_flash_mla_metadata()
    test_flash_mla_correctness()
    test_flash_mla_performance()
    print("All tests passed!")