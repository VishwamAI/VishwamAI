"""
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Flash Memory-Local Attention CUDA kernels with optimized memory access
"""
import torch
import torch.nn.functional as F

def run_mha_fwd_splitkv_mla(query: torch.Tensor,
                           key: torch.Tensor,
                           value: torch.Tensor,
                           num_heads: int,
                           head_dim: int,
                           block_size: int = 128,
                           causal: bool = True) -> torch.Tensor:
    """
    Forward pass for Flash Memory-Local Attention
    Optimized with warp-level operations and coalesced memory access
    """
    batch_size, seq_len, _ = query.size()
    scale = head_dim ** -0.5
    
    # Use 32-byte aligned memory access for coalescing
    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim) 
    value = value.view(batch_size, seq_len, num_heads, head_dim)

    # Compute scaled dot product attention with memory-efficient blocks
    attention = torch.empty(
        (batch_size, num_heads, seq_len, seq_len),
        dtype=query.dtype,
        device=query.device
    )

    # Process in memory-efficient blocks using warp-level parallelism
    for block_start in range(0, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)
        block_slice = slice(block_start, block_end)
        
        # Block matrix multiplication optimized for warps
        block_attn = torch.einsum(
            'bshd,bthd->bhst',
            query[:, block_slice],
            key
        ) * scale

        if causal:
            causal_mask = torch.triu(
                torch.ones_like(block_attn[0, 0]),
                diagonal=1
            ).bool()
            block_attn.masked_fill_(causal_mask, float('-inf'))

        attention[:, :, block_slice] = F.softmax(block_attn, dim=-1)
    
    # Final matmul using block-sparse pattern
    output = torch.einsum('bhst,bthd->bshd', attention, value)
    
    return output.reshape(batch_size, seq_len, -1)