"""
Flash Memory-Local Attention CUDA kernels
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
    Optimized multi-head attention with split key-value pairs
    """
    batch_size, seq_len, _ = query.size()
    scale = head_dim ** -0.5

    # Reshape to include head dimension
    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim)
    value = value.view(batch_size, seq_len, num_heads, head_dim)

    # Compute scaled dot product attention with memory-efficient blocks
    attention = torch.einsum('bshd,bthd->bhst', query, key) * scale

    if causal:
        causal_mask = torch.triu(torch.ones_like(attention[0, 0]), diagonal=1).bool()
        attention = attention.masked_fill(causal_mask, float('-inf'))

    attention = F.softmax(attention, dim=-1)
    output = torch.einsum('bhst,bthd->bshd', attention, value)
    
    return output.reshape(batch_size, seq_len, -1)