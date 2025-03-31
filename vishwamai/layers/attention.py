"""Flash Attention implementations for VishwamAI.

This module provides efficient attention mechanisms using Flash Attention
and related optimizations for transformer models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FlashAttention(nn.Module):
    """Flash Attention implementation with optimized memory usage."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_seq_length: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with memory-efficient flash attention."""
        batch_size, seq_len, _ = q.shape
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)
        
        # Apply flash attention
        output = flash_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            key_padding_mask=key_padding_mask,
            dropout=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        
        # Reshape output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.dim)
        
        return output

    @staticmethod
    def _compute_max_positions(seq_len: int, block_size: int) -> int:
        """Compute maximum sequence positions for block-wise attention."""
        return math.ceil(seq_len / block_size) * block_size

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scale: float = None,
    block_size: int = 1024,
) -> torch.Tensor:
    """Compute attention using Flash Attention algorithm with block-wise optimization."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    if scale is None:
        scale = head_dim ** -0.5
    
    # Initialize output
    output = torch.zeros_like(q)
    
    # Block-wise computation
    for block_start in range(0, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)
        
        # Current block of queries
        q_block = q[:, :, block_start:block_end]
        
        # Compute attention scores for current block
        scores = torch.matmul(q_block, k.transpose(-2, -1)) * scale
        
        # Apply masking if provided
        if mask is not None:
            mask_block = mask[:, :, block_start:block_end] if mask.dim() == 4 else mask[:, block_start:block_end]
            scores = scores.masked_fill(mask_block == 0, float('-inf'))
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
        
        # Improved numerical stability for softmax
        max_score = torch.max(scores, dim=-1, keepdim=True)[0]
        scores = scores - max_score
        
        # Compute attention weights with improved stability
        exp_scores = torch.exp(scores)
        attention_weights = exp_scores / (torch.sum(exp_scores, dim=-1, keepdim=True) + 1e-6)
        
        if dropout > 0:
            attention_weights = F.dropout(attention_weights, p=dropout)
        
        # Compute output for current block
        output[:, :, block_start:block_end] = torch.matmul(attention_weights, v)
    
    return output

def flash_attention_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Optimized Flash Attention for inference with KV caching."""
    if past_kv is not None:
        past_key, past_value = past_kv
        k = torch.cat([past_key, k], dim=2)
        v = torch.cat([past_value, v], dim=2)
    
    output = flash_attention(q, k, v)
    return output, (k, v)

def create_flash_attention(config):
    """Factory function to create Flash Attention instance."""
    return FlashAttention(
        dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        dropout=config.attention_dropout,
        max_seq_length=config.max_position_embeddings
    )

def mha_with_flash_attention(dim, num_heads=8):
    """Helper function to create multi-head attention with Flash Attention."""
    return FlashAttention(dim=dim, num_heads=num_heads)