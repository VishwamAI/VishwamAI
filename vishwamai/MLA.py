"""Multi-head Locality-sensitive Attention implementation."""

import torch
import torch.nn as nn
from typing import Optional
from .base_layers import Linear
from .config import ModelArgs
from .utils import apply_rotary_emb

class MLA(nn.Module):
    """Multi-head Locality-sensitive Attention module."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Store configuration
        self.dim = args.dim
        self.n_heads = args.n_heads
        
        # Compute head dimensions
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        
        # Initialize projection matrices
        self.wq = Linear(self.dim, self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), bias=False)
        self.wk = Linear(self.dim, self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), bias=False)
        self.wv = Linear(self.dim, self.n_heads * self.v_head_dim, bias=False)
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the attention layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            start_pos: Starting position for relative attention
            freqs_cis: Precomputed frequency tensor for rotary embeddings
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to q, k, v
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Split heads and dimensions
        qnope, qrope = xq.split([
            self.n_heads * self.qk_nope_head_dim,
            self.n_heads * self.qk_rope_head_dim
        ], dim=-1)
        knope, krope = xk.split([
            self.n_heads * self.qk_nope_head_dim,
            self.n_heads * self.qk_rope_head_dim
        ], dim=-1)
        
        # Reshape for attention computation
        qnope = qnope.view(batch_size, seq_len, self.n_heads, self.qk_nope_head_dim)
        knope = knope.view(batch_size, seq_len, self.n_heads, self.qk_nope_head_dim)
        qrope = qrope.view(batch_size, seq_len, self.n_heads, self.qk_rope_head_dim)
        krope = krope.view(batch_size, seq_len, self.n_heads, self.qk_rope_head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.v_head_dim)
        
        # Apply rotary embeddings to RoPE parts
        qrope, krope = apply_rotary_emb(qrope, krope, freqs_cis=freqs_cis)
        
        # Concatenate NoRoPE and RoPE parts
        q = torch.cat([qnope, qrope], dim=-1)
        k = torch.cat([knope, krope], dim=-1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1), dtype=q.dtype))
        
        if mask is not None:
            scores = scores + mask
            
        # Apply attention
        attn = torch.softmax(scores.float(), dim=-1).type_as(scores)
        out = torch.matmul(attn, xv)
        
        # Reshape and project output
        out = out.reshape(batch_size, seq_len, -1)
        out = self.wo(out)
        
        return out
