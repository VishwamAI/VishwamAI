"""
Memory-Efficient Attention Implementation
======================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.cuda.amp import autocast

class FlashMultiHeadAttention(nn.Module):
    """Memory-efficient attention using chunked computation"""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, batch_size: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape  # batch, length, dim
        
        # Memory-efficient chunked QKV projection
        chunk_size = 128  # Adjust based on available memory
        q_chunks = []
        k_chunks = []
        v_chunks = []
        
        for i in range(0, L, chunk_size):
            chunk = x[:, i:min(i + chunk_size, L)]
            with autocast(enabled=True):
                qkv_chunk = self.qkv(chunk)
                q_chunk, k_chunk, v_chunk = qkv_chunk.chunk(3, dim=-1)
                
                # Reshape chunks
                q_chunk = q_chunk.view(-1, chunk.size(1), self.num_heads, self.head_dim).transpose(1, 2)
                k_chunk = k_chunk.view(-1, chunk.size(1), self.num_heads, self.head_dim).transpose(1, 2)
                v_chunk = v_chunk.view(-1, chunk.size(1), self.num_heads, self.head_dim).transpose(1, 2)
                
                q_chunks.append(q_chunk)
                k_chunks.append(k_chunk)
                v_chunks.append(v_chunk)
        
        # Concatenate chunks
        q = torch.cat(q_chunks, dim=2)
        k = torch.cat(k_chunks, dim=2)
        v = torch.cat(v_chunks, dim=2)
        
        # Flash attention implementation
        with autocast(enabled=True):
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q / scale, k.transpose(-2, -1))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        
        # Memory-efficient output computation
        out_chunks = []
        chunk_size = min(128, L)  # Adjust based on GPU memory
        
        for i in range(0, L, chunk_size):
            with autocast(enabled=True):
                # Process attention output in chunks
                attn_chunk = attn[:, :, :, i:min(i + chunk_size, L)]
                v_chunk = v[:, :, i:min(i + chunk_size, L)]
                out_chunk = torch.matmul(attn_chunk, v_chunk)
                out_chunks.append(out_chunk)
        
        out = torch.cat(out_chunks, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        
        # Final projection
        out = self.proj(out)
        out = self.drop(out)
        
        return out

class SelfAttention(FlashMultiHeadAttention):
    """Self-attention layer for transformer architecture."""
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(x, mask)

class CrossAttention(FlashMultiHeadAttention):
    """Cross-attention layer for encoder-decoder attention."""
    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Memory-efficient cross attention
        chunk_size = 128  # Adjust based on memory
        q_chunks = []
        k_chunks = []
        v_chunks = []
        
        # Process query chunks from x
        for i in range(0, L, chunk_size):
            with autocast(enabled=True):
                x_chunk = x[:, i:min(i + chunk_size, L)]
                q_chunk = self.qkv(x_chunk).chunk(3, dim=-1)[0]
                q_chunks.append(q_chunk)
        
        # Process key/value chunks from context
        for i in range(0, context.size(1), chunk_size):
            with autocast(enabled=True):
                ctx_chunk = context[:, i:min(i + chunk_size, context.size(1))]
                k_chunk, v_chunk = self.qkv(ctx_chunk).chunk(3, dim=-1)[1:]
                k_chunks.append(k_chunk)
                v_chunks.append(v_chunk)
        
        # Concatenate chunks
        q = torch.cat(q_chunks, dim=1)
        k = torch.cat(k_chunks, dim=1)
        v = torch.cat(v_chunks, dim=1)
        
        # Reshape to multiple heads
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, L, -1)
        
        out = self.proj(out)
        out = self.drop(out)
        
        return out

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention mechanism."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        out = self.drop(out)
        
        return out
