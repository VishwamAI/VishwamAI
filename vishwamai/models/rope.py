"""
Rotary Positional Embeddings (RoPE) implementation.

This module implements RoPE as described in the paper:
"RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embeddings for enhanced position encoding.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Generate and cache position embeddings
        self.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        )
        self.register_buffer("inv_freq", self.inv_freq)
        
        # Create cache for faster forward pass
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        
    def _update_cos_sin_cache(self, x: torch.Tensor, seq_dim: int = 1):
        """Update cached cos and sin values for given sequence length."""
        seq_len = x.shape[seq_dim]
        
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(
                seq_len,
                device=x.device,
                dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
            
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_dim: int = 1
    ) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_heads, head_dim]
            position_ids: Optional position IDs [batch_size, seq_len]
            seq_dim: Dimension corresponding to sequence length
            
        Returns:
            Tensor with positional information encoded via rotation
        """
        # Update cache if needed
        self._update_cos_sin_cache(x, seq_dim)
        
        if position_ids is not None:
            # Use provided position IDs
            cos = F.embedding(
                position_ids,
                self._cos_cached.squeeze(0).squeeze(0)
            ).unsqueeze(-2)
            sin = F.embedding(
                position_ids,
                self._sin_cached.squeeze(0).squeeze(0)
            ).unsqueeze(-2)
        else:
            # Use cached values
            cos = self._cos_cached[:, :, :x.shape[seq_dim]]
            sin = self._sin_cached[:, :, :x.shape[seq_dim]]
            
        # Reshape input for rotation
        x_reshape = x.view(*x.shape[:-1], -1, self.dim // 2, 2)
        
        # Prepare cos/sin for rotation
        cos = cos.view(cos.shape[:-1] + (cos.shape[-1] // 2, 2))
        sin = sin.view(sin.shape[:-1] + (sin.shape[-1] // 2, 2))
        
        # Apply rotation using complex multiplication
        x_out = torch.stack([
            x_reshape[..., 0] * cos[..., 0] - x_reshape[..., 1] * sin[..., 0],
            x_reshape[..., 1] * cos[..., 1] + x_reshape[..., 0] * sin[..., 1]
        ], dim=-1)
        
        # Reshape back to original shape
        x_out = x_out.view(*x.shape)
        
        return x_out
        
    def _compute_rope_embeddings(
        self,
        positions: torch.Tensor,
        dim: int,
        base: int = 10000,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE embeddings for given positions.
        
        Args:
            positions: Position tensor
            dim: Dimension of embeddings
            base: Base for frequency computation
            device: Device to create tensors on
            
        Returns:
            Tuple of (cos, sin) embeddings
        """
        # Generate frequency bands
        freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        
        # Create position-frequency product
        angles = torch.outer(positions, freq)
        
        # Compute cos and sin
        cos = torch.cos(angles).view(*positions.shape, -1)
        sin = torch.sin(angles).view(*positions.shape, -1)
        
        return cos, sin
        
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f'dim={self.dim}, '
            f'max_position_embeddings={self.max_position_embeddings}, '
            f'base={self.base}'
        )
