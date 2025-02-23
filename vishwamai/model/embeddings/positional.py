"""Positional encoding implementations for transformers."""

import math
from typing import Optional, Literal

import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""
    
    def __init__(
        self,
        embedding_dim: int,
        max_seq_length: int = 2048,
        dropout_prob: float = 0.1,
        batch_first: bool = True,
        scale: Optional[float] = None,
    ):
        """Initialize sinusoidal positional encoding.
        
        Args:
            embedding_dim: Dimension of embeddings (must be even)
            max_seq_length: Maximum sequence length
            dropout_prob: Dropout probability
            batch_first: Whether batch dimension is first
            scale: Optional scale factor for embeddings
        """
        super().__init__()
        
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {embedding_dim}")
            
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.batch_first = batch_first
        self.scale = scale or 1.0
        
        # Create position encoding buffer
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if batch_first:
            pe = pe.unsqueeze(0)  # [1, max_seq_length, embedding_dim]
        else:
            pe = pe.unsqueeze(1)  # [max_seq_length, 1, embedding_dim]
            
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0
    ) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim] if batch_first
               or [seq_length, batch_size, embedding_dim] otherwise
            offset: Starting position offset
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1 if self.batch_first else 0)
        if offset + seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {offset + seq_len} exceeds maximum length {self.max_seq_length}"
            )
            
        pos_enc = self.pe[:, offset:offset+seq_len] if self.batch_first else self.pe[offset:offset+seq_len]
        x = x + (pos_enc * self.scale)
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    """Learned absolute positional encoding."""
    
    def __init__(
        self,
        embedding_dim: int,
        max_seq_length: int = 2048,
        dropout_prob: float = 0.1,
        batch_first: bool = True,
    ):
        """Initialize learned positional encoding.
        
        Args:
            embedding_dim: Dimension of embeddings
            max_seq_length: Maximum sequence length
            dropout_prob: Dropout probability
            batch_first: Whether batch dimension is first
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.batch_first = batch_first
        
        self.weight = nn.Parameter(torch.empty(max_seq_length, embedding_dim))
        self.dropout = nn.Dropout(dropout_prob)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize position embedding weights."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0
    ) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim] if batch_first
               or [seq_length, batch_size, embedding_dim] otherwise
            offset: Starting position offset
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1 if self.batch_first else 0)
        if offset + seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {offset + seq_len} exceeds maximum length {self.max_seq_length}"
            )
            
        pos_emb = self.weight[offset:offset+seq_len]
        if self.batch_first:
            pos_emb = pos_emb.unsqueeze(0)  # [1, seq_length, embedding_dim]
        else:
            pos_emb = pos_emb.unsqueeze(1)  # [seq_length, 1, embedding_dim]
            
        x = x + pos_emb
        return self.dropout(x)

class RotaryPositionalEncoding(nn.Module):
    """Rotary position embeddings (RoPE)."""
    
    def __init__(
        self,
        embedding_dim: int,
        max_seq_length: int = 2048,
        base: int = 10000,
        scale_base: Optional[float] = None,
        scaling_type: Literal["linear", "dynamic", "none"] = "none",
    ):
        """Initialize rotary position encoding.
        
        Args:
            embedding_dim: Dimension of embeddings (must be even)
            max_seq_length: Maximum sequence length
            base: Base for frequency computation
            scale_base: Optional base for dynamic scaling
            scaling_type: Type of position scaling to apply
        """
        super().__init__()
        
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {embedding_dim}")
            
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.base = base
        self.scale_base = scale_base
        self.scaling_type = scaling_type
        
        # Create position encoding buffers
        position = torch.arange(max_seq_length, dtype=torch.float)
        freqs = 1.0 / (base ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        theta = position.unsqueeze(1) * freqs.unsqueeze(0)  # [max_seq_length, dim/2]
        
        # Create and register sin/cos buffers
        cos_pos = torch.cos(theta).repeat_interleave(2, dim=-1)  # [max_seq_length, dim]
        sin_pos = torch.sin(theta).repeat_interleave(2, dim=-1)  # [max_seq_length, dim]
        self.register_buffer('cos_pos', cos_pos)
        self.register_buffer('sin_pos', sin_pos)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position encoding to query and key tensors.
        
        Args:
            q: Query tensor
            k: Key tensor
            seq_len: Optional sequence length override
            offset: Starting position offset
            
        Returns:
            Tuple of transformed (query, key) tensors
        """
        if seq_len is None:
            seq_len = q.size(-2)
            
        if offset + seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {offset + seq_len} exceeds maximum length {self.max_seq_length}"
            )
            
        cos = self.cos_pos[offset:offset+seq_len]
        sin = self.sin_pos[offset:offset+seq_len]
        
        # Apply position scaling if enabled
        if self.scaling_type != "none":
            scale = self._compute_position_scale(seq_len)
            cos = cos * scale
            sin = sin * scale
            
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        
        # Apply rotary transformation
        def rotate(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([
                x1 * cos - x2 * sin,
                x2 * cos + x1 * sin,
            ], dim=-1)
            
        return rotate(q), rotate(k)
    
    def _compute_position_scale(self, seq_len: int) -> torch.Tensor:
        """Compute position-dependent scaling factor.
        
        Args:
            seq_len: Current sequence length
            
        Returns:
            Scaling tensor of shape [seq_len, 1]
        """
        if self.scaling_type == "linear":
            # Linear scaling based on sequence length
            scale = 1.0 / seq_len
        elif self.scaling_type == "dynamic":
            # Dynamic scaling using log base
            if self.scale_base is None:
                raise ValueError("scale_base must be provided for dynamic scaling")
            scale = torch.log(torch.arange(seq_len) + 1) / math.log(self.scale_base)
        else:
            scale = 1.0
            
        return scale.to(self.cos_pos.device)
    
    def extend_max_seq_length(self, new_max_length: int):
        """Extend maximum sequence length.
        
        Args:
            new_max_length: New maximum sequence length
        """
        if new_max_length <= self.max_seq_length:
            return
            
        # Create extended position encodings
        position = torch.arange(new_max_length, dtype=torch.float)
        freqs = 1.0 / (self.base ** (torch.arange(0, self.embedding_dim, 2).float() / self.embedding_dim))
        theta = position.unsqueeze(1) * freqs.unsqueeze(0)
        
        # Update sin/cos buffers
        cos_pos = torch.cos(theta).repeat_interleave(2, dim=-1)
        sin_pos = torch.sin(theta).repeat_interleave(2, dim=-1)
        
        self.register_buffer('cos_pos', cos_pos, persistent=False)
        self.register_buffer('sin_pos', sin_pos, persistent=False)
        self.max_seq_length = new_max_length
