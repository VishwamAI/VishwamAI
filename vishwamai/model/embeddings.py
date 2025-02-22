"""
Embedding components for Vishwamai model
"""
import math
import torch
import torch.nn as nn
from typing import Optional

class TokenEmbedding(nn.Module):
    """Token embedding with precision support"""
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
            dtype=dtype
        )
        self.hidden_size = hidden_size
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings with proper scaling"""
        embeddings = self.embedding(input_ids)
        # Scale embeddings by sqrt(hidden_size)
        return embeddings * math.sqrt(self.hidden_size)

class PositionalEmbedding(nn.Module):
    """
    Positional embedding with support for different types and precisions
    """
    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
        position_embedding_type: str = "RoPE",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.position_embedding_type = position_embedding_type
        self.dtype = dtype
        
        if position_embedding_type == "learned":
            self.position_embeddings = nn.Embedding(
                max_position_embeddings,
                hidden_size,
                dtype=dtype
            )
        elif position_embedding_type == "RoPE":
            # RoPE doesn't need learned parameters
            self.register_buffer(
                "inv_freq",
                1.0 / (10000 ** (torch.arange(0, hidden_size, 2).float() / hidden_size)),
                persistent=False
            )
        else:
            raise ValueError(f"Unknown position embedding type: {position_embedding_type}")
            
    def _get_rotary_embeddings(
        self,
        positions: torch.Tensor,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Generate rotary position embeddings"""
        dtype = dtype or self.dtype
        
        # Generate sinusoidal pattern
        t = positions.float().unsqueeze(1) * self.inv_freq.to(positions.device)
        freqs = torch.cat([t, t], dim=-1).to(dtype)
        
        # Generate rotations
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        
        # Reshape to match hidden size
        return emb.view(*positions.shape, self.hidden_size)
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get positional embeddings"""
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0)
        
        if self.position_embedding_type == "learned":
            return self.position_embeddings(position_ids)
        else:  # RoPE
            return self._get_rotary_embeddings(position_ids)

def apply_rotary_embeddings(
    x: torch.Tensor,
    position_embeddings: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor"""
    # Split features for rotation
    x_rot, x_pass = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
    # Get cos and sin components
    cos = position_embeddings[..., :x.shape[-1]//2]
    sin = position_embeddings[..., x.shape[-1]//2:]
    
    # Apply rotation
    x_rot = torch.cat([
        x_rot[..., ::2] * cos - x_rot[..., 1::2] * sin,
        x_rot[..., 1::2] * cos + x_rot[..., ::2] * sin
    ], dim=-1)
    
    # Concatenate with pass-through features
    return torch.cat([x_rot, x_pass], dim=-1)

class AxialPositionalEmbedding(nn.Module):
    """
    Axial positional embeddings for 2D attention patterns
    """
    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
        axial_shape: tuple = (32, 32),
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        assert (
            axial_shape[0] * axial_shape[1] >= max_position_embeddings
        ), "Axial shape must cover max position embeddings"
        
        self.shape = axial_shape
        self.row_embeddings = nn.Embedding(
            axial_shape[0],
            hidden_size // 2,
            dtype=dtype
        )
        self.col_embeddings = nn.Embedding(
            axial_shape[1],
            hidden_size - hidden_size // 2,  # Handle odd hidden sizes
            dtype=dtype
        )
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get axial positional embeddings"""
        seq_length = input_ids.size(1)
        row_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device
        ) // self.shape[1]
        col_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device
        ) % self.shape[1]
        
        row_embeddings = self.row_embeddings(row_ids)
        col_embeddings = self.col_embeddings(col_ids)
        
        # Concatenate row and column embeddings
        return torch.cat([row_embeddings, col_embeddings], dim=-1)
