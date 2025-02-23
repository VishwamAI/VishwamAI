"""Attention mechanisms for the Vishwamai model."""

from typing import Optional, Type

import torch
import torch.nn as nn

from .self_attention import SelfAttention
from .flash_attention import FlashAttention
from .cross_attention import CrossAttention

def create_attention_mask(
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype,
    device: torch.device,
    causal: bool = True,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create attention mask for a given sequence length.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        dtype: Data type for mask
        device: Device to create mask on
        causal: Whether to create causal mask
        padding_mask: Optional padding mask of shape [batch_size, seq_length]
        
    Returns:
        Attention mask of shape [batch_size, 1, seq_length, seq_length]
    """
    # Create causal mask if needed
    if causal:
        mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=device),
            diagonal=1
        )
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        mask = mask.expand(batch_size, 1, seq_length, seq_length)
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(mask, float("-inf"))
    else:
        mask = torch.zeros(
            (batch_size, 1, seq_length, seq_length),
            dtype=dtype,
            device=device
        )
        
    # Apply padding mask if provided
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        mask = mask.masked_fill(~padding_mask, float("-inf"))
        
    return mask

def create_cross_attention_mask(
    batch_size: int,
    tgt_length: int,
    src_length: int,
    dtype: torch.dtype,
    device: torch.device,
    tgt_padding_mask: Optional[torch.Tensor] = None,
    src_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create mask for cross-attention.
    
    Args:
        batch_size: Batch size
        tgt_length: Target sequence length
        src_length: Source sequence length
        dtype: Data type for mask
        device: Device to create mask on
        tgt_padding_mask: Optional target padding mask of shape [batch_size, tgt_length]
        src_padding_mask: Optional source padding mask of shape [batch_size, src_length]
        
    Returns:
        Cross-attention mask of shape [batch_size, 1, tgt_length, src_length]
    """
    # Initialize empty mask
    mask = torch.zeros(
        (batch_size, 1, tgt_length, src_length),
        dtype=dtype,
        device=device
    )
    
    # Apply target padding mask if provided
    if tgt_padding_mask is not None:
        mask = mask.masked_fill(
            ~tgt_padding_mask.unsqueeze(1).unsqueeze(-1),
            float("-inf")
        )
        
    # Apply source padding mask if provided
    if src_padding_mask is not None:
        mask = mask.masked_fill(
            ~src_padding_mask.unsqueeze(1).unsqueeze(-2),
            float("-inf")
        )
        
    return mask

def get_attention_mechanism(name: str) -> Type[nn.Module]:
    """Get attention mechanism class by name.
    
    Args:
        name: Name of attention mechanism ('self', 'flash', or 'cross')
        
    Returns:
        Attention mechanism class
    """
    attention_mechanisms = {
        "self": SelfAttention,
        "flash": FlashAttention,
        "cross": CrossAttention
    }
    
    if name not in attention_mechanisms:
        raise ValueError(
            f"Unknown attention mechanism: {name}. "
            f"Available options are: {list(attention_mechanisms.keys())}"
        )
        
    return attention_mechanisms[name]

def create_rotary_embeddings(
    seq_length: int,
    dim: int,
    base: int = 10000,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create rotary position embeddings.
    
    Args:
        seq_length: Maximum sequence length
        dim: Dimension of embeddings (must be even)
        base: Base for frequency computation
        device: Device to create embeddings on
        
    Returns:
        Tuple of (cos, sin) tensors for rotary embeddings
    """
    if dim % 2 != 0:
        raise ValueError(f"Dimension must be even, got {dim}")
        
    # Create position indices
    position = torch.arange(seq_length, device=device)
    
    # Create frequency indices
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Compute angles
    theta = position.unsqueeze(1) * freqs.unsqueeze(0)  # [L, D/2]
    
    # Compute sin and cos
    cos = torch.cos(theta).repeat_interleave(2, dim=-1)  # [L, D]
    sin = torch.sin(theta).repeat_interleave(2, dim=-1)  # [L, D]
    
    return cos.unsqueeze(0), sin.unsqueeze(0)  # [1, L, D]

__all__ = [
    # Attention mechanisms
    "SelfAttention",
    "FlashAttention",
    "CrossAttention",
    
    # Utility functions
    "create_attention_mask",
    "create_cross_attention_mask",
    "get_attention_mechanism",
    "create_rotary_embeddings",
]
