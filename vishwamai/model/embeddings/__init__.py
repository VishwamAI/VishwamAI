"""Embedding implementations for Vishwamai."""

from typing import Optional, Type, Union, Literal

import torch
import torch.nn as nn

from .token_embedding import TokenEmbedding
from .positional import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding
)

def get_positional_encoding(
    name: str,
    embedding_dim: int,
    **kwargs
) -> nn.Module:
    """Get positional encoding by name.
    
    Args:
        name: Name of positional encoding ('sinusoidal', 'learned', or 'rotary')
        embedding_dim: Dimension of embeddings
        **kwargs: Additional arguments passed to the positional encoding
        
    Returns:
        Positional encoding module
    """
    encodings = {
        "sinusoidal": SinusoidalPositionalEncoding,
        "learned": LearnedPositionalEncoding,
        "rotary": RotaryPositionalEncoding
    }
    
    if name not in encodings:
        raise ValueError(
            f"Unknown positional encoding: {name}. "
            f"Available options are: {list(encodings.keys())}"
        )
        
    return encodings[name](embedding_dim, **kwargs)

class EmbeddingLayer(nn.Module):
    """Combined token and positional embedding layer."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_seq_length: int = 2048,
        embedding_scale: Optional[float] = None,
        dropout_prob: float = 0.1,
        positional_encoding: Literal["sinusoidal", "learned", "rotary", "none"] = "sinusoidal",
        tie_weights: bool = False,
        **pos_encoding_kwargs
    ):
        """Initialize embedding layer.
        
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: Index used for padding token
            max_seq_length: Maximum sequence length
            embedding_scale: Optional custom embedding scale factor
            dropout_prob: Dropout probability
            positional_encoding: Type of positional encoding to use
            tie_weights: Whether to enable weight tying
            **pos_encoding_kwargs: Additional arguments for positional encoding
        """
        super().__init__()
        
        # Token embedding
        self.token_embedding = TokenEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            scale_embeddings=embedding_scale is not None,
            embedding_scale=embedding_scale,
            dropout_prob=dropout_prob,
            tie_weights=tie_weights,
        )
        
        # Positional encoding
        self.positional_encoding = None
        if positional_encoding != "none":
            self.positional_encoding = get_positional_encoding(
                name=positional_encoding,
                embedding_dim=embedding_dim,
                max_seq_length=max_seq_length,
                dropout_prob=dropout_prob,
                **pos_encoding_kwargs
            )
            
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        offset: int = 0,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Convert input IDs to embeddings.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            positions: Optional position IDs
            offset: Starting position offset for positional encoding
            
        Returns:
            - If using regular positional encodings:
              Tensor of shape [batch_size, seq_len, embedding_dim]
            - If using rotary encoding:
              Tuple of (embeddings, cos, sin) for rotary computation
        """
        # Get token embeddings
        embeddings = self.token_embedding(input_ids, positions)
        
        # Apply positional encoding if available
        if isinstance(self.positional_encoding, (SinusoidalPositionalEncoding, LearnedPositionalEncoding)):
            embeddings = self.positional_encoding(embeddings, offset)
            return embeddings
        elif isinstance(self.positional_encoding, RotaryPositionalEncoding):
            # For rotary encoding, return embeddings and position encodings separately
            seq_len = input_ids.size(1)
            cos = self.positional_encoding.cos_pos[offset:offset+seq_len]
            sin = self.positional_encoding.sin_pos[offset:offset+seq_len]
            return embeddings, cos, sin
        else:
            return embeddings
            
    def tie_weights_with(self, output_layer: nn.Linear):
        """Tie embedding weights with output layer.
        
        Args:
            output_layer: Linear output layer to tie weights with
        """
        self.token_embedding.tie_weights_with(output_layer)
        
    def resize_embeddings(self, new_num_tokens: int):
        """Resize token embeddings.
        
        Args:
            new_num_tokens: New vocabulary size
        """
        self.token_embedding.resize_embeddings(new_num_tokens)

__all__ = [
    # Main classes
    "TokenEmbedding",
    "EmbeddingLayer",
    
    # Positional encodings
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "RotaryPositionalEncoding",
    
    # Factory functions
    "get_positional_encoding",
]
