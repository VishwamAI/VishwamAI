"""Token embedding module for VishwamAI using Flax."""

import math
from typing import Optional, Tuple, Any
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

from dataclasses import field

class TokenEmbedding(nn.Module):
    """Token embedding with optional factorization for memory efficiency."""
    
    vocab_size: int
    hidden_size: int
    padding_idx: Optional[int] = None
    factorized: bool = False
    factorized_dim: Optional[int] = None
    embedding_std: float = 0.02
    layer_norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    deterministic: bool = False
    
    def setup(self):
        """Initialize the embedding layers."""
        self.actual_factorized_dim = self.factorized_dim or int(math.sqrt(self.hidden_size))
        
        if self.factorized:
            # Factorized embedding initialization
            self.first_embed = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.actual_factorized_dim,
                embedding_init=partial(
                    jax.nn.initializers.normal,
                    stddev=self.embedding_std
                )
            )
            self.second_embed = nn.Dense(
                features=self.hidden_size,
                use_bias=False,
                kernel_init=partial(
                    jax.nn.initializers.normal,
                    stddev=self.embedding_std / math.sqrt(self.actual_factorized_dim)
                )
            )
        else:
            # Standard embedding initialization
            self.embedding = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.hidden_size,
                embedding_init=partial(
                    jax.nn.initializers.normal,
                    stddev=self.embedding_std
                )
            )
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(epsilon=self.layer_norm_eps)
        
    def __call__(self, input_ids: jnp.ndarray, deterministic: Optional[bool] = None) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            deterministic: Whether to run in deterministic mode (disables dropout)
            
        Returns:
            Token embeddings [batch_size, seq_len, hidden_size]
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        
        if self.factorized:
            # Two-step embedding through factorized matrices
            embeddings = self.first_embed(input_ids)  # [B, L, F]
            embeddings = self.second_embed(embeddings)  # [B, L, H]
        else:
            # Direct embedding
            embeddings = self.embedding(input_ids)  # [B, L, H]
            
        # Handle padding if specified
        if self.padding_idx is not None:
            embeddings = jnp.where(
                input_ids[..., None] == self.padding_idx,
                0.0,
                embeddings
            )
            
        # Apply layer norm
        embeddings = self.layer_norm(embeddings)
        
        # Apply dropout during training
        if not deterministic:
            key = self.make_rng('dropout')
            embeddings = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(embeddings, deterministic=deterministic)
            
        return embeddings
        
    def count_params(self) -> int:
        """Get number of parameters in embedding.
        
        Returns:
            Number of parameters
        """
        if self.factorized:
            return (self.vocab_size * self.actual_factorized_dim +
                   self.actual_factorized_dim * self.hidden_size)
        else:
            return self.vocab_size * self.hidden_size
