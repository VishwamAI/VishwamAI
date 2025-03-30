"""VishwamAI model implementation."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Optional, Tuple

from .layers.layers import (
    TPUGEMMLinear,
    TPULayerNorm,
    TPUMultiHeadAttention,
    TPUMoELayer
)
from vishwamai.layers.attention import FlashAttention

class VishwamAI(nn.Module):
    """VishwamAI transformer model."""
    vocab_size: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float = 0.1
    
    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim
        )
        
        self.layers = [
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_layers)
        ]
        
        self.ln_f = TPULayerNorm()
        self.lm_head = TPUGEMMLinear(features=self.vocab_size)
    
    def __call__(self, input_ids: jnp.ndarray, training: bool = False):
        x = self.embed(input_ids)
        
        for layer in self.layers:
            x = layer(x, training=training)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

class TransformerBlock(nn.Module):
    """Transformer block with optimized attention."""
    hidden_dim: int
    num_heads: int 
    dropout_rate: float = 0.1

    def setup(self):
        self.attention = FlashAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )
        self.ln1 = TPULayerNorm()
        self.ln2 = TPULayerNorm()
        self.ff = FeedForward(self.hidden_dim)
        
    def __call__(self, x: jnp.ndarray, training: bool = False):
        h = self.ln1(x)
        h = self.attention(h, training=training)
        x = x + h
        
        h = self.ln2(x)
        h = self.ff(h)
        x = x + h
        
        return x

class FeedForward(nn.Module):
    """Feed-forward layer with optimized GEMM."""
    dim: int
    expansion_factor: int = 4
    
    def setup(self):
        hidden_dim = self.dim * self.expansion_factor
        self.fc1 = TPUGEMMLinear(features=hidden_dim)
        self.fc2 = TPUGEMMLinear(features=self.dim)
        self.gelu = lambda x: jax.nn.gelu(x)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(self.gelu(self.fc1(x)))
