"""Positional encoding modules for VishwamAI using Flax."""

import math
from typing import Optional, Tuple, Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

def create_sinusoidal_positions(seq_length: int, dim: int) -> jnp.ndarray:
    """Create sinusoidal position embeddings.
    
    Args:
        seq_length: Maximum sequence length
        dim: Embedding dimension
        
    Returns:
        Position embeddings [seq_length, dim]
    """
    # Create position indices
    position = jnp.arange(seq_length)[:, None]
    div_term = jnp.exp(jnp.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    
    # Calculate sine and cosine components
    pe = jnp.zeros((seq_length, dim))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    
    return pe

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    max_seq_length: int
    hidden_size: int
    dropout_rate: float = 0.1
    scale: float = 1.0
    deterministic: bool = False
    
    def setup(self):
        """Initialize positional encoding."""
        # Create fixed positional encodings
        self.pe = create_sinusoidal_positions(
            self.max_seq_length,
            self.hidden_size
        )
        
    def __call__(self, x: jnp.ndarray, 
                 deterministic: Optional[bool] = None) -> jnp.ndarray:
        """Add positional encodings to input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_length, hidden_size]
            deterministic: Whether to run in deterministic mode
            
        Returns:
            Embeddings with positional encoding added
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        seq_length = x.shape[1]
        
        # Get positional encodings for current sequence length
        pe = self.pe[:seq_length]
        
        # Add positional encodings to input
        x = x + (pe * self.scale)
        
        # Apply dropout during training
        if not deterministic:
            key = self.make_rng('dropout')
            x = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(x, deterministic=deterministic)
            
        return x

def fixed_pos_embedding(x: jnp.ndarray, seq_dim: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create sinusoidal position embeddings for RoPE.
    
    Args:
        x: Input tensor
        seq_dim: Dimension containing sequence length
        
    Returns:
        Tuple of (cos, sin) tensors for RoPE
    """
    dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    
    seq_len = x.shape[seq_dim]
    position_ids = jnp.arange(seq_len)
    
    sinusoid_inp = jnp.einsum('i,j->ij', position_ids, inv_freq)
    
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)

def rotate_every_two(x: jnp.ndarray) -> jnp.ndarray:
    """Rotate every two elements in the final dimension."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return jnp.stack((-x2, x1), axis=-1).reshape(x.shape)

def apply_rotary_pos_emb(x: jnp.ndarray, sin: jnp.ndarray, 
                        cos: jnp.ndarray, offset: int = 0) -> jnp.ndarray:
    """Apply rotary position embeddings to input.
    
    Args:
        x: Input tensor
        sin: Sine component of position embeddings
        cos: Cosine component of position embeddings
        offset: Position offset
        
    Returns:
        Tensor with RoPE applied
    """
    sin = jnp.repeat(sin[None, offset:x.shape[1] + offset, None, :], 
                     x.shape[0], axis=0)
    cos = jnp.repeat(cos[None, offset:x.shape[1] + offset, None, :],
                     x.shape[0], axis=0)
    
    return (x * cos) + (rotate_every_two(x) * sin)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)."""
    
    max_seq_length: int
    dim: int
    base: int = 10000
    scale: bool = False
    scale_base: float = 512.0
    
    def setup(self):
        """Initialize RoPE parameters."""
        if self.scale:
            self.scale_factor = (self.max_seq_length / self.scale_base)
            inv_freq = 1.0 / (self.base ** 
                (jnp.arange(0, self.dim, 2) / self.dim / self.scale_factor))
        else:
            inv_freq = 1.0 / (self.base ** 
                (jnp.arange(0, self.dim, 2) / self.dim))
            
        # Calculate fixed position embeddings
        position_ids = jnp.arange(self.max_seq_length)
        sinusoidal_inp = jnp.einsum('i,j->ij', position_ids, inv_freq)
        
        self.sin = jnp.sin(sinusoidal_inp)
        self.cos = jnp.cos(sinusoidal_inp)
        
    def __call__(self, x: jnp.ndarray, offset: int = 0) -> jnp.ndarray:
        """Apply RoPE to input.
        
        Args:
            x: Input tensor [batch_size, seq_length, num_heads, head_dim]
            offset: Position offset for cache usage
            
        Returns:
            Tensor with RoPE applied
        """
        return apply_rotary_pos_emb(x, self.sin, self.cos, offset)
