"""Utility functions for vision models."""

from __future__ import annotations
from collections.abc import Sequence
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, TypeVar, Literal
from numpy.typing import NDArray, ArrayLike

# Define type variables for better type hinting
B = TypeVar('B')  # Batch dimension
M = TypeVar('M')  # Sequence length dimension 
D = TypeVar('D')  # Embedding dimension
N = TypeVar('N')  # Image height dimension
P = TypeVar('P')  # Image width dimension

def _posemb_sincos_2d(
    h: int,
    w: int,
    width: int,
    temperature: float = 10000.0,
    dtype: Any = jnp.float32,
    scale: float = 1.0  # Added scaling factor
) -> jnp.ndarray:
    """Generate 2D sinusoidal position embeddings with TPU optimizations.
    
    Args:
        h: Image height in patches
        w: Image width in patches
        width: Hidden dimension size
        temperature: Temperature for frequency scaling
        dtype: Data type for embeddings
        scale: Optional scaling factor for embeddings
        
    Returns:
        Position embeddings [h*w, width]
    """
    # Generate grid indices
    y, x = jnp.meshgrid(
        jnp.arange(h, dtype=jnp.float32),
        jnp.arange(w, dtype=jnp.float32),
        indexing='ij'
    )
    
    # Flatten grid coordinates
    y = y.reshape(-1)
    x = x.reshape(-1)
    
    # Efficient computation of frequency bands
    omega = jnp.exp(-jnp.log(temperature) * jnp.arange(0, width // 4, dtype=jnp.float32) / (width // 4))
    
    # Compute embeddings efficiently using TPU-optimized operations
    y_emb = y[:, None] * omega[None, :]
    x_emb = x[:, None] * omega[None, :]
    
    # Concatenate sin/cos embeddings
    pos_emb = jnp.concatenate([
        jnp.sin(y_emb),
        jnp.cos(y_emb),
        jnp.sin(x_emb),
        jnp.cos(x_emb)
    ], axis=1)
    
    # Scale if needed
    if scale != 1.0:
        pos_emb = pos_emb * scale
        
    return pos_emb.astype(dtype)

class MAPHead(nn.Module):
    """Multi-head Attention Pooling for global feature extraction."""
    
    block_id: int
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Apply multi-head attention pooling.
        
        Args:
            x: Input features [batch, seq, hidden]
            deterministic: Whether in training mode
            
        Returns:
            Pooled features [batch, hidden]
        """
        hidden_dim = x.shape[-1]
        mlp_dim = self.mlp_dim or hidden_dim * 4
        
        # Query token
        query = self.param(
            f'query_{self.block_id}',
            nn.initializers.normal(0.02),
            (1, 1, hidden_dim)
        )
        query = jnp.tile(query, [x.shape[0], 1, 1])
        
        # Multi-head attention
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            deterministic=deterministic,
            name=f'attn_{self.block_id}'
        )(query, x)
        
        # MLP
        y = nn.LayerNorm(dtype=self.dtype)(attn)
        y = nn.Dense(
            mlp_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.02),
            name=f'mlp1_{self.block_id}'
        )(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(
            hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.02),
            name=f'mlp2_{self.block_id}'
        )(y)
        
        # Return pooled features
        return y[:, 0]  # Return CLS token only

def efficient_patch_extraction(
    images: jnp.ndarray,
    patch_size: int,
    hidden_dim: int,
    dtype: Any = jnp.float32
) -> Tuple[jnp.ndarray, int, int]:
    """Extract patches from images efficiently using TPU-optimized convolution.
    
    Args:
        images: Input images [batch, height, width, channels]
        patch_size: Size of patches
        hidden_dim: Hidden dimension size
        dtype: Data type for patches
        
    Returns:
        Tuple of:
        - Extracted patches [batch, num_patches, hidden_dim]
        - Number of patches in height
        - Number of patches in width
    """
    # Efficient patch extraction using conv
    patch_embedder = nn.Conv(
        features=hidden_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding='VALID',
        dtype=dtype
    )
    
    x = patch_embedder(images)
    
    # Get shape info
    batch_size, h, w, c = x.shape
    num_patches = h * w
    
    # Reshape to sequence
    x = x.reshape(batch_size, num_patches, c)
    
    return x, h, w