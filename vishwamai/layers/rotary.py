"""Rotary position embedding implementation optimized for TPU."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Any, Union

class TPURotaryEmbedding(nn.Module):
    """TPU-optimized rotary position embedding."""
    dim: int
    max_seq_len: int = 2048
    base: int = 10000
    dtype: Any = jnp.float32
    
    def setup(self):
        # Precompute inv_freq with proper shape for TPU efficiency
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=self.dtype) / self.dim))
        self.inv_freq = self.variable('rope', 'inv_freq', lambda: inv_freq)
        
        # Precompute position indices for common use cases
        positions = jnp.arange(self.max_seq_len, dtype=self.dtype)
        self.position_ids = self.variable('rope', 'position_ids', lambda: positions)
    
    def __call__(self, x: jnp.ndarray, seq_len: Optional[int] = None, offset: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute cos and sin for rotary embeddings."""
        if seq_len is None:
            seq_len = x.shape[1]  # Assume shape (batch, seq_len, ...)
            
        # Get position IDs with offset
        position_ids = self.position_ids.value[offset:offset+seq_len]
        
        # Compute freqs
        freqs = jnp.outer(position_ids, self.inv_freq.value)
        
        # Compute sin and cos
        # Efficient on TPU: first compute all, then reshape
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos_emb = jnp.cos(emb)
        sin_emb = jnp.sin(emb)
        
        # Reshape for broadcasting
        cos_emb = cos_emb.reshape(seq_len, 1, self.dim)
        sin_emb = sin_emb.reshape(seq_len, 1, self.dim)
        
        return cos_emb, sin_emb

def apply_rotary_pos_emb(q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        cos: Cosine part of rotary embeddings
        sin: Sine part of rotary embeddings
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Reshape q and k for rotation
    q_embed = q.reshape(*q.shape[:-1], -1, 2)
    k_embed = k.reshape(*k.shape[:-1], -1, 2)
    
    # Prepare for complex multiplication
    q_1, q_2 = jnp.split(q_embed, 2, axis=-1)
    k_1, k_2 = jnp.split(k_embed, 2, axis=-1)
    
    # Complex multiplication:
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    q_rotated_1 = q_1 * cos - q_2 * sin
    q_rotated_2 = q_1 * sin + q_2 * cos
    k_rotated_1 = k_1 * cos - k_2 * sin
    k_rotated_2 = k_1 * sin + k_2 * cos
    
    # Concatenate and reshape back
    q_rotated = jnp.concatenate([q_rotated_1, q_rotated_2], axis=-1)
    k_rotated = jnp.concatenate([k_rotated_1, k_rotated_2], axis=-1)
    
    q_rotated = q_rotated.reshape(q.shape)
    k_rotated = k_rotated.reshape(k.shape)
    
    return q_rotated, k_rotated

class TPUFrequencyRotaryEmbedding(nn.Module):
    """Improved rotary embedding with frequency-based decomposition for TPUs."""
    dim: int
    max_position_embeddings: int = 2048
    base: int = 10000
    scaling_factor: float = 1.0
    
    @nn.compact
    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, positions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Compute inverse frequency with dynamic scaling
        dim_t = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (self.base ** (dim_t / self.dim))
        
        # Apply scaling factor for extended context
        if self.scaling_factor != 1.0:
            inv_freq = inv_freq * self.scaling_factor
            
        # Compute frequencies for given positions
        freqs = jnp.outer(positions, inv_freq)
        
        # Cache the freqs for TPU efficiency
        freqs = self.variable('cache', 'freqs', lambda: freqs).value
        
        # Compute sin and cos
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        # Reshape for broadcasting
        cos = cos[:, None, None, :]  # [seq_len, 1, 1, dim/2]
        sin = sin[:, None, None, :]  # [seq_len, 1, 1, dim/2]
        
        # Efficient rotary application
        q_embed = q.reshape(*q.shape[:-1], -1, 2)
        k_embed = k.reshape(*k.shape[:-1], -1, 2)
        
        q_1, q_2 = q_embed[..., 0], q_embed[..., 1]
        k_1, k_2 = k_embed[..., 0], k_embed[..., 1]
        
        q_out_1 = q_1 * cos - q_2 * sin
        q_out_2 = q_1 * sin + q_2 * cos
        k_out_1 = k_1 * cos - k_2 * sin
        k_out_2 = k_1 * sin + k_2 * cos
        
        # Interleave the results
        q_out = jnp.stack([q_out_1, q_out_2], axis=-1).reshape(q.shape)
        k_out = jnp.stack([k_out_1, k_out_2], axis=-1).reshape(k.shape)
        
        return q_out, k_out
