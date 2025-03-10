# /home/kasinadhsarma/VishwamAI/vishwamai/models/transformer.py
"""
TPU-optimized transformer implementation using JAX/Haiku
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Dict, Tuple
import math

from .attention import FlashMLAttentionTPU, MultiModalAttentionTPU, TemporalAttentionTPU
from .kernel_layers import TPUGEMMLinear, TPULayerNorm, gelu_kernel
from .core import apply_rotary_embedding, create_causal_mask

class PositionalEncoding(hk.Module):
    """TPU-optimized positional encoding using rotary embeddings."""
    
    def __init__(self, embed_dim: int, max_seq_len: int = 2048,
                 scale_base: int = 10000, name: Optional[str] = None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.scale_base = scale_base

    def __call__(self, x: jnp.ndarray, positions: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if positions is None:
            positions = jnp.arange(x.shape[1])
            
        # Generate frequency bands
        freqs = self.scale_base ** (2 * (jnp.arange(self.embed_dim//2) // 2) / self.embed_dim)
        angles = positions[:, None] / freqs[None, :]
        
        # Generate rotary embeddings
        freqs_cis = jnp.exp(1j * angles).astype(jnp.complex64)
        
        # Use centralized rotary embedding function
        return apply_rotary_embedding(x, freqs_cis)

class TokenEmbedding(hk.Module):
    """TPU-optimized token embedding with weight tying support."""
    
    def __init__(self, vocab_size: int, embed_dim: int,
                 scale_grad_by_freq: bool = False,
                 tie_weights: bool = False, name: Optional[str] = None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.scale_grad_by_freq = scale_grad_by_freq
        self.tie_weights = tie_weights
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Initialize embedding matrix
        w_init = hk.initializers.TruncatedNormal(stddev=1.0 / math.sqrt(self.embed_dim))
        w_embed = hk.get_parameter("w_embed", 
                                 shape=[self.vocab_size, self.embed_dim],
                                 dtype=x.dtype,
                                 init=w_init)
        
        # Embedding lookup
        embedded = jnp.take(w_embed, x, axis=0)
        
        # Scale embeddings
        if self.scale_grad_by_freq:
            # Count token frequencies for scaling
            counts = jnp.bincount(x.reshape(-1), 
                                length=self.vocab_size,
                                minlength=self.vocab_size)
            scale = 1.0 / jnp.maximum(counts, 1.0)
            embedded = embedded * scale[x][..., None]
            
        return embedded * math.sqrt(self.embed_dim)

class FeedForward(hk.Module):
    """TPU-optimized feed-forward network with GEGLU activation."""
    
    def __init__(self, embed_dim: int, ff_dim: int,
                 dropout_rate: float = 0.1,
                 activation=gelu_kernel,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        
    def __call__(self, x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        # Project to intermediate dimension
        x = TPUGEMMLinear(self.ff_dim)(x)
        x = self.activation(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if is_training else x
        
        # Project back to embedding dimension
        x = TPUGEMMLinear(self.embed_dim)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if is_training else x
        
        return x

class TransformerComputeLayerTPU(hk.Module):
    """TPU-optimized transformer layer with memory-efficient attention"""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout_rate: float = 0.1, block_size: int = 128,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.block_size = block_size

        # Initialize attention and feed-forward layers
        self.attention = FlashMLAttentionTPU(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout_rate=dropout_rate
        )
        self.ff_network = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )
        
        # Layer normalization
        self.norm1 = TPULayerNorm(embed_dim)
        self.norm2 = TPULayerNorm(embed_dim)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                is_training: bool = True) -> jnp.ndarray:
        # Pre-norm transformer architecture
        normed_x = self.norm1(x)
        attention_output = self.attention(normed_x, mask=mask, is_training=is_training)
        x = x + attention_output

        normed_x = self.norm2(x)
        ff_output = self.ff_network(normed_x)
        x = x + ff_output

        return x

class TransformerMemoryLayerTPU(hk.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 num_memory_slots: int = 32,
                 dropout_rate: float = 0.1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_memory_slots = num_memory_slots
        self.dropout_rate = dropout_rate

    def __call__(self, x, is_training=True):
        # Memory slots
        memory = hk.get_parameter(
            "memory",
            shape=[self.num_memory_slots, self.embed_dim],
            init=hk.initializers.TruncatedNormal()
        )
        
        # Memory attention
        q = TPUGEMMLinear(self.embed_dim)(x)
        k = TPUGEMMLinear(self.embed_dim)(memory)
        v = TPUGEMMLinear(self.embed_dim)(memory)
        
        # Split heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # Attention
        scale = 1.0 / jnp.sqrt(self.embed_dim // self.num_heads)
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
        weights = jax.nn.softmax(scores)
        weights = hk.dropout(hk.next_rng_key(), self.dropout_rate, weights)
        
        output = jnp.matmul(weights, v)
        output = self._combine_heads(output)
        output = TPUGEMMLinear(self.embed_dim)(output)
        
        return x + hk.dropout(hk.next_rng_key(), self.dropout_rate, output)
        
    def _split_heads(self, x):
        batch, seq_len, features = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, -1)
        return jnp.transpose(x, (0, 2, 1, 3))
        
    def _combine_heads(self, x):
        batch, heads, seq_len, features = x.shape
        x = jnp.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, heads * features)

class HybridThoughtAwareAttentionTPU(hk.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 num_thought_slots: int = 8,
                 dropout_rate: float = 0.1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_thought_slots = num_thought_slots
        self.dropout_rate = dropout_rate
        
    def __call__(self, x, mask=None, is_training=True):
        thought_slots = hk.get_parameter(
            "thought_slots",
            shape=[self.num_thought_slots, self.embed_dim],
            init=hk.initializers.TruncatedNormal()
        )
        
        # Main attention
        q = TPUGEMMLinear(self.embed_dim)(x)
        k = TPUGEMMLinear(self.embed_dim)(x)
        v = TPUGEMMLinear(self.embed_dim)(x)
        
        # Thought attention
        t_q = TPUGEMMLinear(self.embed_dim)(x)
        t_k = TPUGEMMLinear(self.embed_dim)(thought_slots)
        t_v = TPUGEMMLinear(self.embed_dim)(thought_slots)
        
        # Process main and thought attention in parallel
        main_output = self._process_attention(q, k, v, mask)
        thought_output = self._process_attention(t_q, t_k, t_v)
        
        # Combine outputs
        combined = main_output + thought_output
        return TPUGEMMLinear(self.embed_dim)(combined)
        
    def _process_attention(self, q, k, v, mask=None):
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        scale = 1.0 / jnp.sqrt(self.embed_dim // self.num_heads)
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
        
        if mask is not None:
            scores = jnp.where(mask[..., None, None, :], scores, -1e9)
            
        weights = jax.nn.softmax(scores)
        weights = hk.dropout(hk.next_rng_key(), self.dropout_rate, weights)
        
        output = jnp.matmul(weights, v)
        return self._combine_heads(output)
        
    def _split_heads(self, x):
        batch, seq_len, features = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, -1)
        return jnp.transpose(x, (0, 2, 1, 3))
        
    def _combine_heads(self, x):
        batch, heads, seq_len, features = x.shape
        x = jnp.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, heads * features)