# /home/kasinadhsarma/VishwamAI/vishwamai/models/transformer.py
"""
TPU-optimized transformer implementation using JAX/Haiku
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Dict

from .attention import FlashMLAttentionTPU, MultiModalAttentionTPU, TemporalAttentionTPU
from .kernel_layers import TPUGEMMLinear

class TransformerComputeLayerTPU(hk.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 ff_dim: int = 2048, dropout_rate: float = 0.1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
    def __call__(self, x, mask=None, is_training=True):
        # Self attention
        normed = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        attention = FlashMLAttentionTPU(self.embed_dim, self.num_heads)(
            normed, mask=mask, is_training=is_training
        )
        x = x + hk.dropout(hk.next_rng_key(), self.dropout_rate, attention)
        
        # Feed-forward
        normed = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        ff = TPUGEMMLinear(self.ff_dim)(normed)
        ff = jax.nn.relu(ff)
        ff = TPUGEMMLinear(self.embed_dim)(ff)
        return x + hk.dropout(hk.next_rng_key(), self.dropout_rate, ff)

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