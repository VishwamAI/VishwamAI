"""Self-attention module for VishwamAI using JAX/Flax."""

from typing import Optional, Tuple, Any, Callable
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.nn import softmax
import flax.linen as nn

from ..embeddings.positional import RotaryPositionalEmbedding

def scaled_dot_product(query: jnp.ndarray,
                      key: jnp.ndarray,
                      value: jnp.ndarray,
                      mask: Optional[jnp.ndarray] = None,
                      attention_dropout: float = 0.0,
                      deterministic: bool = True,
                      dtype: jnp.dtype = jnp.float32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute scaled dot-product attention.
    
    Args:
        query: Query tensor [batch, heads, q_length, d_k]
        key: Key tensor [batch, heads, kv_length, d_k]
        value: Value tensor [batch, heads, kv_length, d_v]
        mask: Optional attention mask [batch, 1, q_length, kv_length]
        attention_dropout: Dropout rate for attention weights
        deterministic: Whether to apply dropout
        dtype: Data type for computation
        
    Returns:
        Tuple of (attention output, attention weights)
    """
    d_k = query.shape[-1]
    
    # Compute attention scores
    attention_scores = jnp.einsum('bhqd,bhkd->bhqk', query, key)
    attention_scores = attention_scores / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        attention_scores = jnp.where(mask, attention_scores, -1e9)
    
    # Apply softmax and dropout
    attention_weights = softmax(attention_scores, axis=-1)
    
    if not deterministic and attention_dropout > 0:
        rng = jax.random.PRNGKey(0)  # Should be passed from parent
        attention_weights = nn.Dropout(
            rate=attention_dropout,
            deterministic=deterministic
        )(attention_weights, deterministic=deterministic)
    
    # Compute attention output
    attention_output = jnp.einsum('bhqk,bhkv->bhqv', attention_weights, value)
    
    return attention_output.astype(dtype), attention_weights.astype(dtype)

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional rotary embeddings."""
    
    hidden_size: int
    num_heads: int
    head_dim: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    use_rope: bool = True
    max_sequence_length: int = 2048
    rope_scaling: Optional[float] = None
    
    def setup(self):
        """Initialize attention components."""
        self.actual_head_dim = self.head_dim or self.hidden_size // self.num_heads
        
        # Projection matrices
        self.q_proj = nn.Dense(
            self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="q_proj"
        )
        self.k_proj = nn.Dense(
            self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="k_proj"
        )
        self.v_proj = nn.Dense(
            self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="v_proj"
        )
        self.o_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="o_proj"
        )
        
        # Rotary embeddings if enabled
        if self.use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(
                max_seq_length=self.max_sequence_length,
                dim=self.actual_head_dim,
                scale=bool(self.rope_scaling),
                scale_base=self.rope_scaling if self.rope_scaling else 512.0
            )
            
    def _split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        """Split hidden dim into multiple heads.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Reshaped tensor [batch, num_heads, seq_len, head_dim]
        """
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.actual_head_dim)
        return jnp.transpose(x, (0, 2, 1, 3))
        
    def _merge_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        """Merge multiple heads back into hidden dim.
        
        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim]
            
        Returns:
            Reshaped tensor [batch, seq_len, hidden_dim]
        """
        batch, _, seq_len, _ = x.shape
        x = jnp.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, self.hidden_size)
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 position_ids: Optional[jnp.ndarray] = None,
                 deterministic: bool = True,
                 output_attentions: bool = False) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Apply self-attention to input.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch, 1, seq_len, seq_len]
            position_ids: Optional position indices for RoPE
            deterministic: Whether to apply dropout
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (attention output, optional attention weights)
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project inputs to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Split heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # Apply rotary embeddings if enabled
        if self.use_rope:
            query = self.rotary_emb(query)
            key = self.rotary_emb(key)
            
        # Compute attention
        attention_output, attention_weights = scaled_dot_product(
            query=query,
            key=key,
            value=value,
            mask=attention_mask,
            attention_dropout=self.attention_dropout,
            deterministic=deterministic,
            dtype=self.dtype
        )
        
        # Merge heads and project output
        attention_output = self._merge_heads(attention_output)
        attention_output = self.o_proj(attention_output)
        
        # Apply dropout
        if not deterministic:
            attention_output = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(attention_output, deterministic=deterministic)
            
        if output_attentions:
            return attention_output, attention_weights
        return attention_output, None
        
    def _create_causal_mask(self, batch_size: int, seq_length: int) -> jnp.ndarray:
        """Create causal attention mask.
        
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            
        Returns:
            Causal mask [batch, 1, seq_len, seq_len]
        """
        # Create mask for lower triangular matrix
        mask = jnp.triu(jnp.ones((seq_length, seq_length)), k=1)
        mask = mask.astype(bool)
        
        # Expand dimensions for batch and heads
        mask = jnp.expand_dims(mask, axis=(0, 1))
        mask = jnp.repeat(mask, batch_size, axis=0)
        
        return mask
