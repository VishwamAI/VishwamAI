"""Flash Attention implementation for VishwamAI using JAX."""

from typing import Optional, Tuple, Any, Callable
import math

import jax
import jax.numpy as jnp
from jax.nn import softmax
import flax.linen as nn

def _block_matmul(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                block_size: int) -> jnp.ndarray:
    """Compute attention with blocked matrix multiplication.
    
    Args:
        q: Query tensor [batch, heads, q_length, d_k]
        k: Key tensor [batch, heads, kv_length, d_k]
        v: Value tensor [batch, heads, kv_length, d_v]
        block_size: Size of blocks for tiling
        
    Returns:
        Output tensor [batch, heads, q_length, d_v]
    """
    batch_size, num_heads, q_length, d_k = q.shape
    _, _, kv_length, _ = k.shape
    
    # Reshape inputs into blocks
    q_blocks = q.reshape(batch_size, num_heads,
                        math.ceil(q_length / block_size), block_size,
                        d_k)
    k_blocks = k.reshape(batch_size, num_heads,
                        math.ceil(kv_length / block_size), block_size,
                        d_k)
    v_blocks = v.reshape(batch_size, num_heads,
                        math.ceil(kv_length / block_size), block_size,
                        -1)
    
    # Compute block-wise attention scores
    scores = jnp.einsum('bhnid,bhmjd->bhnmij', q_blocks, k_blocks)
    scores = scores / math.sqrt(d_k)
    
    # Apply softmax over the key dimension
    attn_weights = softmax(scores, axis=(3, 5))
    
    # Compute attention outputs
    outputs = jnp.einsum('bhnmij,bhmjd->bhnid', attn_weights, v_blocks)
    
    # Reshape back to original dimensions
    outputs = outputs.reshape(batch_size, num_heads, q_length, -1)
    
    return outputs

class FlashAttention(nn.Module):
    """Flash Attention implementation optimized for TPU."""
    
    hidden_size: int
    num_heads: int
    head_dim: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    block_size: int = 64  # Size of attention blocks
    causal: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
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
                 deterministic: bool = True) -> jnp.ndarray:
        """Apply flash attention to input.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch, 1, seq_len, seq_len]
            deterministic: Whether to apply dropout
            
        Returns:
            Attention output
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
        
        # Compute attention with block-sparse pattern
        attention_output = _block_matmul(
            query, key, value,
            block_size=self.block_size
        )
        
        # Merge heads and project output
        attention_output = self._merge_heads(attention_output)
        attention_output = self.o_proj(attention_output)
        
        # Apply dropout during training
        if not deterministic:
            attention_output = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(attention_output, deterministic=deterministic)
            
        return attention_output
        
    def _mask_by_block(self, mask: jnp.ndarray,
                      block_size: int) -> jnp.ndarray:
        """Convert attention mask to block format.
        
        Args:
            mask: Attention mask [batch, 1, q_length, kv_length]
            block_size: Size of attention blocks
            
        Returns:
            Block mask [batch, 1, num_q_blocks, num_kv_blocks, block_size, block_size]
        """
        batch_size, _, q_length, kv_length = mask.shape
        
        # Reshape mask into blocks
        mask = mask.reshape(
            batch_size, 1,
            math.ceil(q_length / block_size), block_size,
            math.ceil(kv_length / block_size), block_size
        )
        
        # Transpose to match block attention pattern
        mask = jnp.transpose(mask, (0, 1, 2, 4, 3, 5))
        
        return mask
