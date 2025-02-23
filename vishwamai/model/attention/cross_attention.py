"""Cross-attention module for VishwamAI using JAX."""

from typing import Optional, Tuple, Any, Callable
import math

import jax
import jax.numpy as jnp
from jax.nn import softmax
import flax.linen as nn

from ..embeddings.positional import RotaryPositionalEmbedding

class CrossAttention(nn.Module):
    """Cross-attention for inter-layer communication in MLA blocks."""
    
    hidden_size: int
    num_heads: int
    head_dim: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    use_rope: bool = True
    max_sequence_length: int = 2048
    num_prev_layers: int = 4  # Number of previous layers to attend to
    use_gate: bool = True  # Whether to use gating mechanism
    
    def setup(self):
        """Initialize cross-attention components."""
        self.actual_head_dim = self.head_dim or self.hidden_size // self.num_heads
        
        # Query projection (for current layer)
        self.q_proj = nn.Dense(
            self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="q_proj"
        )
        
        # Key/Value projections (for previous layers)
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
        
        # Output projection
        self.o_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="o_proj"
        )
        
        # Layer gating mechanism
        if self.use_gate:
            self.layer_gate = nn.Dense(
                self.num_prev_layers,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.normal(stddev=0.02),
                use_bias=False,
                name="layer_gate"
            )
            
        # Rotary embeddings if enabled
        if self.use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(
                max_seq_length=self.max_sequence_length,
                dim=self.actual_head_dim
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
        
    def _compute_layer_weights(self, query_layer: jnp.ndarray) -> jnp.ndarray:
        """Compute attention weights for different layers.
        
        Args:
            query_layer: Current layer representation [batch, seq_len, hidden_dim]
            
        Returns:
            Layer weights [batch, num_prev_layers]
        """
        # Global average pooling over sequence length
        pooled = jnp.mean(query_layer, axis=1)  # [batch, hidden_dim]
        
        # Project to get layer weights
        layer_weights = self.layer_gate(pooled)  # [batch, num_prev_layers]
        
        # Apply softmax to get attention weights
        layer_weights = softmax(layer_weights, axis=-1)
        
        return layer_weights
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 prev_hidden_states: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        """Apply cross-attention between current and previous layers.
        
        Args:
            hidden_states: Current layer states [batch, seq_len, hidden_dim]
            prev_hidden_states: Previous layer states [num_prev, batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch, 1, seq_len, seq_len]
            deterministic: Whether to apply dropout
            
        Returns:
            Updated hidden states
        """
        num_prev, batch_size, seq_length, _ = prev_hidden_states.shape
        
        # Project current layer to queries
        queries = self.q_proj(hidden_states)
        queries = self._split_heads(queries)  # [batch, heads, seq, head_dim]
        
        if self.use_rope:
            queries = self.rotary_emb(queries)
            
        # Process each previous layer
        layer_outputs = []
        for i in range(num_prev):
            # Project previous layer to keys and values
            keys = self.k_proj(prev_hidden_states[i])
            values = self.v_proj(prev_hidden_states[i])
            
            keys = self._split_heads(keys)
            values = self._split_heads(values)
            
            if self.use_rope:
                keys = self.rotary_emb(keys)
            
            # Compute attention scores
            attn_weights = jnp.einsum('bhqd,bhkd->bhqk', queries, keys)
            attn_weights = attn_weights / math.sqrt(self.actual_head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = jnp.where(attention_mask, attn_weights, -1e9)
                
            # Apply softmax and dropout
            attn_weights = softmax(attn_weights, axis=-1)
            if not deterministic:
                attn_weights = nn.Dropout(
                    rate=self.attention_dropout,
                    deterministic=deterministic
                )(attn_weights, deterministic=deterministic)
                
            # Compute layer output
            layer_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, values)
            layer_output = self._merge_heads(layer_output)
            layer_outputs.append(layer_output)
            
        # Stack layer outputs
        layer_outputs = jnp.stack(layer_outputs)  # [num_prev, batch, seq, hidden]
        
        if self.use_gate:
            # Compute dynamic weights for each layer
            layer_weights = self._compute_layer_weights(hidden_states)
            layer_weights = jnp.expand_dims(layer_weights, axis=(2, 3))
            
            # Weighted combination of layer outputs
            combined_output = jnp.sum(
                layer_outputs * layer_weights, axis=0
            )
        else:
            # Simple average if not using gating
            combined_output = jnp.mean(layer_outputs, axis=0)
            
        # Project output
        output = self.o_proj(combined_output)
        
        # Apply dropout during training
        if not deterministic:
            output = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(output, deterministic=deterministic)
            
        return output
