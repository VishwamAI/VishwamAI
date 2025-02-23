"""Multi-Layer Attention module for cross-layer attention."""

from typing import Optional, Tuple, Dict, Any, List
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..embeddings.positional import RotaryPositionalEmbedding

class MultiLayerAttention(nn.Module):
    """Attention mechanism for cross-layer communication."""
    
    hidden_size: int
    num_heads: int
    head_dim: Optional[int] = None
    num_prev_layers: int = 4
    attention_window: int = 4
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    use_rope: bool = True
    max_sequence_length: int = 2048
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    deterministic: bool = False
    
    def setup(self):
        """Initialize multi-layer attention components."""
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
        
        # Layer selection gate
        self.layer_gate = nn.Dense(
            self.num_prev_layers,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
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
        layer_logits = self.layer_gate(pooled)  # [batch, num_prev_layers]
        
        # Apply softmax to get attention weights
        layer_weights = jax.nn.softmax(layer_logits, axis=-1)
        
        return layer_weights
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 prev_hidden_states: List[jnp.ndarray],
                 attention_mask: Optional[jnp.ndarray] = None,
                 deterministic: Optional[bool] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Apply multi-layer attention.
        
        Args:
            hidden_states: Current layer states [batch, seq_len, hidden_dim]
            prev_hidden_states: Previous layer states list [[batch, seq, hidden], ...]
            attention_mask: Optional attention mask
            deterministic: Whether to run in deterministic mode
            
        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_dim]
                - Dict of auxiliary outputs
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project current layer to queries
        queries = self.q_proj(hidden_states)
        queries = self._split_heads(queries)  # [batch, heads, seq, head_dim]
        
        if self.use_rope:
            queries = self.rotary_emb(queries)
            
        # Process previous layers
        layer_outputs = []
        all_attention_weights = []
        
        # Take most recent layers up to attention window
        prev_states = prev_hidden_states[-self.attention_window:]
        
        for layer_idx, prev_states in enumerate(prev_states):
            # Project previous layer to keys and values
            keys = self.k_proj(prev_states)
            values = self.v_proj(prev_states)
            
            keys = self._split_heads(keys)
            values = self._split_heads(values)
            
            if self.use_rope:
                keys = self.rotary_emb(keys)
                
            # Compute attention scores
            attention_scores = jnp.einsum('bhqd,bhkd->bhqk', queries, keys)
            attention_scores = attention_scores / math.sqrt(self.actual_head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attention_scores = jnp.where(attention_mask, attention_scores, -1e9)
                
            # Apply softmax and dropout
            attention_weights = jax.nn.softmax(attention_scores, axis=-1)
            if not deterministic:
                attention_weights = nn.Dropout(
                    rate=self.attention_dropout,
                    deterministic=deterministic
                )(attention_weights, deterministic=deterministic)
                
            # Compute layer output
            layer_output = jnp.einsum('bhqk,bhkd->bhqd', attention_weights, values)
            layer_output = self._merge_heads(layer_output)
            
            layer_outputs.append(layer_output)
            all_attention_weights.append(attention_weights)
            
        # Stack layer outputs
        layer_outputs = jnp.stack(layer_outputs)  # [num_prev, batch, seq, hidden]
        
        # Compute layer weights
        layer_weights = self._compute_layer_weights(hidden_states)  # [batch, num_prev]
        
        # Expand dimensions for broadcasting
        layer_weights = jnp.expand_dims(layer_weights, axis=(2, 3))  # [batch, num_prev, 1, 1]
        
        # Weighted combination of layer outputs
        output = jnp.sum(layer_outputs * layer_weights, axis=0)  # [batch, seq, hidden]
        
        # Project output
        output = self.o_proj(output)
        
        # Apply dropout during training
        if not deterministic:
            output = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(output, deterministic=deterministic)
            
        aux_outputs = {
            'layer_weights': layer_weights,
            'attention_weights': all_attention_weights
        }
        
        return output, aux_outputs
