"""Multi-Layer Attention block combining attention, caching, and residuals."""

from typing import Optional, Tuple, Dict, Any, List
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .attention import MultiLayerAttention
from .layer_manager import MLALayerManager
from .residual import MLAResidual

class MLABlock(nn.Module):
    """Multi-Layer Attention block with state management."""
    
    hidden_size: int
    num_heads: int
    head_dim: Optional[int] = None
    num_prev_layers: int = 4
    attention_window: int = 4
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    use_rope: bool = True
    max_sequence_length: int = 2048
    layer_id: int = 0
    use_gate: bool = True
    gate_init_eps: float = 0.1
    layer_scale: bool = True
    layer_scale_init_value: float = 0.1
    use_compressed_cache: bool = False
    compression_dim: Optional[int] = None
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    deterministic: bool = False
    
    def setup(self):
        """Initialize MLA block components."""
        # Multi-layer attention
        self.attention = MultiLayerAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_prev_layers=self.num_prev_layers,
            attention_window=self.attention_window,
            dropout_rate=self.dropout_rate,
            attention_dropout=self.attention_dropout,
            use_rope=self.use_rope,
            max_sequence_length=self.max_sequence_length,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            deterministic=self.deterministic
        )
        
        # Layer manager for caching
        self.layer_manager = MLALayerManager(
            hidden_size=self.hidden_size,
            num_layers=self.num_prev_layers,
            max_cache_size=self.attention_window,
            dtype=self.dtype,
            deterministic=self.deterministic,
            normalize_cached=True,
            use_compressed_cache=self.use_compressed_cache,
            compression_dim=self.compression_dim,
            eviction_policy="lru"
        )
        
        # Residual connection
        self.residual = MLAResidual(
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            use_gate=self.use_gate,
            gate_init_eps=self.gate_init_eps,
            layer_scale=self.layer_scale,
            layer_scale_init_value=self.layer_scale_init_value,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            deterministic=self.deterministic
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        
    def _get_relevant_states(self,
                           cache: Dict[str, Any]) -> List[jnp.ndarray]:
        """Get relevant previous layer states from cache.
        
        Args:
            cache: Layer state cache
            
        Returns:
            List of relevant previous states
        """
        return self.layer_manager.get_layer_states(
            cache=cache,
            current_layer=self.layer_id,
            num_layers=self.attention_window
        )
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 cache: Dict[str, Any],
                 attention_mask: Optional[jnp.ndarray] = None,
                 deterministic: Optional[bool] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Apply MLA block to input.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            cache: Layer state cache
            attention_mask: Optional attention mask
            deterministic: Whether to run in deterministic mode
            
        Returns:
            Tuple of:
                - Output tensor
                - Dict of auxiliary outputs
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        
        # Get layer normalized input
        normed_states = self.layer_norm(hidden_states)
        
        # Get relevant previous layer states
        prev_states = self._get_relevant_states(cache)
        
        # Apply multi-layer attention
        attention_output, attention_aux = self.attention(
            hidden_states=normed_states,
            prev_hidden_states=prev_states,
            attention_mask=attention_mask,
            deterministic=deterministic
        )
        
        # Apply residual connection
        output, residual_aux = self.residual(
            x=hidden_states,
            transformed=attention_output,
            cross_layer=None if not prev_states else prev_states[-1],
            deterministic=deterministic
        )
        
        # Update cache with current layer state
        cache = self.layer_manager.update_cache(
            cache=cache,
            layer_state=output,
            layer_id=self.layer_id
        )
        
        # Collect auxiliary outputs
        aux_outputs = {
            'cache': cache,
            'attention': attention_aux,
            'residual': residual_aux,
            'cache_stats': self.layer_manager.get_cache_stats(cache)
        }
        
        return output, aux_outputs
        
    def init_cache(self) -> Dict[str, Any]:
        """Initialize empty cache.
        
        Returns:
            Empty cache dictionary
        """
        return self.layer_manager.init_cache()
        
    def clear_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Clear the layer cache.
        
        Args:
            cache: Cache to clear
            
        Returns:
            Empty cache dictionary
        """
        return self.layer_manager.clear_cache(cache)
