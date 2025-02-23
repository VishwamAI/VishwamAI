"""Base transformer block implementation."""

from typing import Optional, Tuple, Dict, Any
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..attention import MultiHeadSelfAttention, FlashAttention

def create_activation(name: str) -> Any:
    """Create activation function by name."""
    if name == "gelu":
        return nn.gelu
    elif name == "swiglu":
        return lambda x: jax.nn.silu(x[..., ::2]) * x[..., 1::2]
    elif name == "silu":
        return nn.swish
    else:
        return nn.relu

class TransformerBlock(nn.Module):
    """Base transformer block with self-attention and feed-forward layers."""
    
    hidden_size: int
    num_attention_heads: int
    intermediate_size: Optional[int] = None
    head_dim: Optional[int] = None
    activation: str = "gelu"
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    drop_path: float = 0.0
    use_flash_attention: bool = False
    use_rope: bool = True
    max_sequence_length: int = 2048
    layer_norm_eps: float = 1e-5
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    deterministic: bool = False
    
    def setup(self):
        """Initialize transformer block components."""
        # Compute actual dimensions
        self.actual_head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        self.actual_intermediate = self.intermediate_size or self.hidden_size * 4
        
        # Attention layer
        attention_cls = FlashAttention if self.use_flash_attention else MultiHeadSelfAttention
        self.attention = attention_cls(
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            head_dim=self.actual_head_dim,
            dropout_rate=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            use_rope=self.use_rope,
            max_sequence_length=self.max_sequence_length,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            deterministic=self.deterministic
        )
        
        # Feed-forward network
        # For SwiGLU, double intermediate size for gating
        actual_intermediate = (
            self.actual_intermediate * 2 if self.activation == "swiglu"
            else self.actual_intermediate
        )
        
        self.ff_up = nn.Dense(
            actual_intermediate,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="ff_up"
        )
        self.ff_down = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="ff_down"
        )
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.ff_norm = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        
        # Get activation function
        self.act_fn = create_activation(self.activation)
        
    def _drop_path(self,
                  x: jnp.ndarray,
                  deterministic: bool) -> jnp.ndarray:
        """Apply drop path regularization.
        
        Args:
            x: Input tensor
            deterministic: Whether to run in deterministic mode
            
        Returns:
            Output with drop path applied
        """
        if deterministic or self.drop_path == 0.0:
            return x
            
        keep_prob = 1.0 - self.drop_path
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = jax.random.bernoulli(
            self.make_rng('dropout'),
            p=keep_prob,
            shape=shape
        )
        return jnp.where(mask, x / keep_prob, 0)
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 deterministic: Optional[bool] = None,
                 output_attentions: bool = False) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Apply transformer block to input.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            deterministic: Whether to run in deterministic mode
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Output tensor
                - Dict of auxiliary outputs
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        
        # Self-attention with normalization and residual
        normed = self.attention_norm(hidden_states)
        attention_output, attention_aux = self.attention(
            hidden_states=normed,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions
        )
        attention_output = self._drop_path(
            attention_output,
            deterministic=deterministic
        )
        hidden_states = hidden_states + attention_output
        
        # Feed-forward with normalization and residual
        normed = self.ff_norm(hidden_states)
        ff_output = self.ff_up(normed)
        ff_output = self.act_fn(ff_output)
        
        if not deterministic:
            ff_output = nn.Dropout(
                rate=self.hidden_dropout,
                deterministic=deterministic
            )(ff_output, deterministic=deterministic)
            
        ff_output = self.ff_down(ff_output)
        ff_output = self._drop_path(
            ff_output,
            deterministic=deterministic
        )
        hidden_states = hidden_states + ff_output
        
        aux_outputs = {
            'attention': attention_aux
        }
        
        return hidden_states, aux_outputs
