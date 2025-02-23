"""Expert feed-forward network implementation."""

from typing import Optional, Tuple, Dict, Any, Callable
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

def create_expert_activation(name: str) -> Callable:
    """Create activation function for expert layer.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function
    """
    if name == "swiglu":
        return lambda x: jax.nn.silu(x[..., ::2]) * x[..., 1::2]
    elif name == "gelu":
        return nn.gelu
    elif name == "silu":
        return nn.swish
    else:
        return nn.relu

class ExpertFFN(nn.Module):
    """Expert feed-forward network with configurable activation."""
    
    hidden_size: int
    intermediate_size: Optional[int] = None
    activation: str = "swiglu"
    dropout_rate: float = 0.1
    use_bias: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    deterministic: bool = False
    
    def setup(self):
        """Initialize expert components."""
        actual_intermediate = (
            self.intermediate_size or self.hidden_size * 4
        )
        
        # For SwiGLU, double intermediate size for gating
        if self.activation == "swiglu":
            actual_intermediate *= 2
            
        # Up projection
        self.up_proj = nn.Dense(
            actual_intermediate,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="up_proj"
        )
        
        # Down projection
        self.down_proj = nn.Dense(
            self.hidden_size,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="down_proj"
        )
        
        # Activation function
        self.act_fn = create_expert_activation(self.activation)
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 deterministic: Optional[bool] = None) -> jnp.ndarray:
        """Apply expert computation.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            deterministic: Whether to run in deterministic mode
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        
        # Up projection
        hidden_states = self.up_proj(hidden_states)
        
        # Activation
        hidden_states = self.act_fn(hidden_states)
        
        # Dropout during training
        if not deterministic:
            hidden_states = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=False
            )(hidden_states)
            
        # Down projection
        hidden_states = self.down_proj(hidden_states)
        
        return hidden_states
        
    def init_expert(self,
                   rng: jax.random.PRNGKey,
                   scale: float = 1.0) -> None:
        """Initialize expert weights.
        
        Args:
            rng: PRNG key
            scale: Scaling factor for initialization
        """
        # Initialize up projection
        up_rng = jax.random.fold_in(rng, 0)
        up_shape = (self.hidden_size, self.intermediate_size or self.hidden_size * 4)
        self.up_proj.kernel = jax.random.normal(up_rng, up_shape) * 0.02 * scale
        if self.use_bias:
            self.up_proj.bias = jnp.zeros(up_shape[1])
            
        # Initialize down projection
        down_rng = jax.random.fold_in(rng, 1)
        down_shape = (up_shape[1], self.hidden_size)
        self.down_proj.kernel = jax.random.normal(down_rng, down_shape) * 0.02 * scale
        if self.use_bias:
            self.down_proj.bias = jnp.zeros(self.hidden_size)
