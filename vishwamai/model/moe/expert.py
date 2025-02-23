"""Expert module for Mixture of Experts."""

from typing import Optional, Callable, Any
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

def create_activation(name: str) -> Callable:
    """Create activation function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function
    """
    if name == "swiglu":
        return lambda x: jax.nn.silu(x[..., ::2]) * x[..., 1::2]
    elif name == "gelu":
        return nn.gelu
    elif name == "relu":
        return nn.relu
    elif name == "silu":
        return nn.swish
    else:
        raise ValueError(f"Unknown activation function: {name}")

class ExpertLayer(nn.Module):
    """Expert feed-forward network layer."""
    
    hidden_size: int
    intermediate_size: Optional[int] = None
    activation: str = "swiglu"
    dropout_rate: float = 0.1
    use_bias: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.kaiming_uniform()
    bias_init: Callable = nn.initializers.zeros
    deterministic: bool = False
    
    def setup(self):
        """Initialize expert layer."""
        # If intermediate size not specified, use 4x hidden size
        actual_intermediate = (self.intermediate_size or self.hidden_size * 4)
        
        # For SwiGLU, we need twice the width for gating
        if self.activation == "swiglu":
            actual_intermediate = actual_intermediate * 2
            
        # Up projection
        self.up_proj = nn.Dense(
            actual_intermediate,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="up_proj"
        )
        
        # Down projection
        self.down_proj = nn.Dense(
            self.hidden_size,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="down_proj"
        )
        
        # Get activation function
        self.act_fn = create_activation(self.activation)
        
    def __call__(self, 
                 hidden_states: jnp.ndarray,
                 deterministic: Optional[bool] = None) -> jnp.ndarray:
        """Apply expert to input.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            deterministic: Whether to run in deterministic mode
            
        Returns:
            Transformed tensor [batch, seq_len, hidden_size]
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        
        # Up project and activate
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        
        # Apply dropout during training
        if not deterministic:
            hidden_states = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(hidden_states, deterministic=deterministic)
            
        # Down project
        hidden_states = self.down_proj(hidden_states)
        
        # Apply dropout during training
        if not deterministic:
            hidden_states = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(hidden_states, deterministic=deterministic)
            
        return hidden_states
        
    def init_expert(self, rng: jax.random.PRNGKey,
                   scale: float = 1.0) -> None:
        """Initialize expert weights with custom scaling.
        
        Args:
            rng: PRNG key
            scale: Scaling factor for initialization
        """
        # Split RNG key for each projection
        rng_up, rng_down = jax.random.split(rng)
        
        # Scale the kernel initializers
        scaled_kernel_init = lambda *args: self.kernel_init(*args) * scale
        
        # Re-initialize projections with scaled initializers
        self.up_proj = nn.Dense(
            self.up_proj.features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=scaled_kernel_init,
            bias_init=self.bias_init,
            name="up_proj"
        )
        
        self.down_proj = nn.Dense(
            self.down_proj.features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=scaled_kernel_init,
            bias_init=self.bias_init,
            name="down_proj"
        )
        
    def reset_parameters(self) -> None:
        """Reset parameters to initial values."""
        # Re-initialize with original initializers
        self.up_proj = nn.Dense(
            self.up_proj.features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="up_proj"
        )
        
        self.down_proj = nn.Dense(
            self.down_proj.features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="down_proj"
        )
