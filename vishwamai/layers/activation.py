"""TPU-optimized activation layers for VishwamAI."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Dict

from vishwamai.kernels.activation import gelu_approx, silu_optimized, quick_gelu

class GELUActivation(nn.Module):
    """
    GELU activation optimized for TPU.
    
    Uses approximate GELU implementation for better TPU performance.
    """
    approximate: bool = True
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.approximate:
            # Use approximation for better TPU performance
            return gelu_approx(x).astype(self.dtype)
        else:
            # Exact GELU
            return jax.nn.gelu(x).astype(self.dtype)

class SwiGLUActivation(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation.
    
    Used in PaLM, LLaMA and other modern transformer architectures.
    References:
        https://arxiv.org/abs/2204.02311
    """
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    bias: bool = True
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden_features = self.hidden_features or self.in_features
        out_features = self.out_features or self.in_features
        
        # Project to hidden features
        gate = nn.Dense(hidden_features, use_bias=self.bias, 
                        dtype=self.dtype, name="gate")(x)
        value = nn.Dense(hidden_features, use_bias=self.bias,
                         dtype=self.dtype, name="value")(x)
        
        # Apply SwiGLU activation: gate * silu(value)
        hidden = value * jax.nn.swish(gate)
        
        # Project back to output dimension
        return nn.Dense(out_features, use_bias=self.bias,
                        dtype=self.dtype, name="output")(hidden)

class GeGLUActivation(nn.Module):
    """
    Gaussian Error Linear Unit Gated Linear Unit (GeGLU).
    
    References:
        https://arxiv.org/abs/2002.05202
    """
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    bias: bool = True
    dtype: Any = jnp.float32
    approximate_gelu: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden_features = self.hidden_features or self.in_features
        out_features = self.out_features or self.in_features
        
        # Project to hidden features
        gate = nn.Dense(hidden_features, use_bias=self.bias, 
                        dtype=self.dtype, name="gate")(x)
        value = nn.Dense(hidden_features, use_bias=self.bias,
                        dtype=self.dtype, name="value")(x)
        
        # Apply GeGLU activation: value * GELU(gate)
        if self.approximate_gelu:
            gelu_gate = gelu_approx(gate)
        else:
            gelu_gate = jax.nn.gelu(gate)
            
        hidden = value * gelu_gate
        
        # Project back to output dimension
        return nn.Dense(out_features, use_bias=self.bias,
                        dtype=self.dtype, name="output")(hidden)

class MixedActivation(nn.Module):
    """
    Mixed activation function that combines different activations.
    
    Can help models learn more complex functions.
    """
    activation_type: str = "swiglu"  # One of: swiglu, geglu, reglu, quick_gelu
    in_features: int = 0
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.activation_type in ["swiglu", "geglu", "reglu"]:
            # These are gated activations requiring linear projections
            if self.in_features == 0:
                raise ValueError("in_features must be specified for gated activations")
                
            hidden_features = self.hidden_features or self.in_features
            out_features = self.out_features or self.in_features
            
            # First projection
            gate = nn.Dense(hidden_features, dtype=self.dtype, name="gate")(x)
            value = nn.Dense(hidden_features, dtype=self.dtype, name="value")(x)
            
            # Apply appropriate activation to gate
            if self.activation_type == "swiglu":
                gate_act = jax.nn.swish(gate)
            elif self.activation_type == "geglu":
                gate_act = gelu_approx(gate)
            elif self.activation_type == "reglu":
                gate_act = jax.nn.relu(gate)
            
            # Apply gating
            hidden = value * gate_act
            
            # Project to output dimension
            return nn.Dense(out_features, dtype=self.dtype, name="output")(hidden)
        else:
            # Simple activations without projections
            if self.activation_type == "quick_gelu":
                return quick_gelu(x)
            elif self.activation_type == "silu":
                return silu_optimized(x)
            elif self.activation_type == "gelu":
                return gelu_approx(x)
            else:
                raise ValueError(f"Unknown activation type: {self.activation_type}")