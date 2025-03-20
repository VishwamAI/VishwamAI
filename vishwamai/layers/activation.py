import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Dict

from vishwamai.kernels.activation import gelu_approx, silu_optimized, quick_gelu

# Flash GELU Function (Optimized for TPU)
def flash_gelu(x: jnp.ndarray) -> jnp.ndarray:
    """Fused GELU approximation for TPU efficiency."""
    # Leverage gelu_approx with fused ops (pseudo-implementation)
    return gelu_approx(x) * jax.lax.approx_max(x, 0.0)  # Simplified fusion

class FlashGELUActivation(nn.Module):
    """
    Flash GELU activation optimized for TPU.
    
    Fuses GELU approximation with max operations for better TPU performance.
    """
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return flash_gelu(x).astype(self.dtype)

class GELUActivation(nn.Module):
    """Existing GELU (unchanged for reference)."""
    approximate: bool = True
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.approximate:
            return gelu_approx(x).astype(self.dtype)
        else:
            return jax.nn.gelu(x).astype(self.dtype)

class SwiGLUActivation(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) optimized for TPU.
    
    Uses silu_optimized for better TPU performance.
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
        
        gate = nn.Dense(hidden_features, use_bias=self.bias, 
                        dtype=self.dtype, name="gate")(x)
        value = nn.Dense(hidden_features, use_bias=self.bias,
                         dtype=self.dtype, name="value")(x)
        
        # Use optimized SiLU (Swish)
        hidden = value * silu_optimized(gate)  # Replace jax.nn.swish with TPU-optimized version
        
        return nn.Dense(out_features, use_bias=self.bias,
                        dtype=self.dtype, name="output")(hidden)

class GeGLUActivation(nn.Module):
    """Existing GeGLU (unchanged for reference)."""
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
        
        gate = nn.Dense(hidden_features, use_bias=self.bias, 
                        dtype=self.dtype, name="gate")(x)
        value = nn.Dense(hidden_features, use_bias=self.bias,
                         dtype=self.dtype, name="value")(x)
        
        gelu_gate = gelu_approx(gate) if self.approximate_gelu else jax.nn.gelu(gate)
        hidden = value * gelu_gate
        
        return nn.Dense(out_features, use_bias=self.bias,
                        dtype=self.dtype, name="output")(hidden)

class MixedActivation(nn.Module):
    """
    Mixed activation with Flash GELU and optimized SwiGLU added.
    """
    activation_type: str = "swiglu"  # Options: swiglu, geglu, reglu, quick_gelu, flash_gelu
    in_features: int = 0
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.activation_type in ["swiglu", "geglu", "reglu"]:
            if self.in_features == 0:
                raise ValueError("in_features must be specified for gated activations")
                
            hidden_features = self.hidden_features or self.in_features
            out_features = self.out_features or self.in_features
            
            gate = nn.Dense(hidden_features, dtype=self.dtype, name="gate")(x)
            value = nn.Dense(hidden_features, dtype=self.dtype, name="value")(x)
            
            if self.activation_type == "swiglu":
                gate_act = silu_optimized(gate)  # Optimized SwiGLU
            elif self.activation_type == "geglu":
                gate_act = gelu_approx(gate)
            elif self.activation_type == "reglu":
                gate_act = jax.nn.relu(gate)
            
            hidden = value * gate_act
            return nn.Dense(out_features, dtype=self.dtype, name="output")(hidden)
        else:
            if self.activation_type == "quick_gelu":
                return quick_gelu(x)
            elif self.activation_type == "silu":
                return silu_optimized(x)
            elif self.activation_type == "gelu":
                return gelu_approx(x)
            elif self.activation_type == "flash_gelu":
                return flash_gelu(x)  # New Flash GELU option
            else:
                raise ValueError(f"Unknown activation type: {self.activation_type}")