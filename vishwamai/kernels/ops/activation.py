"""TPU-optimized activation functions for VishwamAI."""

import jax
import jax.numpy as jnp
import numpy as np

def gelu_approx(x):
    """
    Fast approximation of GELU activation optimized for TPU.
    
    This implementation uses a simpler approximation that's more TPU-friendly:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor with GELU activation applied
    """
    # Constants optimized for TPU execution
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi).astype(x.dtype)
    coeff = 0.044715
    
    # Fast approximation with TPU-friendly operations
    inner = sqrt_2_over_pi * (x + coeff * x**3)
    return 0.5 * x * (1.0 + jnp.tanh(inner))

def silu_optimized(x):
    """
    Optimized SiLU (Swish) activation function for TPU.
    
    SiLU(x) = x * sigmoid(x)
    
    This implementation avoids potential numerical issues on TPU.
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor with SiLU activation applied
    """
    # Use logistic implementation for better TPU performance
    return x * jax.nn.sigmoid(x)

def quick_gelu(x):
    """
    Quick GELU approximation used by some transformer models.
    
    QuickGELU(x) = x * sigmoid(1.702 * x)
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with QuickGELU activation applied
    """
    return x * jax.nn.sigmoid(1.702 * x)

def hard_silu(x):
    """
    Hard version of SiLU that's more efficient on TPU.
    
    Uses min/max operations instead of sigmoid for faster computation.
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor with HardSiLU activation applied
    """
    lower = jnp.zeros_like(x)
    upper = jnp.ones_like(x)
    
    # Hard sigmoid: min(max(0, (x+3)/6), 1)
    hard_sigmoid = jnp.minimum(jnp.maximum(lower, (x + 3) / 6), upper)
    return x * hard_sigmoid

def leaky_relu_optimized(x, negative_slope=0.01):
    """
    Optimized leaky ReLU implementation for TPU.
    
    Args:
        x: Input tensor
        negative_slope: Slope for negative inputs
    
    Returns:
        Tensor with LeakyReLU applied
    """
    return jnp.maximum(x, negative_slope * x)
