import math
from typing import Optional, Tuple
import torch

def find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    """Calculates the correction dimension for RoPE scaling.
    
    Args:
        num_rotations: Number of rotations
        dim: Model dimension
        base: Base value for the calculation
        max_seq_len: Maximum sequence length
    
    Returns:
        float: Correction dimension
    """
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

def find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int) -> Tuple[int, int]:
    """Finds the correction range for RoPE scaling.
    
    Args:
        low_rot: Lower rotation value
        high_rot: Higher rotation value
        dim: Model dimension
        base: Base value for the calculation
        max_seq_len: Maximum sequence length
    
    Returns:
        Tuple[int, int]: Low and high correction values
    """
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim-1)

def linear_ramp_factor(min_val: int, max_val: int, dim: int) -> torch.Tensor:
    """Calculates linear ramp factor for smooth RoPE scaling.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        dim: Dimension size
    
    Returns:
        torch.Tensor: Ramp factor tensor
    """
    if min_val == max_val:
        max_val += 0.001  # Prevent division by zero
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    rope_factor: float = 1.0,
    original_length: Optional[int] = None,
    beta_fast: Optional[float] = None,
    beta_slow: Optional[float] = None
) -> torch.Tensor:
    """Precomputes frequency-based complex exponential values for rotary positional embeddings.
    
    This function implements NTK-aware RoPE scaling when original_length is provided.
    
    Args:
        dim: Model dimension (must be even)
        end: Maximum sequence length
        theta: Base value for frequency computation (default: 10000.0)
        rope_factor: Scaling factor for RoPE (default: 1.0)
        original_length: Original sequence length for NTK-aware scaling (optional)
        beta_fast: Fast beta value for NTK-aware scaling (optional)
        beta_slow: Slow beta value for NTK-aware scaling (optional)
    
    Returns:
        torch.Tensor: Complex tensor containing the precomputed frequency values
        
    Raises:
        ValueError: If dim is not even or if end is less than 1
    """
    if dim % 2 != 0:
        raise ValueError("Model dimension must be even")
    if end < 1:
        raise ValueError("End position must be at least 1")
        
    # Compute base frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    # Apply NTK-aware scaling if original_length is provided
    if original_length is not None and end > original_length and beta_fast is not None and beta_slow is not None:
        try:
            low, high = find_correction_range(beta_fast, beta_slow, dim, theta, original_length)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth
        except Exception as e:
            print(f"Warning: Error in NTK-aware scaling: {e}. Falling back to base frequencies.")

    # Compute position-based frequencies
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    
    # Convert to complex values
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embeddings to the input tensor.
    
    Args:
        x: Input tensor of shape (..., dim)
        freqs_cis: Complex tensor of precomputed frequencies
        
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied
        
    Raises:
        ValueError: If input shapes are incompatible
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError("Last dimension of input tensor must be even")
    if freqs_cis.shape[-1] != x.shape[-1] // 2:
        raise ValueError("Frequency tensor shape incompatible with input tensor")
        
    dtype = x.dtype
    
    try:
        # Reshape input to complex numbers
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Reshape frequencies for broadcasting
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
        
        # Apply complex multiplication
        x_rot = torch.view_as_real(x_complex * freqs_cis)
        
        # Restore original shape and dtype
        return x_rot.flatten(3).to(dtype)
    except Exception as e:
        raise RuntimeError(f"Error applying rotary embeddings: {e}")
