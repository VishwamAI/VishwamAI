import torch
import torch.nn.functional as F

def create_attention_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive generation.
    
    Args:
        seq_length: Length of input sequence
        device: Device to create tensor on
        
    Returns:
        Attention mask tensor of shape (seq_length, seq_length)
    """
    # Create causal mask
    mask = torch.full(
        (seq_length, seq_length),
        float("-inf"),
        device=device
    )
    mask = torch.triu(mask, diagonal=1)
    
    return mask

def apply_rope_rotation(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embedding.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        freqs_cis: Complex rotation tensor
        
    Returns:
        Rotated tensor
    """
    # Reshape for complex multiplication
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.reshape(*freqs_cis.shape[:2], 1, -1)
    
    # Apply rotation
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    
    return x_rotated.type_as(x)

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0
) -> torch.Tensor:
    """
    Precompute frequencies for rotary embeddings.
    
    Args:
        dim: Hidden dimension
        end: Sequence length
        theta: Base for exponential
        scaling_factor: RoPE scaling factor
    
    Returns:
        Complex rotation tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs) * scaling_factor
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def get_slopes(n_heads: int) -> torch.Tensor:
    """
    Get slopes for ALiBi positional embedding.
    
    Args:
        n_heads: Number of attention heads
        
    Returns:
        Tensor of slopes for each head
    """
    def get_slopes_power_of_2(n_heads: int) -> torch.Tensor:
        start = 2 ** (-(2 ** -(torch.log2(torch.tensor(n_heads)) - 3)))
        ratio = start
        return torch.pow(ratio, torch.arange(1, n_heads + 1))

    # Initialize slopes
    if torch.log2(torch.tensor(n_heads)).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** torch.floor(torch.log2(torch.tensor(n_heads)))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra_base = torch.tensor(1.0)
        extra_steps = n_heads - closest_power_of_2
        for _ in range(extra_steps):
            extra_base = torch.sqrt(extra_base)
            slopes = torch.cat([slopes, slopes[-1:] * extra_base])

    return -torch.sort(-slopes)[0]  # Sort in descending order

def maybe_cuda(t: torch.Tensor, cuda: bool = True) -> torch.Tensor:
    """Move tensor to CUDA if available and requested."""
    return t.cuda() if cuda and torch.cuda.is_available() else t

def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -1) -> torch.Tensor:
    """Pad tensor to multiple along specified dimension."""
    size = x.size(dim)
    pad_size = (multiple - (size % multiple)) % multiple
    if pad_size == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_size
    return torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)

def get_activation_fn(name: str):
    """Get activation function by name."""
    if name == "gelu":
        return F.gelu
    elif name == "relu":
        return F.relu
    elif name == "silu":
        return F.silu
    else:
        raise ValueError(f"Activation function {name} not supported")
