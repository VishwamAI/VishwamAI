"""TPU-optimized Flash Attention implementation"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Tuple, Dict, Any
import numpy as np

def flash_attention(
    q: jnp.ndarray,  # shape: [batch, q_len, num_heads, head_dim]
    k: jnp.ndarray,  # shape: [batch, kv_len, num_heads, head_dim] 
    v: jnp.ndarray,  # shape: [batch, kv_len, num_heads, head_dim]
    mask: Optional[jnp.ndarray] = None,
    dropout_rng: Optional[jnp.ndarray] = None,
    dropout_rate: float = 0.0,
    causal: bool = False,
    block_size: int = 128,
    tpu_block_multiple: int = 128,
    precision: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """TPU-optimized Flash Attention implementation.
    
    Args:
        q: Query tensor
        k: Key tensor  
        v: Value tensor
        mask: Optional attention mask
        dropout_rng: Optional RNG for dropout
        dropout_rate: Dropout probability
        causal: Whether to apply causal masking
        block_size: Size of blocks for chunked attention computation
        tpu_block_multiple: Multiple of block size optimized for TPU
        precision: Numerical precision to use
        
    Returns:
        Tuple of (output, attention probs)
    """
    batch_size, q_len, num_heads, head_dim = q.shape
    _, kv_len, _, _ = k.shape
    
    # Adjust block size to TPU-friendly multiple
    block_size = (block_size + tpu_block_multiple - 1) // tpu_block_multiple * tpu_block_multiple
    
    # Cast to specified precision
    dtype = precision
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    
    # Scale query
    scaling = jnp.sqrt(head_dim).astype(dtype)
    q = q / scaling
    
    # Initialize output accumulators
    o = jnp.zeros((batch_size, q_len, num_heads, head_dim), dtype=dtype)
    l = jnp.zeros((batch_size, q_len, num_heads, 1), dtype=dtype)
    m = jnp.ones((batch_size, q_len, num_heads, 1), dtype=dtype) * -jnp.inf
    
    # Process attention in blocks
    for block_start in range(0, kv_len, block_size):
        block_end = min(block_start + block_size, kv_len)
        
        # Get key/value block
        k_block = jax.lax.dynamic_slice(
            k,
            (0, block_start, 0, 0),
            (batch_size, block_end - block_start, num_heads, head_dim)
        )
        v_block = jax.lax.dynamic_slice(
            v,
            (0, block_start, 0, 0),
            (batch_size, block_end - block_start, num_heads, head_dim)
        )
        
        # Compute attention scores for block
        s = jnp.einsum('bqhd,bkhd->bqhk', q, k_block)
        
        # Apply masking
        if causal:
            causal_mask = jnp.triu(
                jnp.ones((q_len, block_end - block_start), dtype=bool),
                k=block_start + 1
            )
            s = jnp.where(causal_mask[:, None, :], -jnp.inf, s)
        
        if mask is not None:
            mask_block = jax.lax.dynamic_slice(
                mask,
                (0, 0, block_start),
                (batch_size, q_len, block_end - block_start)
            )
            s = jnp.where(mask_block[..., None, :], -jnp.inf, s)
        
        # Update running maximum
        m_block = jnp.max(s, axis=-1, keepdims=True)
        m_new = jnp.maximum(m, m_block)
        
        # Update output with re-normalized contributions
        exp_scale = jnp.exp(m - m_new)
        exp_s = jnp.exp(s - m_block)
        
        # Apply dropout if specified
        if dropout_rate > 0.0 and dropout_rng is not None:
            dropout_mask = jax.random.bernoulli(
                dropout_rng,
                p=1.0 - dropout_rate,
                shape=exp_s.shape
            )
            exp_s = exp_s * dropout_mask / (1.0 - dropout_rate)
        
        # Accumulate weighted values
        l_new = l * exp_scale + jnp.sum(exp_s, axis=-1, keepdims=True)
        o_new = o * exp_scale + jnp.einsum('bqhk,bkhd->bqhd', exp_s, v_block)
        
        # Update accumulators
        l = l_new
        o = o_new
        m = m_new
    
    # Compute final output
    o = o / l
    
    # Compute attention probabilities (optional)
    p = None
    if not jnp.isnan(o).any():  # Only compute if numerically stable
        p = jnp.exp(s - m) / l
    
    return o, p

def mha_with_flash_attention(
    qkv: jnp.ndarray,
    config: Dict[str, Any],
    mask: Optional[jnp.ndarray] = None,
    deterministic: bool = True
) -> jnp.ndarray:
    """Multi-head attention using Flash Attention.
    
    Args:
        qkv: Combined query/key/value tensor [batch, seq_len, 3 * hidden_dim]
        config: Model configuration
        mask: Optional attention mask
        deterministic: Whether to use deterministic attention (no dropout)
        
    Returns:
        Output tensor [batch, seq_len, hidden_dim]
    """
    batch_size, seq_len, _ = qkv.shape
    hidden_dim = config["hidden_dim"]
    num_heads = config["num_heads"]
    head_dim = hidden_dim // num_heads
    dropout_rate = config.get("attention_dropout_rate", 0.0)
    
    # Split qkv into separate tensors
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    
    # Generate dropout RNG if needed
    dropout_rng = None
    if dropout_rate > 0.0 and not deterministic:
        dropout_rng = jax.random.PRNGKey(0)  # Should use proper RNG handling
    
    # Call flash attention
    output, _ = flash_attention(
        q=q,
        k=k,
        v=v,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate if not deterministic else 0.0,
        causal=config.get("causal", False),
        block_size=config.get("flash_attention_block_size", 128),
        tpu_block_multiple=config.get("tpu_block_multiple", 128),
        precision=config.get("attention_precision", jnp.float32)
    )
    
    # Reshape output
    output = output.reshape(batch_size, seq_len, hidden_dim)
    
    return output

class FlashAttentionLayer:
    """Flash Attention layer with TPU optimizations."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        name: str = "flash_attention"
    ):
        """Initialize Flash Attention layer.
        
        Args:
            config: Model configuration
            name: Name of the layer
        """
        self.config = config
        self.name = name
        
        # Get dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.hidden_dim // self.num_heads
        
        # Initialize parameters
        self.qkv_proj = None  # Will be initialized on first call
        self.out_proj = None
    
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Apply Flash Attention layer.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            mask: Optional attention mask
            deterministic: Whether to use deterministic attention
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Initialize parameters if needed
        if self.qkv_proj is None:
            self.qkv_proj = jnp.zeros((self.hidden_dim, 3 * self.hidden_dim))
            self.out_proj = jnp.zeros((self.hidden_dim, self.hidden_dim))
        
        # Project input to q,k,v
        qkv = jnp.einsum('bsh,hd->bsd', x, self.qkv_proj)
        
        # Apply flash attention
        output = mha_with_flash_attention(
            qkv=qkv,
            config=self.config,
            mask=mask,
            deterministic=deterministic
        )
        
        # Project output
        output = jnp.einsum('bsh,hd->bsd', output, self.out_proj)
        
        return output

def create_flash_attention_layer(
    config: Dict[str, Any],
    name: str = "flash_attention"
) -> FlashAttentionLayer:
    """Create a Flash Attention layer."""
    return FlashAttentionLayer(config=config, name=name)