"""TPU-optimized Flash Attention implementation"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any
import numpy as np
from functools import partial

class FlashAttention(nn.Module):
    """Main Flash Attention interface for VishwamAI."""
    
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    causal: bool = False
    block_size: int = 128
    tpu_block_multiple: int = 128
    dtype: Any = jnp.float32

    def setup(self):
        """Initialize module parameters."""
        self.head_dim = self.hidden_dim // self.num_heads
        # Ensure block_size is a multiple of tpu_block_multiple
        self.block_size = ((self.block_size + self.tpu_block_multiple - 1) 
                          // self.tpu_block_multiple * self.tpu_block_multiple)
                          
        # QKV projection
        self.qkv = nn.Dense(
            features=3 * self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            name='qkv'
        )
        
        # Output projection
        self.out = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            name='out'
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Apply Flash Attention.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            mask: Optional attention mask
            deterministic: Whether to use deterministic behavior
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to q,k,v
        qkv = self.qkv(x)
        
        # Split into heads and separate q,k,v
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        
        # Scale query
        scale = jnp.sqrt(self.head_dim).astype(self.dtype)
        q = q / scale
        
        # Compute attention scores with optimized memory access pattern
        s = jnp.einsum('bqhd,bkhd->bhqk', q, k)
        
        if self.causal:
            causal_mask = jnp.greater_equal(
                jnp.arange(seq_len)[:, None],
                jnp.arange(seq_len)[None, :]
            )
            s = jnp.where(causal_mask.reshape(1, 1, seq_len, seq_len), s, -1e10)
            
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.reshape(batch_size, 1, seq_len, seq_len)
            s = jnp.where(mask, s, -1e10)
            
        # Compute attention weights
        attn = jax.nn.softmax(s, axis=-1)
        
        # Apply dropout if training
        if self.dropout_rate > 0.0 and not deterministic:
            keep_prob = 1.0 - self.dropout_rate
            dropout_rng = self.make_rng('dropout')
            attn = jax.random.bernoulli(
                dropout_rng,
                p=keep_prob,
                shape=attn.shape
            ) * attn / keep_prob
            
        # Compute weighted sum
        output = jnp.einsum('bhqk,bkhd->bqhd', attn, v)
        
        # Reshape back
        output = output.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Project output
        output = self.out(output)
        
        return output

    @classmethod
    def create(cls, config: Dict[str, Any]) -> 'FlashAttention':
        """Factory method to create FlashAttention instance."""
        return cls(
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            dropout_rate=config.get("attention_dropout_rate", 0.0),
            causal=config.get("causal", False),
            block_size=config.get("flash_attention_block_size", 128),
            tpu_block_multiple=config.get("tpu_block_multiple", 128),
            dtype=config.get("attention_precision", jnp.float32)
        )

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
        block_size: Size of blocks for tiling (should be tuned for TPU architecture)
        tpu_block_multiple: Block multiple for TPU memory alignment
        precision: Precision to use for intermediate calculations
        
    Returns:
        Output tensor and attention weights
    """
    # Ensure block_size is a multiple of tpu_block_multiple
    block_size = (block_size + tpu_block_multiple - 1) // tpu_block_multiple * tpu_block_multiple
    
    # Extract dimensions
    batch_size, q_len, num_heads, head_dim = q.shape
    _, kv_len, _, _ = k.shape
    
    # Cast to preferred precision for computation
    dtype = precision
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    
    # Initialize output and normalization terms
    # o will accumulate the weighted values
    o = jnp.zeros((batch_size, q_len, num_heads, head_dim), dtype=dtype)
    # l will accumulate the softmax normalization denominator
    l = jnp.zeros((batch_size, q_len, num_heads, 1), dtype=dtype)
    # m will track the max value for numerical stability
    m = jnp.ones((batch_size, q_len, num_heads, 1), dtype=dtype) * -jnp.inf
    
    # Constant scaling factor for Q*K
    scale = jnp.sqrt(head_dim).astype(dtype)
    
    # Perform Flash Attention with tiled processing
    # Process KV sequence in blocks to maintain O(1) memory complexity
    def scan_fn(carry, block_idx):
        o_partial, l_partial, m_partial = carry
        
        # Calculate start/end indices for this block
        block_start = block_idx * block_size
        block_end = jnp.minimum(block_start + block_size, kv_len)
        block_len = block_end - block_start
        
        # Extract key/value blocks - shape optimization for TPU memory layout
        k_block = lax.dynamic_slice(
            k, 
            (0, block_start, 0, 0), 
            (batch_size, block_len, num_heads, head_dim)
        )
        v_block = lax.dynamic_slice(
            v, 
            (0, block_start, 0, 0), 
            (batch_size, block_len, num_heads, head_dim)
        )
        
        # Compute attention scores for this block
        # Reshape to [batch, num_heads, q_len, block_len] for efficient matmul on TPU
        # Note: This transposed layout matches TPU's preferred memory access pattern
        s = jnp.einsum('bqhd,bkhd->bhqk', q, k_block, precision=lax.Precision.HIGHEST) / scale
        
        # Apply causal masking if needed
        if causal:
            causal_mask = jnp.greater_equal(
                jnp.arange(q_len)[:, None], 
                jnp.arange(block_start, block_end)[None, :]
            )
            causal_mask = causal_mask.reshape(1, 1, q_len, block_len)
            s = jnp.where(causal_mask, s, -1e10)
        
        # Apply attention mask if provided
        if mask is not None:
            mask_block = jax.lax.dynamic_slice(
                mask, 
                (0, 0, 0, block_start) if mask.ndim == 4 else (0, 0, block_start),
                (batch_size, num_heads, q_len, block_len) if mask.ndim == 4 else (batch_size, q_len, block_len)
            )
            if mask.ndim == 3:
                mask_block = mask_block.reshape(batch_size, 1, q_len, block_len)
            s = jnp.where(mask_block, s, -1e10)
        
        # Find max for numerical stability within this block
        m_block = jnp.max(s, axis=-1, keepdims=True)
        
        # Compute new running max combining current block and previous blocks
        m_new = jnp.maximum(m_partial, m_block)
        
        # Update exponential terms with numerical stability
        exp_scale = jnp.exp(m_partial - m_new)
        exp_s = jnp.exp(s - m_block)
        
        # Apply dropout if specified
        if dropout_rate > 0.0 and dropout_rng is not None:
            dropout_shape = (batch_size, num_heads, q_len, block_len)
            keep_prob = 1.0 - dropout_rate
            keep_mask = jax.random.bernoulli(dropout_rng, keep_prob, shape=dropout_shape)
            keep_mask = keep_mask / keep_prob  # Scale to preserve expectation
            exp_s = exp_s * keep_mask
        
        # Update normalization term l
        l_new = l_partial * exp_scale + jnp.sum(exp_s, axis=-1, keepdims=True)
        
        # Update output accumulation - efficient fused matmul pattern for TPU
        o_new = o_partial * exp_scale + jnp.einsum('bhqk,bkhd->bqhd', exp_s, v_block, precision=lax.Precision.HIGHEST)
        
        return (o_new, l_new, m_new), None
    
    # Number of blocks to process
    num_blocks = (kv_len + block_size - 1) // block_size
    
    # Scan over blocks
    (o, l, m), _ = lax.scan(
        scan_fn,
        init=(o, l, m),
        xs=jnp.arange(num_blocks)
    )
    
    # Final normalization
    out = o / l
    
    # Compute attention weights for visualization/analysis (optional)
    # This can be removed in production as it requires O(N²) memory
    if dropout_rate > 0 and dropout_rng is not None:
        attn_weights = None  # Don't compute weights when using dropout for memory efficiency
    else:
        qk = jnp.einsum('bqhd,bkhd->bhqk', q, k) / scale
        if causal:
            causal_mask = jnp.greater_equal(
                jnp.arange(q_len)[:, None], jnp.arange(kv_len)[None, :]
            )
            qk = jnp.where(causal_mask.reshape(1, 1, q_len, kv_len), qk, -1e10)
        attn_weights = jax.nn.softmax(qk, axis=-1)
    
    return out, attn_weights

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

@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def flash_attention_inference(
    q: jnp.ndarray,
    k: jnp.ndarray, 
    v: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    past_key_values: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    block_size: int = 128,
    head_dim: int = 64,
    num_heads: int = 8,
    use_fp8: bool = True
) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
    """Optimized Flash Attention implementation for inference.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        mask: Optional attention mask
        past_key_values: Optional cached key/values from previous forward pass
        block_size: Size of blocks for tiled attention
        head_dim: Size of attention head dimension
        num_heads: Number of attention heads
        use_fp8: Whether to use FP8 quantization
        
    Returns:
        Tuple of:
        - Output tensor [batch, heads, seq_len, head_dim]
        - New key/value cache tuple
    """
    # Add past key/values if provided
    if past_key_values is not None:
        past_k, past_v = past_key_values
        k = jnp.concatenate([past_k, k], axis=2)
        v = jnp.concatenate([past_v, v], axis=2)
    
    # Initialize output
    batch_size = q.shape[0]
    seq_len = q.shape[2]
    kv_len = k.shape[2]
    
    # For autoregressive generation, fast path when q is just the last token
    is_single_token = seq_len == 1
    
    # Fast path for single-token generation: direct attention calculation without tiling
    if is_single_token:
        # Compute attention scores
        scale = 1.0 / jnp.sqrt(head_dim)
        
        # Efficient matmul for single-token q
        scores = jnp.einsum('bhsd,bhtd->bhst', q, k) * scale
        
        # Apply mask if needed
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
        
        # Apply softmax
        probs = jax.nn.softmax(scores, axis=-1)
        
        # Weighted sum of values
        output = jnp.einsum('bhst,bhtd->bhsd', probs, v)
    else:
        # Multi-token case: use tiled flash attention
        output = jnp.zeros((batch_size, num_heads, seq_len, head_dim), dtype=q.dtype)
        l = jnp.zeros((batch_size, num_heads, seq_len, 1), dtype=q.dtype)
        m = jnp.ones((batch_size, num_heads, seq_len, 1), dtype=q.dtype) * -jnp.inf
        scale = 1.0 / jnp.sqrt(head_dim)
        
        # Process blocks of keys/values
        for block_start in range(0, kv_len, block_size):
            block_end = min(block_start + block_size, kv_len)
            k_block = k[:, :, block_start:block_end]
            v_block = v[:, :, block_start:block_end]
            
            # Compute scores for this block - optimized for TPU memory layout
            s = jnp.einsum('bhsd,bhtd->bhst', q, k_block) * scale
            
            # Apply mask if needed
            if mask is not None:
                mask_block = mask[:, :, :, block_start:block_end]
                s = jnp.where(mask_block, s, jnp.finfo(s.dtype).min)
            
            # Update running max
            m_block = jnp.max(s, axis=-1, keepdims=True)
            m_new = jnp.maximum(m, m_block)
            
            # Numerically stable update
            exp_scale = jnp.exp(m - m_new)
            exp_s = jnp.exp(s - m_block)
            
            # Update accumulators
            l_new = l * exp_scale + jnp.sum(exp_s, axis=-1, keepdims=True)
            output = (output * exp_scale + jnp.einsum('bhst,bhtd->bhsd', exp_s, v_block)) / l_new * l_new
            
            # Update running values
            l = l_new
            m = m_new
        
        # Final normalization
        output = output / l
    
    # Cache key/values for next forward pass
    present = (k, v) if past_key_values is not None else None
    
    return output, present

def create_fused_attention(config: Dict[str, Any]):
    """Create optimized TPU-specific fused attention functions."""
    
    @partial(jax.jit, static_argnums=(4, 5, 6))
    def fused_qkv_attention(
        qkv_proj: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        dropout_rng: Optional[jnp.ndarray] = None,
        dropout_rate: float = 0.0,
        num_heads: int = 8,
        head_dim: int = 64,
        block_size: int = 128,
    ):
        """Fused QKV attention for better TPU utilization.
        
        Args:
            qkv_proj: Combined QKV projection [batch, seq_len, 3 * num_heads * head_dim]
            mask: Optional attention mask
            dropout_rng: Optional RNG for dropout
            dropout_rate: Dropout probability
            num_heads: Number of attention heads
            head_dim: Size of attention head dimension
            block_size: Block size for tiled attention
            
        Returns:
            Output tensor and attention weights
        """
        # Split qkv_proj into q, k, v components
        batch_size, seq_len, _ = qkv_proj.shape
        qkv = qkv_proj.reshape(batch_size, seq_len, 3, num_heads, head_dim)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        
        # Run flash attention
        output, weights = flash_attention(
            q, k, v, 
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=dropout_rate,
            causal=config.get("causal_mask", True),
            block_size=block_size,
            tpu_block_multiple=config.get("tpu_block_multiple", 128),
            precision=config.get("compute_dtype", jnp.bfloat16)
        )
        
        # Reshape output
        output = output.reshape(batch_size, seq_len, -1)
        return output, weights
    
    return {
        "flash_attention": flash_attention,
        "flash_attention_inference": flash_attention_inference,
        "fused_qkv_attention": fused_qkv_attention
    }

class ChunkwiseCausalAttention(nn.Module):
    """Causal self-attention with chunked computation for memory efficiency."""
    
    hidden_dim: int
    num_heads: int
    chunk_size: int = 128
    dropout_rate: float = 0.0
    causal: bool = True
    dtype: Any = jnp.float32

    def setup(self):
        """Initialize module parameters."""
        self.head_dim = self.hidden_dim // self.num_heads
        
        # QKV projection
        self.qkv = nn.Dense(
            features=3 * self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            name='qkv'
        )
        
        # Output projection
        self.out = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            name='out'
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Apply chunked causal self-attention.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            mask: Optional attention mask
            deterministic: Whether to use deterministic behavior
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to q,k,v
        qkv = self.qkv(x)
        
        # Split into heads and separate q,k,v
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        
        # Process in chunks
        output = jnp.zeros((batch_size, seq_len, self.hidden_dim), dtype=self.dtype)
        
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            
            # Get current chunk
            q_chunk = q[:, chunk_start:chunk_end]
            
            # Compute attention scores
            scale = jnp.sqrt(self.head_dim).astype(self.dtype)
            scores = jnp.einsum('bqhd,bkhd->bhqk', q_chunk, k) / scale
            
            # Apply causal mask within chunk
            if self.causal:
                causal_mask = jnp.greater_equal(
                    jnp.arange(chunk_end - chunk_start)[:, None] + chunk_start,
                    jnp.arange(seq_len)[None, :]
                )
                scores = jnp.where(
                    causal_mask.reshape(1, 1, -1, seq_len),
                    scores,
                    -1e10
                )
            
            # Apply attention mask if provided
            if mask is not None:
                mask_chunk = mask[:, :, chunk_start:chunk_end]
                scores = jnp.where(mask_chunk, scores, -1e10)
            
            # Compute attention weights
            attn = jax.nn.softmax(scores, axis=-1)
            
            # Apply dropout during training
            if self.dropout_rate > 0.0 and not deterministic:
                keep_prob = 1.0 - self.dropout_rate
                dropout_rng = self.make_rng('dropout')
                attn = jax.random.bernoulli(
                    dropout_rng,
                    p=keep_prob,
                    shape=attn.shape
                ) * attn / keep_prob
            
            # Compute weighted sum
            chunk_output = jnp.einsum('bhqk,bkhd->bqhd', attn, v)
            
            # Reshape chunk output
            chunk_output = chunk_output.reshape(
                batch_size,
                chunk_end - chunk_start,
                self.hidden_dim
            )
            
            # Update output
            output = output.at[:, chunk_start:chunk_end].set(chunk_output)
        
        # Final projection
        output = self.out(output)
        
        return output

# For backwards compatibility
create_flash_attention = FlashAttention.create

__all__ = [
    'FlashAttention',
    'ChunkwiseCausalAttention',
    'flash_attention_inference',
    'create_flash_attention',
    'flash_attention',
    'mha_with_flash_attention'
]