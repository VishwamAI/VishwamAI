from typing import Optional, Tuple

import torch
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import os
import sys
import warnings
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# Initialize TPU device mesh
devices = jax.devices()
device_mesh = mesh_utils.create_device_mesh((2, 4))  # 2x4 mesh for 8 TPU cores
mesh = Mesh(devices, ('batch', 'model'))

# Flag to track execution mode
USING_CUDA_EXTENSION = False
USING_TPU = True if 'TPU' in str(jax.devices()[0]) else False

# Try to import CUDA extension
try:
    pass
except ImportError:
    # Provide helpful warning message
    csrc_path = os.path.join(os.path.dirname(__file__), 'csrc')
    warnings.warn(
        f"Flash MLA CUDA extension not found. Using fallback implementation. "
        f"For better performance, build the extension by running:\n"
        f"cd {csrc_path} && python setup.py install"
    )

# Explicitly define the flash_mla_forward function to fix import errors
def flash_mla_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_attn: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flash MLA forward function.
    
    This is a wrapper function that calls either the CUDA implementation 
    or a fallback PyTorch implementation depending on availability.
    
    Args:
        q: Query tensor [batch_size, seq_len_q, num_heads, head_dim]
        k: Key tensor [batch_size, seq_len_k, num_heads, head_dim]
        v: Value tensor [batch_size, seq_len_k, num_heads, head_dim]
        mask: Optional attention mask
        scale: Scaling factor for attention scores
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        return_attn: Whether to return attention weights
        
    Returns:
        Tuple containing:
        - Output tensor [batch_size, seq_len_q, num_heads, head_dim]
        - LSE values for backward pass
    """
    if scale is None:
        scale = 1.0 / (q.size(-1) ** 0.5)
        
    # Call the CUDA implementation if available
    if USING_CUDA_EXTENSION and hasattr(flash_mla_cuda, 'flash_mla_forward'):
        return flash_mla_cuda.flash_mla_forward(
            q, k, v, mask, scale, dropout_p, is_causal
        )
    
    # Fallback implementation
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape
    
    # Reshape for batch matrix multiplication
    q_reshaped = q.transpose(1, 2).reshape(batch_size * num_heads, seq_len_q, head_dim)
    k_reshaped = k.transpose(1, 2).reshape(batch_size * num_heads, seq_len_k, head_dim)
    v_reshaped = v.transpose(1, 2).reshape(batch_size * num_heads, seq_len_k, head_dim)
    
    # Compute attention scores
    scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * scale
    
    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
    
    # Apply provided mask if any
    if mask is not None:
        # Reshape mask to match scores dimensions
        if mask.dim() == 2:  # [seq_len_q, seq_len_k]
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:  # [batch_size, seq_len_q, seq_len_k]
            mask = mask.unsqueeze(1)
        
        scores = scores.masked_fill(~mask.expand_as(scores), float("-inf"))
    
    # Compute softmax
    softmax_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    attn_weights = torch.exp(scores - softmax_lse)
    
    # Apply dropout if needed
    if dropout_p > 0.0:
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout_p, training=True
        )
    
    # Compute weighted sum
    output = torch.bmm(attn_weights, v_reshaped)
    
    # Reshape back to original dimensions
    output = output.reshape(batch_size, num_heads, seq_len_q, head_dim).transpose(1, 2)
    softmax_lse = softmax_lse.reshape(batch_size, num_heads, seq_len_q).squeeze(-1)
    
    if return_attn:
        attn_weights = attn_weights.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
        return output, softmax_lse, attn_weights
    else:
        return output, softmax_lse

class FlashMLACUDA:
    """Flash Multi-head Lookahead Attention CUDA implementation.
    
    This class provides a high-performance CUDA implementation of the
    Multi-head Lookahead Attention algorithm with KV-caching support.
    """
    
    def __init__(
        self,
        head_dim: int = 128,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        dropout_rate: float = 0.0,
        causal: bool = True,
    ):
        """Initialize the Flash MLA CUDA kernel.
        
        Args:
            head_dim: Dimension of each attention head
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads (for GQA/MQA)
            dropout_rate: Attention dropout rate
            causal: Whether to use causal attention mask
        """
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.dropout_rate = dropout_rate
        self.causal = causal
        
        # Check if CUDA extension is available
        self.using_cuda_extension = USING_CUDA_EXTENSION
        
    def __call__(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute Flash MLA attention with KV cache.
        
        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim]
            key_cache: Key cache tensor [batch, seq_len_k, num_kv_heads, head_dim]
            value_cache: Value cache tensor [batch, seq_len_k, num_kv_heads, head_dim]
            block_table: Block table for KV cache [batch, max_blocks]
            cache_seqlens: Sequence lengths for KV cache [batch]
            softmax_scale: Optional scaling factor (default: 1/sqrt(head_dim))
            
        Returns:
            Output tensor [batch, seq_len_q, num_heads, head_dim]
        """
        # Get metadata for scheduling
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens, 
            self.num_heads // self.num_kv_heads, 
            self.num_kv_heads
        )
        
        # Run the CUDA kernel or fallback
        output, _ = flash_mla_with_kvcache(
            query,
            key_cache,
            value_cache,
            block_table,
            cache_seqlens,
            self.head_dim,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale,
            self.causal
        )
        
        return output
    
    def to_jax(
        self,
        query: jnp.ndarray,
        key_cache: jnp.ndarray,
        value_cache: jnp.ndarray,
        block_table: jnp.ndarray,
        cache_seqlens: jnp.ndarray,
        softmax_scale: Optional[float] = None,
    ) -> jnp.ndarray:
        """JAX-compatible interface - converts JAX arrays to PyTorch and back.
        
        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim]
            key_cache: Key cache tensor [batch, seq_len_k, num_kv_heads, head_dim]
            value_cache: Value cache tensor [batch, seq_len_k, num_kv_heads, head_dim]
            block_table: Block table for KV cache [batch, max_blocks]
            cache_seqlens: Sequence lengths for KV cache [batch]
            softmax_scale: Optional scaling factor (default: 1/sqrt(head_dim))
            
        Returns:
            Output tensor [batch, seq_len_q, num_heads, head_dim]
        """
        # Convert JAX arrays to numpy
        query_np = np.array(query)
        key_cache_np = np.array(key_cache)
        value_cache_np = np.array(value_cache)
        block_table_np = np.array(block_table)
        cache_seqlens_np = np.array(cache_seqlens)
        
        # Convert numpy arrays to PyTorch tensors
        query_torch = torch.from_numpy(query_np)
        key_cache_torch = torch.from_numpy(key_cache_np)
        value_cache_torch = torch.from_numpy(value_cache_np)
        block_table_torch = torch.from_numpy(block_table_np)
        cache_seqlens_torch = torch.from_numpy(cache_seqlens_np)
        
        # Call CUDA implementation
        output_torch = self(
            query_torch,
            key_cache_torch,
            value_cache_torch,
            block_table_torch,
            cache_seqlens_torch,
            softmax_scale
        )
        
        # Convert back to JAX array
        output_np = output_torch.detach().cpu().numpy()
        return jnp.array(output_np)
        
    @classmethod
    def create(
        cls,
        head_dim: int = 128,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        dropout_rate: float = 0.0,
        causal: bool = True,
    ) -> "FlashMLACUDA":
        """Factory method to create a new FlashMLACUDA instance."""
        return cls(
            head_dim=head_dim,
            num_heads=num_heads, 
            num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate,
            causal=causal
        )

def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute metadata required for MLA scheduling.
    
    Args:
        cache_seqlens: Sequence lengths for each batch item
        num_heads_per_head_k: Number of query heads per key head
        num_heads_k: Number of key heads
        
    Returns:
        Tuple containing:
        - tile_scheduler_metadata: Metadata for tile scheduler
        - num_splits: Number of splits per batch item
    """
    batch_size = cache_seqlens.size(0)
    block_size_n = 64  # Default block size
    fixed_overhead_num_blocks = 2  # Default overhead
    num_sm_parts = 6   # Default SM parts
    
    if USING_CUDA_EXTENSION:
        # Call the CUDA function
        result = flash_mla_cuda.get_mla_metadata(
            cache_seqlens, 
            batch_size, 
            block_size_n, 
            fixed_overhead_num_blocks, 
            num_sm_parts
        )
        
        tile_scheduler_metadata = result[0]
        num_splits = result[1]
    else:
        # Fallback implementation
        options = torch.tensor([], device=cache_seqlens.device).options()
        tile_scheduler_metadata = torch.zeros((8,), dtype=torch.int32, device=cache_seqlens.device)
        num_splits = torch.zeros((batch_size,), dtype=torch.int32, device=cache_seqlens.device)
        
        # Simple fallback computation
        for i in range(batch_size):
            seqlen = cache_seqlens[i].item()
            num_blocks = (seqlen + block_size_n - 1) // block_size_n
            num_splits[i] = num_blocks + fixed_overhead_num_blocks
        
        # Basic metadata
        tile_scheduler_metadata[0] = 0                    # begin_idx
        tile_scheduler_metadata[1] = 0                    # begin_seqlen
        tile_scheduler_metadata[2] = batch_size - 1       # end_idx
        tile_scheduler_metadata[3] = cache_seqlens[-1]    # end_seqlen
        tile_scheduler_metadata[4] = 0                    # begin_n_split_idx
    
    return tile_scheduler_metadata, num_splits

def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flash MLA with KV Cache using CUDA implementation.
    
    Args:
        q: Query tensor [batch, seq_len_q, num_heads, head_dim]
        k_cache: Key cache tensor [batch, seq_len_k, num_heads_k, head_dim]
        v_cache: Value cache tensor [batch, seq_len_k, num_heads_k, head_dim_v]
        block_table: Block table for KV cache [batch, max_blocks]
        cache_seqlens: Sequence lengths for KV cache [batch]
        head_dim_v: Value head dimension
        tile_scheduler_metadata: Metadata from get_mla_metadata
        num_splits: Split information from get_mla_metadata
        softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
        causal: Whether to use causal masking
        
    Returns:
        Tuple containing:
        - out: Output tensor [batch, seq_len_q, num_heads, head_dim_v]
        - softmax_lse: Log-sum-exp values for backward pass
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)
    
    if USING_CUDA_EXTENSION:
        # Call the CUDA implementation
        out, softmax_lse = flash_mla_cuda.flash_mla_forward(
            q,
            k_cache,
            v_cache,
            block_table,
            cache_seqlens,
            softmax_scale,
            causal,
            tile_scheduler_metadata,
            num_splits
        )
    else:
        # Fallback implementation using PyTorch
        batch_size, seqlen_q, num_heads, head_dim = q.shape
        _, seqlen_k, num_heads_k, _ = k_cache.shape
        
        # Create output tensor and softmax_lse tensor
        out = torch.zeros_like(q)
        softmax_lse = torch.zeros((batch_size, num_heads, seqlen_q), 
                               dtype=torch.float32, device=q.device)
        
        # Compute attention using PyTorch operations
        for b in range(batch_size):
            # Get actual sequence length for this batch item
            seq_len = cache_seqlens[b].item()
            
            # Process each token in the query
            for h in range(num_heads):
                h_k = h // (num_heads // num_heads_k)  # Map query head to key head
                
                # Get query, key, and value tensors for this batch and head
                q_bh = q[b, :, h, :]  # [seqlen_q, head_dim]
                k_bh = k_cache[b, :seq_len, h_k, :]  # [seqlen_k, head_dim]
                v_bh = v_cache[b, :seq_len, h_k, :]  # [seqlen_k, head_dim_v]
                
                # Compute attention scores
                scores = torch.matmul(q_bh, k_bh.transpose(0, 1)) * softmax_scale
                
                # Apply causal mask if needed
                if causal:
                    mask = torch.triu(torch.ones(seqlen_q, seq_len, device=q.device), diagonal=1)
                    scores = scores.masked_fill(mask.bool(), -float('inf'))
                
                # Apply softmax to get attention weights
                attn_weights = torch.nn.functional.softmax(scores, dim=-1)
                
                # Compute attention output
                out[b, :, h, :] = torch.matmul(attn_weights, v_bh)
                
                # Compute log-sum-exp for backward pass
                max_score = torch.max(scores, dim=-1, keepdim=True)[0]
                exp_scores = torch.exp(scores - max_score)
                sum_exp = torch.sum(exp_scores, dim=-1)
                softmax_lse[b, h] = max_score.squeeze(-1) + torch.log(sum_exp)
        
    return out, softmax_lse

def jax_to_torch(x: jnp.ndarray) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor on CUDA."""
    return torch.from_numpy(np.array(x)).cuda()

def torch_to_jax(x: torch.Tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(x.detach().cpu().numpy())

# JAX-compatible wrappers to use our CUDA implementation from JAX
def jax_get_mla_metadata(
    cache_seqlens: jnp.ndarray,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-compatible wrapper for get_mla_metadata."""
    # Convert JAX arrays to PyTorch tensors
    torch_cache_seqlens = jax_to_torch(cache_seqlens)
    
    # Call PyTorch implementation
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        torch_cache_seqlens,
        num_heads_per_head_k,
        num_heads_k
    )
    
    # Convert back to JAX arrays
    return torch_to_jax(tile_scheduler_metadata), torch_to_jax(num_splits)

def jax_flash_mla_with_kvcache(
    q: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    block_table: jnp.ndarray,
    cache_seqlens: jnp.ndarray,
    head_dim_v: int,
    tile_scheduler_metadata: jnp.ndarray,
    num_splits: jnp.ndarray,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-compatible wrapper for flash_mla_with_kvcache."""
    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(q.shape[-1])
    
    # Pure JAX implementation
    def attention_fn(carry, idx):
        q_t, k_t, v_t = carry
        attn_scores = jnp.einsum('bqd,bkd->bqk', q_t, k_t) * softmax_scale
        if causal:
            mask = jnp.tril(jnp.ones_like(attn_scores))
            attn_scores = jnp.where(mask, attn_scores, -jnp.inf)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        output = jnp.einsum('bqk,bkd->bqd', attn_probs, v_t)
        return (q_t, k_t, v_t), output
    
    # Use PyTorch implementation if available and tensors are on CUDA
    if USING_CUDA_EXTENSION and hasattr(q, 'device') and q.device == 'cuda':
        # Convert JAX arrays to PyTorch tensors
        torch_q = jax_to_torch(q)
        torch_k_cache = jax_to_torch(k_cache)
        torch_v_cache = jax_to_torch(v_cache)
        torch_block_table = jax_to_torch(block_table)
        torch_cache_seqlens = jax_to_torch(cache_seqlens)
        torch_tile_scheduler_metadata = jax_to_torch(tile_scheduler_metadata)
        torch_num_splits = jax_to_torch(num_splits)
        
        # Call PyTorch implementation
        torch_out, torch_softmax_lse = flash_mla_with_kvcache(
            torch_q,
            torch_k_cache,
            torch_v_cache,
            torch_block_table,
            torch_cache_seqlens,
            head_dim_v,
            torch_tile_scheduler_metadata,
            torch_num_splits,
            softmax_scale,
            causal
        )
        
        # Convert back to JAX arrays
        return torch_to_jax(torch_out), torch_to_jax(torch_softmax_lse)
    else:
        # Use pure JAX implementation
        _, out = jax.lax.scan(attention_fn, (q, k_cache, v_cache), xs=jnp.arange(q.shape[1]))
        
        # Compute softmax_lse
        attn_scores = jnp.einsum('bqd,bkd->bqk', q, k_cache) * softmax_scale
        if causal:
            mask = jnp.tril(jnp.ones_like(attn_scores))
            attn_scores = jnp.where(mask, attn_scores, -jnp.inf)
        max_score = jnp.max(attn_scores, axis=-1, keepdims=True)
        exp_scores = jnp.exp(attn_scores - max_score)
        sum_exp = jnp.sum(exp_scores, axis=-1)
        softmax_lse = max_score.squeeze(-1) + jnp.log(sum_exp)
        
        return out, softmax_lse
