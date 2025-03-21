from typing import Optional, Tuple

import torch
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import os
import sys
import warnings

# Flag to track if we're using the CUDA extension or fallback
USING_CUDA_EXTENSION = False

# Try to import CUDA extension
try:
    import flash_mla_cuda
    USING_CUDA_EXTENSION = True
except ImportError:
    # Provide helpful warning message
    csrc_path = os.path.join(os.path.dirname(__file__), 'csrc')
    warnings.warn(
        f"Flash MLA CUDA extension not found. Using fallback implementation. "
        f"For better performance, build the extension by running:\n"
        f"cd {csrc_path} && python setup.py install"
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
