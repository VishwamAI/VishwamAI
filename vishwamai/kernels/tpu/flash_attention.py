"""TPU-optimized Flash Attention implementation."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Tuple, Union, Dict, Any, NamedTuple
import numpy as np
from functools import partial

from vishwamai.kernels.core.kernel import Kernel, KernelConfig
from vishwamai.kernels.core.kernel_manager import HardwareType
from .tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

class FlashAttentionOutput(NamedTuple):
    """Output from flash attention operation."""
    output: jnp.ndarray
    logsumexp: Optional[jnp.ndarray] = None

class TPUFlashAttention:
    """TPU-optimized Flash Attention implementation.
    
    Provides an efficient implementation of Flash Attention optimized for TPU hardware
    with O(1) memory complexity, allowing for processing longer sequences.
    """
    
    def __init__(
        self,
        block_size: int = 128,
        precision: Optional[lax.Precision] = None,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        dropout_rate: float = 0.0,
        use_bfloat16: bool = True,
    ):
        """Initialize TPU Flash Attention.
        
        Args:
            block_size: Size of blocks for tiling (should be multiple of 128 for TPU)
            precision: JAX precision setting for computation
            causal: Whether to use causal masking
            softmax_scale: Optional scale factor for attention scores (default: 1/sqrt(head_dim))
            dropout_rate: Rate for attention dropout
            use_bfloat16: Whether to use bfloat16 for computation
        """
        self.block_size = block_size
        self.precision = precision or lax.Precision.HIGHEST
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_rate = dropout_rate
        self.use_bfloat16 = use_bfloat16
        
        # Ensure block_size is appropriate for TPU
        if block_size % 128 != 0:
            raise ValueError("Block size must be a multiple of 128 for TPU")
    
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        return_logsumexp: bool = False,
        training: bool = False,
    ) -> Union[jnp.ndarray, FlashAttentionOutput]:
        """Perform Flash Attention optimized for TPU.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len_k, head_dim]
            mask: Optional attention mask
            return_logsumexp: Whether to return logsumexp values (useful for backward pass)
            training: Whether in training mode (for dropout)
            
        Returns:
            Attention output, with optional logsumexp if requested
        """
        return self.forward(query, key, value, mask, return_logsumexp, training)
    
    def forward(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        return_logsumexp: bool = False,
        training: bool = False,
    ) -> Union[jnp.ndarray, FlashAttentionOutput]:
        """Forward pass computation.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len_k, head_dim]
            mask: Optional attention mask
            return_logsumexp: Whether to return logsumexp values
            training: Whether in training mode (for dropout)
            
        Returns:
            Attention output, with optional logsumexp if requested
        """
        # Cast to bfloat16 for TPU optimization if specified
        orig_dtype = query.dtype
        if self.use_bfloat16:
            query = query.astype(jnp.bfloat16)
            key = key.astype(jnp.bfloat16)
            value = value.astype(jnp.bfloat16)
            
        # Optimize memory layout for TPU
        query = optimize_tpu_layout(query)
        key = optimize_tpu_layout(key)
        value = optimize_tpu_layout(value)
            
        # Get dimensions
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Set softmax scaling factor if not provided
        if self.softmax_scale is None:
            softmax_scale = 1.0 / jnp.sqrt(float(head_dim))
        else:
            softmax_scale = self.softmax_scale
            
        # Block sizes
        block_q = min(self.block_size, seq_len_q)
        block_k = min(self.block_size, seq_len_k)
        
        # Initialize accumulators and output
        output = jnp.zeros_like(query)
        logsumexp = jnp.zeros((batch_size, num_heads, seq_len_q), dtype=jnp.float32)
        
        # RNG key for dropout if needed
        if training and self.dropout_rate > 0:
            dropout_rng = jax.random.PRNGKey(0)  # Should use a proper RNG key
        
        # Process blocks
        for q_start in range(0, seq_len_q, block_q):
            q_end = min(q_start + block_q, seq_len_q)
            
            # Get query block
            q_block = jax.lax.dynamic_slice(
                query,
                (0, 0, q_start, 0),
                (batch_size, num_heads, q_end - q_start, head_dim)
            )
            
            # Initialize block accumulators
            block_logsumexp = jnp.full((batch_size, num_heads, q_end - q_start), -jnp.inf, dtype=jnp.float32)
            block_output = jnp.zeros((batch_size, num_heads, q_end - q_start, head_dim), dtype=query.dtype)
            
            # Process key-value blocks
            for k_start in range(0, seq_len_k, block_k):
                k_end = min(k_start + block_k, seq_len_k)
                
                # Get key and value blocks
                k_block = jax.lax.dynamic_slice(
                    key,
                    (0, 0, k_start, 0),
                    (batch_size, num_heads, k_end - k_start, head_dim)
                )
                v_block = jax.lax.dynamic_slice(
                    value,
                    (0, 0, k_start, 0),
                    (batch_size, num_heads, k_end - k_start, head_dim)
                )
                
                # Compute attention scores for current blocks
                scores = jnp.einsum(
                    'bhqd,bhkd->bhqk', 
                    q_block, 
                    k_block, 
                    precision=self.precision
                )
                scores = scores * softmax_scale
                
                # Apply mask if provided
                if mask is not None:
                    # Extract relevant mask block
                    mask_block = jax.lax.dynamic_slice(
                        mask,
                        (0, 0, q_start, k_start) if mask.ndim == 4 else (0, q_start, k_start),
                        (batch_size, num_heads, q_end - q_start, k_end - k_start) 
                          if mask.ndim == 4 else
                          (batch_size, q_end - q_start, k_end - k_start)
                    )
                    if mask.ndim == 3:
                        # Expand to 4D if needed
                        mask_block = mask_block[:, None, :, :] if num_heads > 1 else mask_block[:, None, :, :]
                    
                    scores = jnp.where(mask_block, scores, -jnp.inf)
                
                # Apply causal mask if needed
                if self.causal and k_start < q_end:
                    causal_mask = jnp.greater_equal(
                        jnp.arange(q_start, q_end)[:, None],
                        jnp.arange(k_start, k_end)[None, :]
                    )
                    scores = jnp.where(
                        causal_mask[None, None, :, :], 
                        scores, 
                        -jnp.inf
                    )
                
                # Compute block max for numerical stability
                block_max = jnp.max(scores, axis=-1)
                
                # Update running logsumexp
                exp_scores = jnp.exp(scores - block_max[..., None])
                
                # Apply dropout during training
                if training and self.dropout_rate > 0:
                    dropout_mask = jax.random.bernoulli(
                        dropout_rng, 
                        p=1.0 - self.dropout_rate, 
                        shape=exp_scores.shape
                    )
                    exp_scores = exp_scores * dropout_mask / (1.0 - self.dropout_rate)
                
                # Compute sum for normalization
                exp_sum = jnp.sum(exp_scores, axis=-1)
                
                # Update block accumulators
                old_max = block_logsumexp
                new_max = jnp.maximum(block_max, old_max)
                
                # Update block output
                exp_weights = exp_scores / jnp.maximum(exp_sum, 1e-10)[..., None]
                weighted_value = jnp.einsum(
                    'bhqk,bhkd->bhqd',
                    exp_weights,
                    v_block,
                    precision=self.precision
                )
                
                # Combine with previous results
                old_scale = jnp.exp(old_max - new_max)[..., None]
                new_scale = jnp.exp(block_max - new_max)[..., None]
                block_output = old_scale * block_output + new_scale * weighted_value
                
                # Update block logsumexp
                block_logsumexp = new_max + jnp.log(
                    jnp.exp(old_max - new_max) * jnp.exp(block_logsumexp - old_max) +
                    jnp.exp(block_max - new_max) * exp_sum
                )
            
            # Update output with block results
            output = jax.lax.dynamic_update_slice(
                output,
                block_output,
                (0, 0, q_start, 0)
            )
            
            # Update logsumexp
            logsumexp = jax.lax.dynamic_update_slice(
                logsumexp,
                block_logsumexp,
                (0, 0, q_start)
            )
        
        # Cast back to original dtype if needed
        if self.use_bfloat16 and orig_dtype != jnp.bfloat16:
            output = output.astype(orig_dtype)
        
        if return_logsumexp:
            return FlashAttentionOutput(output, logsumexp)
        else:
            return output
    
    def backward(
        self,
        grad_output: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        logsumexp: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Backward pass computation for gradients.
        
        Args:
            grad_output: Gradient with respect to output
            query: Query tensor from forward pass
            key: Key tensor from forward pass
            value: Value tensor from forward pass
            logsumexp: Logsumexp values from forward pass
            mask: Attention mask used in forward pass
            
        Returns:
            Gradients for query, key, and value
        """
        # Cast to bfloat16 for TPU optimization if specified
        orig_dtype = grad_output.dtype
        if self.use_bfloat16:
            grad_output = grad_output.astype(jnp.bfloat16)
            query = query.astype(jnp.bfloat16)
            key = key.astype(jnp.bfloat16)
            value = value.astype(jnp.bfloat16)
            
        # Get dimensions
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Set softmax scaling factor if not provided
        if self.softmax_scale is None:
            softmax_scale = 1.0 / jnp.sqrt(float(head_dim))
        else:
            softmax_scale = self.softmax_scale
            
        # Block sizes
        block_q = min(self.block_size, seq_len_q)
        block_k = min(self.block_size, seq_len_k)
        
        # Initialize gradient accumulators
        grad_query = jnp.zeros_like(query)
        grad_key = jnp.zeros_like(key)
        grad_value = jnp.zeros_like(value)
        
        # If logsumexp not provided, recompute with forward pass
        if logsumexp is None:
            output_with_logsumexp = self.forward(
                query, key, value, mask, return_logsumexp=True
            )
            logsumexp = output_with_logsumexp.logsumexp
            
        # Process query blocks for backward pass
        for q_start in range(0, seq_len_q, block_q):
            q_end = min(q_start + block_q, seq_len_q)
            
            # Get query block and gradient block
            q_block = jax.lax.dynamic_slice(
                query,
                (0, 0, q_start, 0),
                (batch_size, num_heads, q_end - q_start, head_dim)
            )
            grad_output_block = jax.lax.dynamic_slice(
                grad_output,
                (0, 0, q_start, 0),
                (batch_size, num_heads, q_end - q_start, head_dim)
            )
            logsumexp_block = jax.lax.dynamic_slice(
                logsumexp,
                (0, 0, q_start),
                (batch_size, num_heads, q_end - q_start)
            )
            
            # Initialize gradient parts for key and value
            dkv_accum = jnp.zeros((batch_size, num_heads, seq_len_k, head_dim), dtype=query.dtype)
            
            # Process key-value blocks
            for k_start in range(0, seq_len_k, block_k):
                k_end = min(k_start + block_k, seq_len_k)
                
                # Get key and value blocks
                k_block = jax.lax.dynamic_slice(
                    key,
                    (0, 0, k_start, 0),
                    (batch_size, num_heads, k_end - k_start, head_dim)
                )
                v_block = jax.lax.dynamic_slice(
                    value,
                    (0, 0, k_start, 0),
                    (batch_size, num_heads, k_end - k_start, head_dim)
                )
                
                # Compute attention scores for current blocks
                scores = jnp.einsum(
                    'bhqd,bhkd->bhqk', 
                    q_block, 
                    k_block, 
                    precision=self.precision
                )
                scores = scores * softmax_scale
                
                # Apply mask if provided
                if mask is not None:
                    # Extract relevant mask block
                    mask_block = jax.lax.dynamic_slice(
                        mask,
                        (0, 0, q_start, k_start) if mask.ndim == 4 else (0, q_start, k_start),
                        (batch_size, num_heads, q_end - q_start, k_end - k_start) 
                          if mask.ndim == 4 else
                          (batch_size, q_end - q_start, k_end - k_start)
                    )
                    if mask.ndim == 3:
                        # Expand to 4D if needed
                        mask_block = mask_block[:, None, :, :] if num_heads > 1 else mask_block[:, None, :, :]
                    
                    scores = jnp.where(mask_block, scores, -jnp.inf)
                
                # Apply causal mask if needed
                if self.causal and k_start < q_end:
                    causal_mask = jnp.greater_equal(
                        jnp.arange(q_start, q_end)[:, None],
                        jnp.arange(k_start, k_end)[None, :]
                    )
                    scores = jnp.where(
                        causal_mask[None, None, :, :], 
                        scores, 
                        -jnp.inf
                    )
                
                # Compute attention weights
                exp_scores = jnp.exp(scores - logsumexp_block[..., None])
                
                # Gradients for value block
                grad_v_block = jnp.einsum(
                    'bhqd,bhqk->bhkd',
                    grad_output_block,
                    exp_scores,
                    precision=self.precision
                )
                
                # Gradients for attention weights
                grad_weights = jnp.einsum(
                    'bhqd,bhkd->bhqk',
                    grad_output_block,
                    v_block,
                    precision=self.precision
                )
                
                # Gradients through softmax
                grad_scores = (grad_weights - 
                               jnp.sum(exp_scores * grad_weights, axis=-1, keepdims=True))
                grad_scores = grad_scores * exp_scores
                
                # Apply scaling factor
                grad_scores = grad_scores * softmax_scale
                
                # Gradients for query and key blocks
                grad_q_block = jnp.einsum(
                    'bhqk,bhkd->bhqd',
                    grad_scores,
                    k_block,
                    precision=self.precision
                )
                grad_k_block = jnp.einsum(
                    'bhqk,bhqd->bhkd',
                    grad_scores,
                    q_block,
                    precision=self.precision
                )
                
                # Update query gradient
                grad_query = jax.lax.dynamic_update_slice(
                    grad_query,
                    grad_q_block,
                    (0, 0, q_start, 0)
                )
                
                # Accumulate key and value gradients
                grad_key_part = jax.lax.dynamic_slice(
                    grad_key,
                    (0, 0, k_start, 0),
                    (batch_size, num_heads, k_end - k_start, head_dim)
                )
                grad_key_part = grad_key_part + grad_k_block
                
                grad_value_part = jax.lax.dynamic_slice(
                    grad_value,
                    (0, 0, k_start, 0),
                    (batch_size, num_heads, k_end - k_start, head_dim)
                )
                grad_value_part = grad_value_part + grad_v_block
                
                # Update key and value gradients
                grad_key = jax.lax.dynamic_update_slice(
                    grad_key,
                    grad_key_part,
                    (0, 0, k_start, 0)
                )
                grad_value = jax.lax.dynamic_update_slice(
                    grad_value,
                    grad_value_part,
                    (0, 0, k_start, 0)
                )
        
        # Cast back to original dtype if needed
        if self.use_bfloat16 and orig_dtype != jnp.bfloat16:
            grad_query = grad_query.astype(orig_dtype)
            grad_key = grad_key.astype(orig_dtype)
            grad_value = grad_value.astype(orig_dtype)
        
        return grad_query, grad_key, grad_value