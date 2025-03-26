"""TPU-optimized Attention kernel implementation."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Tuple, Union, Dict, Any, NamedTuple
import numpy as np
from functools import partial

from vishwamai.kernels.core.kernel import Kernel, KernelConfig
from vishwamai.kernels.core.kernel_manager import HardwareType
from .tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

class AttentionOutput(NamedTuple):
    """Output from attention operation."""
    output: jnp.ndarray
    attention_weights: Optional[jnp.ndarray] = None

class TPUAttentionKernel:
    """TPU-optimized attention kernel.
    
    Provides efficient attention mechanism operations optimized for TPU hardware
    with support for various attention patterns like causal, block-sparse, and flash.
    """
    
    def __init__(
        self,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        block_size: int = 128,
        precision: Optional[lax.Precision] = None,
        use_flash: bool = True,
        use_block_sparse: bool = False,
        dropout_rate: float = 0.0,
        causal: bool = False,
        use_bfloat16: bool = True,
    ):
        """Initialize TPU attention kernel.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            block_size: Size of blocks for tiling
            precision: JAX precision setting for computation
            use_flash: Whether to use flash attention algorithm
            use_block_sparse: Whether to use block-sparse attention
            dropout_rate: Rate for attention dropout
            causal: Whether to use causal attention mask
            use_bfloat16: Whether to use bfloat16 for computations
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.precision = precision or lax.Precision.HIGHEST
        self.use_flash = use_flash
        self.use_block_sparse = use_block_sparse
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.use_bfloat16 = use_bfloat16
    
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        return_attention: bool = False,
        training: bool = False,
        key_padding_mask: Optional[jnp.ndarray] = None,
    ) -> Union[jnp.ndarray, AttentionOutput]:
        """Perform attention computation optimized for TPU.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len_k, head_dim]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            training: Whether in training mode (for dropout)
            key_padding_mask: Optional padding mask for keys
            
        Returns:
            Attention output, with optional attention weights if requested
        """
        return self.forward(
            query, key, value, mask, return_attention, training, key_padding_mask
        )
        
    def forward(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        return_attention: bool = False,
        training: bool = False,
        key_padding_mask: Optional[jnp.ndarray] = None,
    ) -> Union[jnp.ndarray, AttentionOutput]:
        """Forward pass computation.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len_k, head_dim]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            training: Whether in training mode (for dropout)
            key_padding_mask: Optional padding mask for keys
            
        Returns:
            Attention output, with optional attention weights if requested
        """
        # Cast to bfloat16 if specified
        if self.use_bfloat16:
            query = query.astype(jnp.bfloat16)
            key = key.astype(jnp.bfloat16)
            value = value.astype(jnp.bfloat16)
            
        # Optimize memory layout for TPU
        query = optimize_tpu_layout(query)
        key = optimize_tpu_layout(key)
        value = optimize_tpu_layout(value)
            
        # Get query dimensions
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Create causal mask if needed
        if self.causal and mask is None:
            mask = jnp.triu(
                jnp.ones((seq_len_q, seq_len_k), dtype=jnp.bool_),
                k=1
            )
            mask = jnp.logical_not(mask)[None, None, :, :]
            
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            if key_padding_mask.ndim == 2:  # [batch, seq_len_k]
                key_padding_mask = key_padding_mask[:, None, None, :]
            elif key_padding_mask.ndim == 3:  # [batch, 1, seq_len_k]
                key_padding_mask = key_padding_mask[:, None, :, :]
                
            # Combine with existing mask or create new mask
            if mask is not None:
                mask = jnp.logical_and(mask, key_padding_mask)
            else:
                mask = key_padding_mask
                
        # Choose attention implementation based on settings
        if self.use_flash:
            output, attention_weights = self._flash_attention(
                query, key, value, mask, training
            )
        elif self.use_block_sparse:
            output, attention_weights = self._block_sparse_attention(
                query, key, value, mask, training
            )
        else:
            output, attention_weights = self._standard_attention(
                query, key, value, mask, training
            )
        
        # Return appropriate output format
        if return_attention:
            return AttentionOutput(output, attention_weights)
        else:
            return output
    
    def _standard_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        training: bool,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Standard scaled dot-product attention optimized for TPU.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Attention output and attention weights
        """
        # Calculate attention scores
        scores = jnp.einsum('bhqd,bhkd->bhqk', query, key, precision=self.precision)
        scores = scores / jnp.sqrt(query.shape[-1]).astype(scores.dtype)
        
        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
        
        # Apply softmax
        weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention dropout during training
        if training and self.dropout_rate > 0:
            dropout_rng = jax.random.PRNGKey(0)  # Should use a proper RNG key in practice
            keep_prob = 1.0 - self.dropout_rate
            dropout_mask = jax.random.bernoulli(
                dropout_rng, 
                p=keep_prob, 
                shape=weights.shape
            )
            weights = weights * dropout_mask / keep_prob
        
        # Apply attention weights to values
        output = jnp.einsum('bhqk,bhkd->bhqd', weights, value, precision=self.precision)
        
        return output, weights
    
    def _flash_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        training: bool,
    ) -> Tuple[jnp.ndarray, None]:
        """Flash attention implementation with O(1) memory complexity.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Attention output (attention weights not computed to save memory)
        """
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Scaling factor
        scale = 1.0 / jnp.sqrt(head_dim)
        
        # Initialize accumulators
        output = jnp.zeros_like(query)
        l = jnp.ones((batch_size, num_heads, seq_len_q, 1)) * -jnp.inf
        m = jnp.ones((batch_size, num_heads, seq_len_q, 1)) * -jnp.inf
        
        # Determine block size for TPU
        block_size = min(self.block_size, seq_len_k)
        
        # Process blocks of keys and values
        def process_block(i, carry):
            output, l, m = carry
            
            block_start = i * block_size
            block_end = min(block_start + block_size, seq_len_k)
            
            # Get current blocks
            k_block = jax.lax.dynamic_slice(
                key,
                (0, 0, block_start, 0),
                (batch_size, num_heads, block_end - block_start, head_dim)
            )
            v_block = jax.lax.dynamic_slice(
                value,
                (0, 0, block_start, 0),
                (batch_size, num_heads, block_end - block_start, head_dim)
            )
            
            # Compute attention scores for current block
            s_block = jnp.einsum('bhqd,bhkd->bhqk', query, k_block) * scale
            
            # Apply mask if provided
            if mask is not None:
                mask_block = jax.lax.dynamic_slice(
                    mask,
                    (0, 0, 0, block_start),
                    (batch_size if mask.shape[0] > 1 else 1, 
                     1 if mask.shape[1] == 1 else num_heads,
                     seq_len_q, 
                     block_end - block_start)
                )
                s_block = jnp.where(mask_block, s_block, -jnp.inf)
            
            # Apply causal mask if needed
            if self.causal and block_start < seq_len_q:
                causal_mask = jnp.triu(
                    jnp.ones((seq_len_q, block_end - block_start), dtype=jnp.bool_),
                    k=1 + block_start
                )
                s_block = jnp.where(jnp.logical_not(causal_mask)[None, None, :, :], 
                                   s_block, 
                                   -jnp.inf)
            
            # Update running maximum
            m_block = jnp.max(s_block, axis=-1, keepdims=True)
            m_new = jnp.maximum(m, m_block)
            
            # Compute attention weights for current block
            p = jnp.exp(s_block - m_new)
            
            # Apply dropout during training
            if training and self.dropout_rate > 0:
                dropout_rng = jax.random.PRNGKey(0)  # Should use a proper RNG key
                keep_prob = 1.0 - self.dropout_rate
                dropout_mask = jax.random.bernoulli(
                    dropout_rng, 
                    p=keep_prob, 
                    shape=p.shape
                )
                p = p * dropout_mask / keep_prob
            
            # Update normalization factor
            l_new = l * jnp.exp(m - m_new) + jnp.sum(p, axis=-1, keepdims=True)
            
            # Update output
            output_block = jnp.einsum('bhqk,bhkd->bhqd', p, v_block)
            output_new = output * jnp.exp(m - m_new) + output_block
            
            return output_new, l_new, m_new
            
        # Process all blocks
        num_blocks = (seq_len_k + block_size - 1) // block_size
        
        if num_blocks > 1:
            # Use lax.scan for better TPU performance with multiple blocks
            init_carry = (output, l, m)
            output, l, m = lax.scan(
                process_block,
                init_carry,
                jnp.arange(num_blocks)
            )[1]
        else:
            # Just process a single block directly
            output, l, m = process_block(0, (output, l, m))
        
        # Normalize output
        output = output / l
        
        return output, None
    
    def _block_sparse_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        training: bool,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Block-sparse attention optimized for TPU.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Attention output and attention weights
        """
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Block sizes for sparse attention
        block_q = min(64, seq_len_q)
        block_k = min(64, seq_len_k)
        
        # Number of blocks
        num_blocks_q = (seq_len_q + block_q - 1) // block_q
        num_blocks_k = (seq_len_k + block_k - 1) // block_k
        
        # Initialize output
        output = jnp.zeros_like(query)
        
        # Process blocks for better TPU utilization
        def process_block_pair(q_idx, k_idx, current_output):
            # Get query block
            q_start = q_idx * block_q
            q_end = min(q_start + block_q, seq_len_q)
            q_block = jax.lax.dynamic_slice(
                query,
                (0, 0, q_start, 0),
                (batch_size, num_heads, q_end - q_start, head_dim)
            )
            
            # Get key and value blocks
            k_start = k_idx * block_k
            k_end = min(k_start + block_k, seq_len_k)
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
            
            # Compute attention scores for this block pair
            scores = jnp.einsum('bhqd,bhkd->bhqk', q_block, k_block, precision=self.precision)
            scores = scores / jnp.sqrt(head_dim)
            
            # Apply mask if provided
            if mask is not None:
                mask_block = jax.lax.dynamic_slice(
                    mask,
                    (0, 0, q_start, k_start),
                    (batch_size if mask.shape[0] > 1 else 1, 
                     1 if mask.shape[1] == 1 else num_heads,
                     q_end - q_start, 
                     k_end - k_start)
                )
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
            
            # Apply softmax
            weights = jax.nn.softmax(scores, axis=-1)
            
            # Apply dropout during training
            if training and self.dropout_rate > 0:
                dropout_rng = jax.random.PRNGKey(0)
                keep_prob = 1.0 - self.dropout_rate
                dropout_mask = jax.random.bernoulli(
                    dropout_rng, 
                    p=keep_prob, 
                    shape=weights.shape
                )
                weights = weights * dropout_mask / keep_prob
            
            # Apply attention weights to values
            block_output = jnp.einsum(
                'bhqk,bhkd->bhqd', 
                weights, 
                v_block, 
                precision=self.precision
            )
            
            # Update the corresponding part of the output
            updated_output = jax.lax.dynamic_update_slice(
                current_output,
                block_output,
                (0, 0, q_start, 0)
            )
            
            return updated_output
        
        # Process all block pairs
        for q_idx in range(num_blocks_q):
            for k_idx in range(num_blocks_k):
                output = process_block_pair(q_idx, k_idx, output)
        
        return output, None  # Don't return attention weights to save memory

    def backward(
        self,
        grad_output: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        attention_weights: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Backward pass computation for gradients.
        
        Args:
            grad_output: Gradient with respect to output [batch, heads, seq_q, dim]
            query: Query tensor from forward pass
            key: Key tensor from forward pass
            value: Value tensor from forward pass
            attention_weights: Attention weights from forward (if available)
            mask: Attention mask used in forward pass
            
        Returns:
            Gradients for query, key, and value
        """
        # If using flash attention, we need to recompute attention weights
        if attention_weights is None or self.use_flash:
            # Compute attention scores
            scores = jnp.einsum('bhqd,bhkd->bhqk', query, key, precision=self.precision)
            scores = scores / jnp.sqrt(query.shape[-1])
            
            # Apply mask if provided
            if mask is not None:
                scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
                
            # Apply softmax to get attention weights
            attention_weights = jax.nn.softmax(scores, axis=-1)
        
        # Gradients with respect to value
        grad_value = jnp.einsum(
            'bhqd,bhqk->bhkd', 
            grad_output, 
            attention_weights, 
            precision=self.precision
        )
        
        # Gradient with respect to attention weights
        grad_weights = jnp.einsum(
            'bhqd,bhkd->bhqk', 
            grad_output, 
            value, 
            precision=self.precision
        )
        
        # Gradient through softmax
        grad_scores = attention_weights * (grad_weights - 
                                           jnp.sum(attention_weights * grad_weights, 
                                                 axis=-1, keepdims=True))
        
        # Apply mask gradient if needed
        if mask is not None:
            grad_scores = jnp.where(mask, grad_scores, 0)
        
        # Scale by sqrt(dim)
        grad_scores = grad_scores / jnp.sqrt(query.shape[-1])
        
        # Gradients with respect to query and key
        grad_query = jnp.einsum(
            'bhqk,bhkd->bhqd', 
            grad_scores, 
            key, 
            precision=self.precision
        )
        grad_key = jnp.einsum(
            'bhqk,bhqd->bhkd', 
            grad_scores, 
            query, 
            precision=self.precision
        )
        
        return grad_query, grad_key, grad_value