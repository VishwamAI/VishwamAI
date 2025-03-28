"""Sparse matrix operations optimized for TPU/GPU."""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, NamedTuple, Union
from functools import partial
from jax import lax

from vishwamai.kernels.tpu.tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

class BlockSparseConfig(NamedTuple):
    """Configuration for block-sparse operations."""
    block_size: int = 128  # TPU optimal block size
    min_sparsity: float = 0.8  # Minimum sparsity to use sparse implementation
    use_bfloat16: bool = True
    precision: Optional[lax.Precision] = None

class SparseOutput(NamedTuple):
    """Output from sparse operations."""
    output: jnp.ndarray
    attention_weights: Optional[jnp.ndarray] = None
    sparsity_mask: Optional[jnp.ndarray] = None

class SparseMatrixOps:
    """TPU-optimized sparse matrix operations."""
    
    def __init__(self, config: Optional[BlockSparseConfig] = None):
        """
        Initialize sparse matrix operations.
        
        Args:
            config: Optional configuration for block-sparse operations
        """
        self.config = config or BlockSparseConfig()
        
        # Validate block size
        if self.config.block_size % 128 != 0:
            raise ValueError("Block size must be multiple of 128 for TPU")

    @partial(jax.jit, static_argnums=(0,))
    def block_sparse_matmul(
        self,
        matrix: jnp.ndarray,
        sparse_values: jnp.ndarray,
        sparse_indices: jnp.ndarray,
        block_mask: jnp.ndarray,
        dense_shape: Optional[Tuple[int, int]] = None
    ) -> jnp.ndarray:
        """
        Block-sparse matrix multiplication optimized for TPU.
        
        Args:
            matrix: Dense input matrix [M, K]
            sparse_values: Sparse matrix values [nnz]
            sparse_indices: Indices for sparse values [nnz, 2]
            block_mask: Binary mask showing which blocks are non-zero [M//B, K//B]
            dense_shape: Optional shape of output dense matrix
            
        Returns:
            Result of sparse matrix multiplication
        """
        orig_dtype = matrix.dtype
        
        # Use bfloat16 for TPU optimization if specified
        if self.config.use_bfloat16:
            matrix = matrix.astype(jnp.bfloat16)
            sparse_values = sparse_values.astype(jnp.bfloat16)
            
        # Optimize memory layout
        matrix = optimize_tpu_layout(matrix)
        
        # Get dimensions
        M, K = matrix.shape
        if dense_shape is None:
            dense_shape = (M, K)
        N = dense_shape[1]
        
        # Compute block dimensions
        B = self.config.block_size
        MB = (M + B - 1) // B
        KB = (K + B - 1) // B
        NB = (N + B - 1) // B
        
        # Pad inputs to block size
        matrix = pad_to_tpu_multiple(matrix, B)
        
        def process_block(carry, block_idx):
            m_idx, k_idx = block_idx // KB, block_idx % KB
            
            # Extract relevant blocks
            if block_mask is not None and not block_mask[m_idx, k_idx]:
                return carry, None
                
            # Get matrix block
            m_start = m_idx * B
            k_start = k_idx * B
            matrix_block = jax.lax.dynamic_slice(
                matrix,
                (m_start, k_start),
                (B, B)
            )
            
            # Get sparse block indices and values
            block_mask = (
                (sparse_indices[:, 0] >= m_start) &
                (sparse_indices[:, 0] < m_start + B) &
                (sparse_indices[:, 1] >= k_start) &
                (sparse_indices[:, 1] < k_start + B)
            )
            block_indices = sparse_indices[block_mask]
            block_values = sparse_values[block_mask]
            
            # Adjust indices to block-local coordinates
            block_indices = block_indices - jnp.array([m_start, k_start])
            
            # Create block-sparse representation
            sparse_block = jnp.zeros((B, B), dtype=matrix.dtype)
            sparse_block = sparse_block.at[
                block_indices[:, 0],
                block_indices[:, 1]
            ].set(block_values)
            
            # Compute block multiplication
            result = jax.lax.dot_general(
                matrix_block,
                sparse_block,
                (((1,), (0,)), ((), ())),
                precision=self.config.precision
            )
            
            # Update output
            output = carry.at[m_start:m_start + B, k_start:k_start + B].add(result)
            
            return output, None
            
        # Initialize output
        output = jnp.zeros(dense_shape, dtype=matrix.dtype)
        
        # Process all blocks
        block_indices = jnp.arange(MB * KB)
        output, _ = jax.lax.scan(process_block, output, block_indices)
        
        # Cast back to original dtype if needed
        if self.config.use_bfloat16 and orig_dtype != jnp.bfloat16:
            output = output.astype(orig_dtype)
            
        return output

    @partial(jax.jit, static_argnums=(0,))
    def block_sparse_attention(
        self,
        queries: jnp.ndarray,
        keys: jnp.ndarray,
        values: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        dropout_rate: float = 0.0,
        training: bool = True,
        block_size: Optional[int] = None
    ) -> SparseOutput:
        """
        Block-sparse attention implementation optimized for TPU.
        
        Args:
            queries: Query vectors [batch, heads, seq_len, dim]
            keys: Key vectors [batch, heads, seq_len, dim]
            values: Value vectors [batch, heads, seq_len, dim]
            mask: Optional attention mask [batch, heads, seq_len, seq_len]
            dropout_rate: Dropout probability
            training: Whether in training mode
            block_size: Optional block size override
            
        Returns:
            SparseOutput containing attention output and optional weights/mask
        """
        orig_dtype = queries.dtype
        block_size = block_size or self.config.block_size
        
        # Use bfloat16 for TPU optimization if specified
        if self.config.use_bfloat16:
            queries = queries.astype(jnp.bfloat16)
            keys = keys.astype(jnp.bfloat16)
            values = values.astype(jnp.bfloat16)
            
        # Optimize memory layout
        queries = optimize_tpu_layout(queries)
        keys = optimize_tpu_layout(keys)
        values = optimize_tpu_layout(values)
        
        # Get dimensions
        batch_size, num_heads, seq_len_q, dim = queries.shape
        _, _, seq_len_k, _ = keys.shape
        
        # Compute attention scores
        scale = jnp.sqrt(dim).astype(queries.dtype)
        scores = jnp.einsum(
            'bhid,bhjd->bhij',
            queries,
            keys,
            precision=self.config.precision
        ) / scale

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
            
        # Compute block-sparse attention pattern
        block_size_q = min(block_size, seq_len_q)
        block_size_k = min(block_size, seq_len_k)
        
        # Create sparsity mask based on top-k scores per query
        k = max(1, int(seq_len_k * (1 - self.config.min_sparsity)))
        top_k_scores = jax.lax.top_k(scores, k)[0][..., -1:]
        sparsity_mask = scores >= top_k_scores
        
        # Ensure block alignment
        num_blocks_q = (seq_len_q + block_size_q - 1) // block_size_q
        num_blocks_k = (seq_len_k + block_size_k - 1) // block_size_k
        
        # Create block mask
        block_mask = jnp.zeros((batch_size, num_heads, num_blocks_q, num_blocks_k))
        
        def update_block_mask(block_idx):
            q_idx, k_idx = block_idx // num_blocks_k, block_idx % num_blocks_k
            q_start = q_idx * block_size_q
            k_start = k_idx * block_size_k
            q_end = min(q_start + block_size_q, seq_len_q)
            k_end = min(k_start + block_size_k, seq_len_k)
            
            block_scores = scores[
                :, :,
                q_start:q_end,
                k_start:k_end
            ]
            return jnp.any(sparsity_mask[
                :, :,
                q_start:q_end,
                k_start:k_end
            ])
            
        block_mask = jax.vmap(update_block_mask)(
            jnp.arange(num_blocks_q * num_blocks_k)
        ).reshape(num_blocks_q, num_blocks_k)
        
        # Apply block-sparse attention
        def process_block(carry, block_idx):
            q_idx, k_idx = block_idx // num_blocks_k, block_idx % num_blocks_k
            
            if not block_mask[q_idx, k_idx]:
                return carry, None
                
            # Get query block
            q_start = q_idx * block_size_q
            q_end = min(q_start + block_size_q, seq_len_q)
            query_block = jax.lax.dynamic_slice(
                queries,
                (0, 0, q_start, 0),
                (batch_size, num_heads, q_end - q_start, dim)
            )
            
            # Get key and value blocks
            k_start = k_idx * block_size_k
            k_end = min(k_start + block_size_k, seq_len_k)
            key_block = jax.lax.dynamic_slice(
                keys,
                (0, 0, k_start, 0),
                (batch_size, num_heads, k_end - k_start, dim)
            )
            value_block = jax.lax.dynamic_slice(
                values,
                (0, 0, k_start, 0),
                (batch_size, num_heads, k_end - k_start, dim)
            )
            
            # Compute attention for this block
            block_scores = scores[
                :, :,
                q_start:q_end,
                k_start:k_end
            ]
            
            # Apply softmax
            block_weights = jax.nn.softmax(block_scores, axis=-1)
            
            if training and dropout_rate > 0:
                keep_prob = 1.0 - dropout_rate
                dropout_mask = jax.random.bernoulli(
                    jax.random.PRNGKey(0),
                    p=keep_prob,
                    shape=block_weights.shape
                )
                block_weights = block_weights * dropout_mask / keep_prob
            
            # Apply attention to values
            block_output = jnp.einsum(
                'bhqk,bhkd->bhqd',
                block_weights,
                value_block,
                precision=self.config.precision
            )
            
            # Update output
            output = carry.at[:, :, q_start:q_end].add(block_output)
            return output, None
            
        # Initialize output
        output = jnp.zeros(
            (batch_size, num_heads, seq_len_q, dim),
            dtype=queries.dtype
        )
        
        # Process all blocks
        block_indices = jnp.arange(num_blocks_q * num_blocks_k)
        output, _ = jax.lax.scan(process_block, output, block_indices)
        
        # Cast back to original dtype if needed
        if self.config.use_bfloat16 and orig_dtype != jnp.bfloat16:
            output = output.astype(orig_dtype)
            
        return SparseOutput(output, attention_weights=None, sparsity_mask=sparsity_mask)

sparse_ops = SparseMatrixOps()  # Create singleton instance
