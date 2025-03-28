"""Efficient parallel operations library optimized for TPU/GPU."""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List, Union, Protocol, Any
from functools import partial
from jax import lax

from vishwamai.kernels.tpu.tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

class ReductionOp(Protocol):
    """Protocol for reduction operations."""
    def __call__(self, x: jnp.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                 keepdims: bool = False) -> jnp.ndarray: ...

class EfficientParallelOps:
    """Collection of efficient parallel operations optimized for TPU/GPU."""
    
    def __init__(
        self,
        use_tpu_optimizations: bool = True,
        block_size: int = 128,
        use_bfloat16: bool = True,
        precision: Optional[lax.Precision] = None
    ):
        """
        Initialize parallel operations.
        
        Args:
            use_tpu_optimizations: Whether to use TPU-specific optimizations
            block_size: Size of blocks for TPU operations (multiple of 128)
            use_bfloat16: Whether to use bfloat16 precision on TPU
            precision: JAX precision setting for computation
        """
        if block_size % 128 != 0:
            raise ValueError("Block size must be a multiple of 128 for TPU")
            
        self.use_tpu_optimizations = use_tpu_optimizations
        self.block_size = block_size
        self.use_bfloat16 = use_bfloat16
        self.precision = precision or lax.Precision.HIGHEST

    @partial(jax.jit, static_argnums=(0,2))
    def batch_matmul(
        self,
        matrices_a: jnp.ndarray,
        matrices_b: jnp.ndarray,
        chunk_size: int = 32,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> jnp.ndarray:
        """
        Efficient batched matrix multiplication optimized for TPU.
        
        Args:
            matrices_a: First batch of matrices [batch, M, K]
            matrices_b: Second batch of matrices [batch, K, N]
            chunk_size: Size of chunks for parallel processing
            transpose_a: Whether to transpose first matrices
            transpose_b: Whether to transpose second matrices
            
        Returns:
            Batched matrix multiplication result [batch, M, N]
        """
        if self.use_tpu_optimizations:
            # Cast to bfloat16 if specified
            if self.use_bfloat16:
                matrices_a = matrices_a.astype(jnp.bfloat16)
                matrices_b = matrices_b.astype(jnp.bfloat16)
                
            # Optimize memory layout
            matrices_a = optimize_tpu_layout(matrices_a)
            matrices_b = optimize_tpu_layout(matrices_b)
            
            # Handle transposes
            if transpose_a:
                matrices_a = jnp.transpose(matrices_a, (0, 2, 1))
            if transpose_b:
                matrices_b = jnp.transpose(matrices_b, (0, 2, 1))
                
            # Use dot_general for TPU-optimized multiplication
            result = jax.lax.batch_dot_general(
                matrices_a,
                matrices_b,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                precision=self.precision
            )
        else:
            # Standard vmap implementation
            result = jax.vmap(lambda a, b: jnp.matmul(
                jnp.transpose(a) if transpose_a else a,
                jnp.transpose(b) if transpose_b else b
            ))(matrices_a, matrices_b)
            
        return result

    @partial(jax.jit, static_argnums=(0,1,2))
    def parallel_scan(
        self,
        sequence: jnp.ndarray,
        chunk_size: int = 1024,
        reverse: bool = False,
        axis: int = 0
    ) -> jnp.ndarray:
        """
        Parallel scan (prefix sum) implementation optimized for TPU.
        
        Args:
            sequence: Input sequence
            chunk_size: Size of chunks for parallel processing
            reverse: Whether to do reverse cumulative sum
            axis: Axis along which to perform scan
            
        Returns:
            Cumulative sum array
        """
        if self.use_tpu_optimizations and self.use_bfloat16:
            sequence = sequence.astype(jnp.bfloat16)
            
        # Reshape input for parallel processing
        orig_shape = sequence.shape
        if axis != 0:
            sequence = jnp.moveaxis(sequence, axis, 0)
            
        # Split into chunks
        chunks = jnp.array_split(sequence, max(1, len(sequence) // chunk_size))
        
        # Compute chunk sums
        chunk_sums = jnp.array([chunk.sum(axis=0) for chunk in chunks])
        prefix_sums = jnp.cumsum(chunk_sums, axis=0) if not reverse else jnp.cumsum(chunk_sums[::-1], axis=0)[::-1]
        
        # Compute parallel scan within chunks
        def scan_chunk(chunk: jnp.ndarray, offset: jnp.ndarray) -> jnp.ndarray:
            chunk_result = jnp.cumsum(chunk, axis=0) if not reverse else jnp.cumsum(chunk[::-1], axis=0)[::-1]
            return chunk_result + offset
            
        # Process chunks in parallel
        results = []
        offset = jnp.zeros_like(prefix_sums[0])
        for i, chunk in enumerate(chunks):
            if i > 0:
                offset = prefix_sums[i-1]
            results.append(scan_chunk(chunk, offset))
            
        result = jnp.concatenate(results)
        
        # Restore original shape and dtype
        if axis != 0:
            result = jnp.moveaxis(result, 0, axis)
            
        if self.use_tpu_optimizations and self.use_bfloat16:
            result = result.astype(sequence.dtype)
            
        return result

    @partial(jax.jit, static_argnums=(0,))
    def parallel_sort(
        self,
        values: jnp.ndarray,
        keys: Optional[jnp.ndarray] = None,
        axis: int = -1,
        stable: bool = True
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Parallel sorting implementation optimized for TPU.
        
        Args:
            values: Values to sort
            keys: Optional keys to sort by
            axis: Axis along which to sort
            stable: Whether to use stable sorting
            
        Returns:
            Sorted array or tuple of (sorted_values, sorted_keys)
        """
        if keys is None:
            if self.use_tpu_optimizations and self.use_bfloat16:
                values = values.astype(jnp.bfloat16)
            result = jax.lax.sort(values, dimension=axis)
            if self.use_tpu_optimizations and self.use_bfloat16:
                result = result.astype(values.dtype)
            return result
        else:
            if self.use_tpu_optimizations and self.use_bfloat16:
                values = values.astype(jnp.bfloat16)
                keys = keys.astype(jnp.bfloat16)
            indices = jax.lax.sort(keys, dimension=axis)
            sorted_values = jnp.take_along_axis(values, indices, axis=axis)
            sorted_keys = jnp.take_along_axis(keys, indices, axis=axis)
            if self.use_tpu_optimizations and self.use_bfloat16:
                sorted_values = sorted_values.astype(values.dtype)
                sorted_keys = sorted_keys.astype(keys.dtype)
            return sorted_values, sorted_keys

    @partial(jax.jit, static_argnums=(0,1,2))
    def strided_reduction(
        self,
        array: jnp.ndarray,
        reduction_size: int,
        op: str = 'sum'
    ) -> jnp.ndarray:
        """
        Parallel strided reduction optimized for TPU.
        
        Args:
            array: Input array
            reduction_size: Size of reduction window
            op: Reduction operation ('sum', 'max', 'min', 'mean')
            
        Returns:
            Reduced array
        """
        ops: dict[str, ReductionOp] = {
            'sum': jnp.sum,
            'max': jnp.max,
            'min': jnp.min,
            'mean': jnp.mean
        }
        
        if op not in ops:
            raise ValueError(f"Unsupported reduction op: {op}")
            
        if self.use_tpu_optimizations:
            # Pad to block size multiple for TPU efficiency
            pad_size = (reduction_size + self.block_size - 1) // self.block_size * self.block_size
            if pad_size > reduction_size:
                # Pad and adjust for reduction
                if op in ('sum', 'mean'):
                    array = jnp.pad(array, ((0, pad_size - reduction_size),))
                else:
                    # For max/min, pad with identity element
                    pad_value = jnp.finfo(array.dtype).min if op == 'max' else jnp.finfo(array.dtype).max
                    array = jnp.pad(array, ((0, pad_size - reduction_size),), 
                                  constant_values=pad_value)
                                  
            if self.use_bfloat16:
                array = array.astype(jnp.bfloat16)
                
        # Reshape for reduction
        reduce_fn = ops[op]
        result = reduce_fn(array.reshape(-1, reduction_size), axis=1)
        
        if self.use_tpu_optimizations and self.use_bfloat16:
            result = result.astype(array.dtype)
            
        return result

# Create singleton instance with default settings
efficient_parallel_ops = EfficientParallelOps()
