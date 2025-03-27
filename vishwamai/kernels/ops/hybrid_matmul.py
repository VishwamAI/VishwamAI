"""Hybrid matrix multiplication strategies for TPU/GPU."""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple

class HybridMatMul:
    """
    Hybrid matrix multiplication that adaptively switches between different
    computation strategies based on input size and hardware.
    """
    
    def __init__(
        self,
        block_size: int = 32,
        min_parallel_size: int = 512,
        use_tree: bool = True
    ):
        """
        Initialize hybrid matrix multiplication.
        
        Args:
            block_size: Size of blocks for blocked multiplication
            min_parallel_size: Minimum matrix size to use parallel strategy
            use_tree: Whether to use tree-based multiplication for large matrices
        """
        self.block_size = block_size
        self.min_parallel_size = min_parallel_size
        self.use_tree = use_tree
    
    def _blocked_matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Blocked matrix multiplication for better cache utilization."""
        m, k = a.shape
        k, n = b.shape
        
        # Pad matrices to block size
        pad_m = (self.block_size - m % self.block_size) % self.block_size
        pad_n = (self.block_size - n % self.block_size) % self.block_size
        pad_k = (self.block_size - k % self.block_size) % self.block_size
        
        a_pad = jnp.pad(a, ((0, pad_m), (0, pad_k)))
        b_pad = jnp.pad(b, ((0, pad_k), (0, pad_n)))
        
        # Reshape into blocks
        a_blocks = a_pad.reshape(-1, self.block_size, k // self.block_size, self.block_size)
        b_blocks = b_pad.reshape(k // self.block_size, self.block_size, -1, self.block_size)
        
        # Block matrix multiplication
        c_blocks = jnp.einsum('ibkj,kjnl->ibnl', a_blocks, b_blocks)
        
        # Reshape back and remove padding
        c = c_blocks.reshape(m + pad_m, n + pad_n)
        return c[:m, :n]
    
    def _parallel_matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Parallel matrix multiplication optimized for TPU/GPU."""
        return jax.vmap(lambda x: x @ b)(a)
    
    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False
    ) -> jnp.ndarray:
        """
        Perform hybrid matrix multiplication.
        
        Args:
            a: First input matrix
            b: Second input matrix
            transpose_a: Whether to transpose first matrix
            transpose_b: Whether to transpose second matrix
            
        Returns:
            Result of matrix multiplication
        """
        if transpose_a:
            a = jnp.transpose(a)
        if transpose_b:
            b = jnp.transpose(b)
            
        # Choose multiplication strategy based on size
        m = a.shape[0]
        if m >= self.min_parallel_size:
            return self._parallel_matmul(a, b)
        else:
            return self._blocked_matmul(a, b)