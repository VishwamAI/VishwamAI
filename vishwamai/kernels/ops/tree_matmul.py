"""Tree-based matrix multiplication for hierarchical computations."""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple

def create_adaptive_depth_mask(
    size: int,
    max_depth: int = 8,
    threshold: float = 0.1
) -> jnp.ndarray:
    """
    Create an adaptive depth mask for tree-based operations.
    
    Args:
        size: Size of the input matrix dimension
        max_depth: Maximum tree depth
        threshold: Threshold for adaptive pruning
        
    Returns:
        Boolean mask for tree depth adaptation
    """
    depths = jnp.arange(max_depth)
    importance = 1.0 / (2 ** depths)
    mask = importance > threshold
    return jnp.resize(mask, (size,))

class TreeMatMul:
    """
    Tree-based matrix multiplication for hierarchical computation.
    Splits large matrices into smaller blocks and computes results
    hierarchically for better parallelization.
    """
    
    def __init__(
        self,
        leaf_size: int = 32,
        max_depth: int = 8,
        adaptive: bool = True
    ):
        """
        Initialize tree matrix multiplication.
        
        Args:
            leaf_size: Size of leaf matrices
            max_depth: Maximum depth of computation tree
            adaptive: Whether to use adaptive depth masking
        """
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.adaptive = adaptive
    
    def _split_matrix(
        self,
        matrix: jnp.ndarray,
        axis: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Split matrix along specified axis."""
        split_size = matrix.shape[axis] // 2
        if axis == 0:
            return matrix[:split_size], matrix[split_size:]
        else:
            return matrix[:, :split_size], matrix[:, split_size:]
    
    def _tree_matmul_recursive(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        depth: int = 0
    ) -> jnp.ndarray:
        """Recursive tree-based matrix multiplication."""
        m, k = a.shape
        k, n = b.shape
        
        # Base case: small enough for direct multiplication
        if m <= self.leaf_size or k <= self.leaf_size or n <= self.leaf_size:
            return a @ b
            
        # Split matrices recursively
        if depth >= self.max_depth:
            return a @ b
            
        # Decide splitting dimension based on largest dimension
        if m >= k and m >= n:
            a1, a2 = self._split_matrix(a, 0)
            c1 = self._tree_matmul_recursive(a1, b, depth + 1)
            c2 = self._tree_matmul_recursive(a2, b, depth + 1)
            return jnp.vstack([c1, c2])
        elif k >= m and k >= n:
            a1, a2 = self._split_matrix(a, 1)
            b1, b2 = self._split_matrix(b, 0)
            c1 = self._tree_matmul_recursive(a1, b1, depth + 1)
            c2 = self._tree_matmul_recursive(a2, b2, depth + 1)
            return c1 + c2
        else:
            b1, b2 = self._split_matrix(b, 1)
            c1 = self._tree_matmul_recursive(a, b1, depth + 1)
            c2 = self._tree_matmul_recursive(a, b2, depth + 1)
            return jnp.hstack([c1, c2])
    
    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Perform tree-based matrix multiplication.
        
        Args:
            a: First input matrix
            b: Second input matrix
            mask: Optional depth adaptation mask
            
        Returns:
            Result of matrix multiplication
        """
        if self.adaptive and mask is None:
            mask = create_adaptive_depth_mask(
                max(a.shape + b.shape),
                self.max_depth
            )
            
        return self._tree_matmul_recursive(a, b)