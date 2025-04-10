"""Common neural network operations."""

from .activation import (
    gelu_approx,
    silu_optimized,
    quick_gelu,
    hard_silu,
    leaky_relu_optimized
)

from .hybrid_matmul import HybridMatMul
from .tree_matmul import TreeMatMul, create_adaptive_depth_mask
from .sparse import SparseMatrixOps
from .eplib import efficient_parallel_ops

__all__ = [
    # Activation functions
    "gelu_approx",
    "silu_optimized",
    "quick_gelu",
    "hard_silu",
    "leaky_relu_optimized",
    
    # Matrix operations
    "HybridMatMul",
    "TreeMatMul",
    "create_adaptive_depth_mask",
    
    # Sparse matrix operations
    "SparseMatrixOps",
    
    # Efficient parallel operations
    "efficient_parallel_ops"
]