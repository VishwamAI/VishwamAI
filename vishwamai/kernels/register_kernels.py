"""Register optimized kernels with VishwamAI's kernel manager."""

from .jit_manager import register_kernel, KernelPlatform
from .flashmla_cuda import (
    jax_get_mla_metadata,
    jax_flash_mla_with_kvcache,
    flash_mla_with_kvcache,
    get_mla_metadata
)
from .sparse import (
    sparse_block_gemm,
    block_sparse_attention,
)
from .tree_matmul import TreeMatMul
from .hybrid_matmul import HybridMatMul

def register_optimized_kernels():
    """Register all optimized kernels with the kernel manager."""
    
    # Register FlashMLA kernels
    register_kernel(
        "flash_mla_metadata",
        KernelPlatform.GPU,
        get_mla_metadata
    )
    register_kernel(
        "flash_mla_metadata",
        KernelPlatform.TPU,
        jax_get_mla_metadata
    )
    
    register_kernel(
        "flash_mla",
        KernelPlatform.GPU,
        flash_mla_with_kvcache
    )
    register_kernel(
        "flash_mla",
        KernelPlatform.TPU,
        jax_flash_mla_with_kvcache
    )
    
    # Register sparse computation kernels
    register_kernel(
        "sparse_block_gemm",
        KernelPlatform.TPU,
        sparse_block_gemm
    )
    register_kernel(
        "block_sparse_attention",
        KernelPlatform.TPU,
        block_sparse_attention
    )
    
    # Register tree matmul kernel
    register_kernel(
        "tree_matmul",
        KernelPlatform.TPU,
        TreeMatMul
    )
    
    # Register hybrid matmul kernel
    register_kernel(
        "hybrid_matmul",
        KernelPlatform.TPU,
        HybridMatMul
    )