"""Register optimized kernels with VishwamAI's kernel manager."""

from typing import Dict, Any, Optional
from enum import Enum, auto

from vishwamai.kernels.jit.jit_manager import register_kernel, KernelPlatform
from vishwamai.kernels.cuda.flashmla_cuda import (
    jax_get_mla_metadata,
    jax_flash_mla_with_kvcache,
    flash_mla_with_kvcache,
    get_mla_metadata
)
from vishwamai.kernels.core.kernel import (
    KernelConfig,
    multi_head_attention_kernel,
    flash_attention,
    sliding_window_attention,
    rope_embedding
)
from vishwamai.kernels.ops.sparse import (
    sparse_block_gemm,
    block_sparse_attention,
    SparseMatrixOps
)
from vishwamai.kernels.ops.tree_matmul import TreeMatMul
from vishwamai.kernels.ops.hybrid_matmul import HybridMatMul
from vishwamai.kernels.ops.eplib import EfficientParallelOps
from vishwamai.kernels.optimizers.moe_balance import (
    rebalance_experts,
    replicate_experts,
    balanced_packing
)

class KernelGroup(Enum):
    """Groups of related kernels for organization."""
    ATTENTION = auto()
    MATMUL = auto()
    SPARSE = auto()
    EXPERT = auto()
    PARALLEL = auto()
    EMBEDDING = auto()

def get_default_config(platform: KernelPlatform) -> Dict[str, Any]:
    """Get default configuration for a platform."""
    if platform == KernelPlatform.TPU:
        return {
            "block_size": 128,  # TPU optimal block size
            "use_bfloat16": True,
            "use_fp8": False,  # Enable when hardware supports it
            "use_efficient_attention": True,
            "min_sequence_length_for_flash": 1024,
            "sharding_mode": "2d"  # Default to 2D sharding
        }
    else:  # GPU
        return {
            "block_size": 64,  # GPU optimal block size
            "use_fp16": True,
            "use_flash_attention": True,
            "max_sequence_length": None,
            "sharding_mode": "1d"  # Default to 1D for GPU
        }

def register_attention_kernels(platform: KernelPlatform, config: Optional[Dict[str, Any]] = None):
    """Register attention-related kernels."""
    cfg = {**get_default_config(platform), **(config or {})}
    
    # Standard attention kernels
    register_kernel(
        "multi_head_attention",
        platform,
        multi_head_attention_kernel,
        config=cfg
    )
    
    register_kernel(
        "flash_attention",
        platform,
        flash_attention if platform == KernelPlatform.TPU else flash_mla_with_kvcache,
        config=cfg
    )
    
    register_kernel(
        "sliding_window_attention",
        platform,
        sliding_window_attention,
        config=cfg
    )
    
    # Flash attention metadata
    register_kernel(
        "flash_mla_metadata",
        platform,
        jax_get_mla_metadata if platform == KernelPlatform.TPU else get_mla_metadata,
        config=cfg
    )

def register_matmul_kernels(platform: KernelPlatform, config: Optional[Dict[str, Any]] = None):
    """Register matrix multiplication kernels."""
    cfg = {**get_default_config(platform), **(config or {})}
    
    # Tree-based matmul
    register_kernel(
        "tree_matmul",
        platform,
        TreeMatMul(
            leaf_size=cfg["block_size"],
            adaptive=True
        ),
        config=cfg
    )
    
    # Hybrid matmul
    register_kernel(
        "hybrid_matmul",
        platform,
        HybridMatMul(
            block_size=cfg["block_size"],
            use_bfloat16=cfg.get("use_bfloat16", False),
            use_tree=True
        ),
        config=cfg
    )

def register_sparse_kernels(platform: KernelPlatform, config: Optional[Dict[str, Any]] = None):
    """Register sparse computation kernels."""
    cfg = {**get_default_config(platform), **(config or {})}
    
    sparse_ops = SparseMatrixOps(
        block_size=cfg["block_size"],
        use_bfloat16=cfg.get("use_bfloat16", False)
    )
    
    register_kernel(
        "sparse_block_gemm",
        platform,
        sparse_ops.block_sparse_matmul,
        config=cfg
    )
    
    register_kernel(
        "block_sparse_attention",
        platform,
        sparse_ops.block_sparse_attention,
        config=cfg
    )

def register_expert_kernels(platform: KernelPlatform, config: Optional[Dict[str, Any]] = None):
    """Register expert parallelism kernels."""
    cfg = {**get_default_config(platform), **(config or {})}
    
    register_kernel(
        "rebalance_experts",
        platform,
        rebalance_experts,
        config=cfg
    )
    
    register_kernel(
        "replicate_experts",
        platform,
        replicate_experts,
        config=cfg
    )
    
    register_kernel(
        "balanced_packing",
        platform,
        balanced_packing,
        config=cfg
    )

def register_parallel_kernels(platform: KernelPlatform, config: Optional[Dict[str, Any]] = None):
    """Register efficient parallel operation kernels."""
    cfg = {**get_default_config(platform), **(config or {})}
    
    ep_ops = EfficientParallelOps(
        use_tpu_optimizations=platform == KernelPlatform.TPU,
        block_size=cfg["block_size"],
        use_bfloat16=cfg.get("use_bfloat16", False)
    )
    
    register_kernel(
        "batch_matmul",
        platform,
        ep_ops.batch_matmul,
        config=cfg
    )
    
    register_kernel(
        "parallel_scan",
        platform,
        ep_ops.parallel_scan,
        config=cfg
    )
    
    register_kernel(
        "parallel_sort",
        platform,
        ep_ops.parallel_sort,
        config=cfg
    )
    
    register_kernel(
        "strided_reduction",
        platform,
        ep_ops.strided_reduction,
        config=cfg
    )

def register_embedding_kernels(platform: KernelPlatform, config: Optional[Dict[str, Any]] = None):
    """Register embedding-related kernels."""
    cfg = {**get_default_config(platform), **(config or {})}
    
    register_kernel(
        "rope_embedding",
        platform,
        rope_embedding,
        config=cfg
    )

def register_optimized_kernels(platform: Optional[KernelPlatform] = None,
                             config: Optional[Dict[str, Any]] = None):
    """
    Register all optimized kernels with the kernel manager.
    
    Args:
        platform: Optional platform to register kernels for. If None, registers for all platforms.
        config: Optional configuration overrides.
    """
    platforms = ([platform] if platform else [KernelPlatform.TPU, KernelPlatform.GPU])
    
    kernel_registrars = {
        KernelGroup.ATTENTION: register_attention_kernels,
        KernelGroup.MATMUL: register_matmul_kernels,
        KernelGroup.SPARSE: register_sparse_kernels,
        KernelGroup.EXPERT: register_expert_kernels,
        KernelGroup.PARALLEL: register_parallel_kernels,
        KernelGroup.EMBEDDING: register_embedding_kernels
    }
    
    for platform in platforms:
        platform_config = {**get_default_config(platform), **(config or {})}
        
        for group, registrar in kernel_registrars.items():
            try:
                registrar(platform, platform_config)
            except Exception as e:
                print(f"Warning: Failed to register {group} kernels for {platform}: {e}")
                continue
