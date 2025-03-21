"""DeepGEMM kernel module."""

# Import core components
from .jit_manager import (
    # Core functionality
    get_kernel,
    register_kernel,
    get_manager,
    JITManager,
    
    # Platform detection
    KernelPlatform,
    
    # Kernel decorators
    tpu_kernel,
    gpu_kernel,
    triton_kernel,
    cpu_kernel,
    no_jit,
    
    # Common kernel functions
    matmul,
    layer_norm,
    gelu,
    flash_attention,
)

# Import optimized kernels
from .kernel import (
    fp8_gemm_optimized,
    act_quant,
    optimize_kernel_layout,
    block_tpu_matmul,
)

from .sparse import (
    sparse_gemm,
    sparse_attention,
    block_sparse_attention,
    sparse_block_gemm,
    create_sparse_mask,
)

from .flash_kv import (
    FlashKVCache,
    KVCache,
)

from .tree_matmul import (
    TreeMatMul,
    create_adaptive_depth_mask,
)

from .hybrid_matmul import (
    HybridMatMul,
)

from .fp8_cast_bf16 import (
    fp8_cast_to_bf16,
    bf16_cast_to_fp8,
    FP8CastManager,
    DynamicFP8Scaler,
)

# Version
__version__ = "0.1.0"

__all__ = [
    # Core functionality
    'get_kernel',
    'register_kernel',
    'get_manager',
    'JITManager',
    'KernelPlatform',
    
    # Kernel decorators
    'tpu_kernel',
    'gpu_kernel',
    'triton_kernel',
    'cpu_kernel',
    'no_jit',
    
    # Common kernel functions
    'matmul',
    'layer_norm',
    'gelu',
    'flash_attention',
    
    # Optimized kernels
    'fp8_gemm_optimized',
    'act_quant',
    'optimize_kernel_layout',
    'block_tpu_matmul',
    
    # Sparse operations
    'sparse_gemm',
    'sparse_attention', 
    'block_sparse_attention',
    'sparse_block_gemm',
    'create_sparse_mask',
    
    # KV cache
    'FlashKVCache',
    'KVCache',
    
    # Tree computation
    'TreeMatMul',
    'create_adaptive_depth_mask',
    
    # Hybrid execution
    'HybridMatMul',
    
    # FP8 casting
    'fp8_cast_to_bf16',
    'bf16_cast_to_fp8',
    'FP8CastManager',
    'DynamicFP8Scaler',
]