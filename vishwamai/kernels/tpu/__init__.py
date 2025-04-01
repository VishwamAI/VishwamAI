"""TPU-optimized kernels for VishwamAI."""

from .tpu_custom_call import (
    tpu_custom_call,
    optimize_tpu_layout,
    pad_to_tpu_multiple,
)

from .gemm import TPUGEMMKernel
from .layer_norm import TPULayerNormKernel
from .flash_attention import TPUFlashAttention, FlashAttentionOutput

from .attention_kernels import (
    flash_attention_inference,
    TPUOptimizedAttention,
    memory_efficient_attention,
    compile_attention_kernels
)

from .kernel_fusion import (
    TPUKernelFusion,
    FusionConfig,
    FusionPattern
)

from .kernel_profiler import (
    TPUKernelProfiler,
    KernelProfile,
)

from .cache_manager import (
    TPUCacheManager,
    CacheConfig,
    CacheState
)

from .distillation_kernels import (
    DistillationKernelManager,
    DistillationKernelConfig,
    DistillationOutput
)

__all__ = [
    # Custom call utilities
    "tpu_custom_call",
    "compile_tpu_kernel",
    "optimize_tpu_layout",
    "pad_to_tpu_multiple",
    "get_optimal_tpu_layout",
    
    # Core TPU kernels
    "TPUGEMMKernel",
    "TPULayerNormKernel",
    "TPUFlashAttention",
    "FlashAttentionOutput",
    
    # Attention mechanisms
    "flash_attention_inference",
    "TPUOptimizedAttention",
    "memory_efficient_attention",
    "compile_attention_kernels",
    
    # Kernel fusion
    "TPUKernelFusion",
    "FusionConfig",
    "FusionPattern",
    
    # Profiling
    "TPUKernelProfiler",
    "KernelProfile",
    
    # Cache management
    "TPUCacheManager",
    "CacheConfig", 
    "CacheState",
    
    # Distillation
    "DistillationKernelManager",
    "DistillationKernelConfig",
    "DistillationOutput"
]