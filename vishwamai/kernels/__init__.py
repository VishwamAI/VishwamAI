"""VishwamAI optimized kernels module."""

# Core kernel functionality
from .core import (
    KernelManager,
    register_kernel,
    get_kernel,
    KernelType,
    HardwareType
)

# Platform-specific implementations
from .cuda import (
    FlashMLACUDA,
    FlashKVCache,
    FP8GEMM
)

from .tpu import (
    TPUGEMMKernel,
    TPUAttentionKernel,
    TPULayerNormKernel
)

# JIT compilation
from .jit import (
    JITManager,
    compile_kernel,
    KernelTemplate
)

# Common operations
from .ops import (
    matmul,
    layer_norm,
    gelu,
    flash_attention,
    sparse_matmul
)

# Optimizations
from .optimizers import (
    quantize,
    prune,
    distill,
    tensor_parallel
)

# Version
__version__ = "0.1.0"

__all__ = [
    # Core
    "KernelManager",
    "register_kernel", 
    "get_kernel",
    "KernelType",
    "HardwareType",

    # CUDA kernels
    "FlashMLACUDA",
    "FlashKVCache",
    "FP8GEMM",

    # TPU kernels
    "TPUGEMMKernel",
    "TPUAttentionKernel", 
    "TPULayerNormKernel",

    # JIT
    "JITManager",
    "compile_kernel",
    "KernelTemplate",

    # Operations
    "matmul",
    "layer_norm",
    "gelu",
    "flash_attention",
    "sparse_matmul",

    # Optimizations
    "quantize",
    "prune",
    "distill",
    "tensor_parallel"
]