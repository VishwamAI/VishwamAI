"""VishwamAI optimized kernels for TPU and GPU."""

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

from .kernel import fp8_gemm_optimized
from .fp8_cast_bf16 import fp8_cast_to_bf16, bf16_cast_to_fp8

# Version
__version__ = "0.1.0"

__all__ = [
    'fp8_gemm_optimized',
    'fp8_cast_to_bf16',
    'bf16_cast_to_fp8'
]