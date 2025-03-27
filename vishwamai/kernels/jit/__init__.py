"""JIT compilation management for VishwamAI kernels."""

from .jit_manager import (
    JITManager,
    get_manager,
    KernelPlatform
)

from .compiler import (
    build_jax,
    compile_kernel,
    KernelTemplate,
    get_cache_dir,
    get_jit_include_dir
)

from .decorators import (
    tpu_kernel,
    gpu_kernel,
    triton_kernel,
    cpu_kernel,
    no_jit
)

from .templates import (
    get_template,
    TEMPLATES,
    MATMUL_TEMPLATE,
    ELEMENTWISE_TEMPLATE,
    REDUCTION_TEMPLATE
)

__all__ = [
    # Core JIT functionality
    "JITManager",
    "get_manager",
    "KernelPlatform",
    
    # Compilation
    "build_jax",
    "compile_kernel",
    "KernelTemplate",
    "get_cache_dir",
    "get_jit_include_dir",
    
    # Decorators
    "tpu_kernel",
    "gpu_kernel", 
    "triton_kernel",
    "cpu_kernel",
    "no_jit",
    
    # Templates
    "get_template",
    "TEMPLATES",
    "MATMUL_TEMPLATE",
    "ELEMENTWISE_TEMPLATE",
    "REDUCTION_TEMPLATE"
]