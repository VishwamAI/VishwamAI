"""JIT compilation system for CUDA and TPU kernels."""
import hashlib
import functools
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union, Callable
import jax
import jax.numpy as jnp
from jax import lax
from jaxlib.xla_extension import compile_custom_call

from vishwamai.kernels.core.kernel import KernelConfig

class KernelPlatform:
    """Supported platforms for kernel compilation."""
    TPU = "tpu"
    GPU = "gpu"
    CPU = "cpu"

class KernelTemplate:
    """Enhanced template for generating kernel code."""
    
    def __init__(
        self,
        template: str,
        platform: str = KernelPlatform.TPU,
        config: Optional[Dict[str, Any]] = None
    ):
        self.template = template
        self.platform = platform
        self.config = config or {}
        
    def instantiate(self, **kwargs) -> str:
        """
        Instantiate the template with given parameters.
        
        Args:
            **kwargs: Template parameters
            
        Returns:
            Instantiated kernel code
        """
        template_args = {**self.config, **kwargs}
        
        # Add platform-specific optimizations
        if self.platform == KernelPlatform.TPU:
            template_args.update({
                "block_size": 128,
                "use_bfloat16": True,
                "tpu_layout_opt": True
            })
        elif self.platform == KernelPlatform.GPU:
            template_args.update({
                "block_size": 64,
                "use_fp16": True,
                "use_tensor_cores": True
            })
            
        return self.template.format(**template_args)

@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    """Return the include directory for JIT kernels."""
    return f'{os.path.dirname(os.path.abspath(__file__))}/../include'

@functools.lru_cache(maxsize=None)
def get_compiler_info(platform: str) -> Tuple[str, str]:
    """
    Get compiler information for the specified platform.
    
    Args:
        platform: Target platform (tpu/gpu/cpu)
        
    Returns:
        Tuple of (compiler_path, version)
    """
    if platform == KernelPlatform.GPU:
        paths = [f'{os.getenv("CUDA_HOME", "/usr/local/cuda")}/bin/nvcc']
        version_pattern = re.compile(r'release (\d+\.\d+)')
        
        for path in paths:
            if os.path.exists(path):
                try:
                    version_output = subprocess.check_output([path, '--version']).decode('utf-8')
                    match = version_pattern.search(version_output)
                    if match:
                        return path, match.group(1)
                except (subprocess.SubprocessError, OSError):
                    continue
        raise RuntimeError('Cannot find NVCC compiler')
        
    elif platform == KernelPlatform.TPU:
        # For TPU, we use JAX's built-in XLA compiler
        return "xla", jax.__version__
    else:
        return "gcc", "latest"  # Fallback for CPU

class KernelCache:
    """Cache for compiled kernels."""
    
    def __init__(self, platform: str):
        self.platform = platform
        self.cache_dir = self._get_cache_dir()
        self.kernels: Dict[str, Callable] = {}
        
    def _get_cache_dir(self) -> str:
        """Get platform-specific cache directory."""
        base_dir = os.path.expanduser('~/.vishwamai_cache')
        platform_dir = os.path.join(base_dir, self.platform)
        os.makedirs(platform_dir, exist_ok=True)
        return platform_dir
        
    def get_kernel(self, name: str, code_hash: str) -> Optional[Callable]:
        """Get cached kernel if it exists."""
        key = f"{name}_{code_hash}"
        
        if key in self.kernels:
            return self.kernels[key]
            
        so_path = os.path.join(self.cache_dir, f"kernel_{key}.so")
        if os.path.exists(so_path):
            kernel = compile_custom_call(so_path)
            self.kernels[key] = kernel
            return kernel
            
        return None
        
    def cache_kernel(self, name: str, code_hash: str, so_path: str, kernel: Callable):
        """Cache a compiled kernel."""
        key = f"{name}_{code_hash}"
        cached_path = os.path.join(self.cache_dir, f"kernel_{key}.so")
        
        if not os.path.exists(cached_path):
            os.replace(so_path, cached_path)
            
        self.kernels[key] = kernel

# Global kernel caches
_kernel_caches: Dict[str, KernelCache] = {}

def get_kernel_cache(platform: str) -> KernelCache:
    """Get or create kernel cache for platform."""
    if platform not in _kernel_caches:
        _kernel_caches[platform] = KernelCache(platform)
    return _kernel_caches[platform]

def compile_kernel(
    name: str,
    arg_defs: tuple,
    template: Union[str, KernelTemplate],
    platform: str = KernelPlatform.TPU,
    template_args: Optional[Dict[str, Any]] = None,
    config: Optional[KernelConfig] = None
) -> Callable:
    """
    Compile a kernel from a template.
    
    Args:
        name: Name of the kernel
        arg_defs: Tuple of argument types
        template: Kernel code template or KernelTemplate instance
        platform: Target platform
        template_args: Arguments to instantiate template
        config: Optional kernel configuration
        
    Returns:
        Compiled kernel function
    """
    if isinstance(template, str):
        template = KernelTemplate(template, platform, config)
        
    code = template.instantiate(**(template_args or {}))
    return build_kernel(name, arg_defs, code, platform)

def build_kernel(
    name: str,
    arg_defs: tuple,
    code: str,
    platform: str = KernelPlatform.TPU
) -> Callable:
    """
    Build a custom kernel for the specified platform.
    
    Args:
        name: The kernel name
        arg_defs: Tuple of argument types
        code: Kernel code as string
        platform: Target platform
        
    Returns:
        A compiled kernel that can be called from JAX
    """
    code_hash = hashlib.md5(code.encode()).hexdigest()
    cache = get_kernel_cache(platform)
    
    # Check cache first
    cached_kernel = cache.get_kernel(name, code_hash)
    if cached_kernel:
        return cached_kernel
    
    # Compile based on platform
    if platform == KernelPlatform.TPU:
        kernel = _build_tpu_kernel(name, arg_defs, code)
    elif platform == KernelPlatform.GPU:
        kernel = _build_gpu_kernel(name, arg_defs, code)
    else:
        kernel = _build_cpu_kernel(name, arg_defs, code)
        
    # Cache the result
    cache.cache_kernel(name, code_hash, f"{cache.cache_dir}/tmp_{uuid.uuid4()}.so", kernel)
    
    return kernel

def _build_tpu_kernel(name: str, arg_defs: tuple, code: str) -> Callable:
    """Build TPU kernel using JAX/XLA."""
    # For TPU, we use JAX's primitives and XLA
    def kernel_impl(*args):
        # Convert code to XLA HLO operations
        # This is a simplified example - actual implementation would parse the code
        # and generate appropriate JAX operations
        if "matmul" in name.lower():
            return jax.lax.dot_general(args[0], args[1], dimension_numbers=(((1,), (0,)), ((), ())))
        elif "reduce" in name.lower():
            return jax.lax.reduce(args[0], 0., jax.lax.add, dimensions=(0,))
        else:
            raise NotImplementedError(f"TPU kernel type not implemented: {name}")
            
    return jax.jit(kernel_impl)

def _build_gpu_kernel(name: str, arg_defs: tuple, code: str) -> Callable:
    """Build CUDA kernel for GPU."""
    nvcc_flags = [
        '-std=c++17', '-shared', '-O3',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '--use_fast_math'
    ]
    cxx_flags = ['-fPIC', '-O3', '-Wno-deprecated-declarations']
    flags = [*nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
    include_dirs = [get_jit_include_dir()]
    
    # Get temporary paths
    cache_dir = get_kernel_cache(KernelPlatform.GPU).cache_dir
    tmp_id = uuid.uuid4()
    so_path = f'{cache_dir}/nvcc.tmp.{tmp_id}.so'
    cuda_path = f'{cache_dir}/kernel_{tmp_id}.cu'
    
    # Generate and compile CUDA code
    with open(cuda_path, 'w') as f:
        f.write(generate_cuda_kernel(name, arg_defs, code))
        
    try:
        nvcc_path = get_compiler_info(KernelPlatform.GPU)[0]
        command = [
            nvcc_path, '-o', so_path, cuda_path,
            *flags, *[f'-I{d}' for d in include_dirs]
        ]
        subprocess.check_call(command)
        return compile_custom_call(so_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"CUDA kernel compilation failed: {e}")
    finally:
        # Cleanup temporary files
        if os.path.exists(cuda_path):
            os.unlink(cuda_path)

def _build_cpu_kernel(name: str, arg_defs: tuple, code: str) -> Callable:
    """Build CPU kernel (fallback implementation)."""
    # For CPU, we use a simple numpy/JAX implementation
    def cpu_kernel(*args):
        if "matmul" in name.lower():
            return jnp.matmul(args[0], args[1])
        elif "reduce" in name.lower():
            return jnp.sum(args[0], axis=0)
        else:
            raise NotImplementedError(f"CPU kernel not implemented: {name}")
            
    return cpu_kernel

def generate_cuda_kernel(name: str, arg_defs: tuple, code: str) -> str:
    """
    Generate CUDA kernel code with appropriate includes and function signatures.
    
    Args:
        name: The kernel name
        arg_defs: Tuple of argument types
        code: Core kernel code
    
    Returns:
        Complete CUDA kernel code as a string
    """
    cuda_code = f"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

// Include VishwamAI headers
#include "deepgemm/utils.cuh"
#include "deepgemm/mma_utils.cuh"
#include "deepgemm/tensor_utils.cuh"
#include "deepgemm/memory_utils.cuh"

extern "C" __global__ void {name}_kernel(
    {', '.join([f'{arg_type} *in{i}' for i, arg_type in enumerate(arg_defs)])}
) {{
    // Get global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Kernel implementation
    {code}
}}

// XLA custom call interface
extern "C" void {name}(
    {', '.join([f'{arg_type} *in{i}' for i, arg_type in enumerate(arg_defs)])}
) {{
    // Configure grid and block dimensions
    const int block_size = 256;
    const int grid_size = (1024 + block_size - 1) / block_size;
    
    dim3 grid(grid_size, 1, 1);
    dim3 block(block_size, 1, 1);
    
    // Launch kernel
    {name}_kernel<<<grid, block>>>(
        {', '.join([f'in{i}' for i in range(len(arg_defs))])}
    );
}}
"""
    return cuda_code
