"""JIT compilation system for CUDA kernels."""
import hashlib
import functools
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import compile_custom_call


@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    """Return the include directory for JIT kernels."""
    return f'{os.path.dirname(os.path.abspath(__file__))}/../include'


@functools.lru_cache(maxsize=None)
def get_nvcc_compiler() -> Tuple[str, str]:
    """Find and return the NVCC compiler path and version."""
    paths = [f'{os.getenv("CUDA_HOME", "/usr/local/cuda")}/bin/nvcc']
    
    version_pattern = re.compile(r'release (\d+\.\d+)')
    for path in paths:
        if os.path.exists(path):
            try:
                version_output = subprocess.check_output([path, '--version']).decode('utf-8')
                match = version_pattern.search(version_output)
                if match:
                    version = match.group(1)
                    return path, version
            except (subprocess.SubprocessError, OSError):
                continue
    raise RuntimeError('Cannot find any available NVCC compiler')


@functools.lru_cache(maxsize=None)
def get_cache_dir() -> str:
    """Return the cache directory for compiled kernels."""
    cache_dir = os.path.expanduser('~/.deepgemm_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class KernelTemplate:
    """Template for generating CUDA kernel code."""
    
    def __init__(self, template: str):
        self.template = template
        
    def instantiate(self, **kwargs) -> str:
        """Instantiate the template with given parameters."""
        return self.template.format(**kwargs)


def compile_kernel(
    name: str,
    arg_defs: tuple,
    template: Union[str, KernelTemplate],
    template_args: Optional[Dict[str, Any]] = None
):
    """
    Compile a kernel from a template.
    
    Args:
        name: Name of the kernel
        arg_defs: Tuple of argument types
        template: Kernel code template or KernelTemplate instance
        template_args: Arguments to instantiate template
        
    Returns:
        Compiled kernel function
    """
    if isinstance(template, KernelTemplate):
        code = template.instantiate(**(template_args or {}))
    else:
        code = template
        
    return build_jax(name, arg_defs, code)


def build_jax(name: str, arg_defs: tuple, code: str):
    """
    Build a custom CUDA kernel and return a JAX-compatible callable.
    
    Args:
        name: The kernel name
        arg_defs: Tuple of argument types
        code: CUDA kernel code as string
        
    Returns:
        A compiled kernel that can be called from JAX
    """
    # Create unique kernel file with hash of code
    code_hash = hashlib.md5(code.encode()).hexdigest()
    kernel_name = f"{name}_{code_hash}"
    
    nvcc_flags = ['-std=c++17', '-shared', '-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda']
    cxx_flags = ['-fPIC', '-O3', '-Wno-deprecated-declarations']
    flags = [*nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
    include_dirs = [get_jit_include_dir()]

    cache_dir = get_cache_dir()
    so_path = f'{cache_dir}/kernel_{kernel_name}.so'
    tmp_so_path = f'{cache_dir}/nvcc.tmp.{uuid.uuid4()}.so'
    cuda_path = f'{cache_dir}/kernel_{kernel_name}.cu'
    
    # Only compile if the kernel doesn't exist
    if not os.path.exists(so_path):
        # Write the kernel to a file
        os.makedirs(os.path.dirname(cuda_path), exist_ok=True)
        with open(cuda_path, 'w') as f:
            f.write(generate_cuda_kernel(name, arg_defs, code))
            
        try:
            nvcc_path = get_nvcc_compiler()[0]
            command = [nvcc_path, '-o', tmp_so_path, cuda_path, *flags, 
                      *[f'-I{d}' for d in include_dirs]]
            
            subprocess.check_call(command)
            os.replace(tmp_so_path, so_path)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Kernel compilation failed: {e}")
    
    return compile_custom_call(so_path)


def generate_cuda_kernel(name: str, arg_defs: tuple, code: str) -> str:
    """
    Generate CUDA kernel code with appropriate includes and function signatures
    
    Args:
        name: The kernel name
        arg_defs: Tuple of argument types
        code: Core kernel code
    
    Returns:
        Complete CUDA kernel code as a string
    """
    # Generate code with appropriate includes and function signatures
    cuda_code = f"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

// Include DeepGEMM headers
#include "deepgemm/utils.cuh"
#include "deepgemm/mma_utils.cuh"

extern "C" __global__ void {name}_kernel(
    {', '.join([f'{arg_type} *in{i}' for i, arg_type in enumerate(arg_defs)])}
) {{
    // Kernel implementation
    {code}
}}

// XLA custom call interface
extern "C" void {name}(
    {', '.join([f'{arg_type} *in{i}' for i, arg_type in enumerate(arg_defs)])}
) {{
    dim3 grid(32, 1, 1);
    dim3 block(256, 1, 1);
    {name}_kernel<<<grid, block>>>(
        {', '.join([f'in{i}' for i in range(len(arg_defs))])}
    );
}}
"""
    return cuda_code


def jax_kernel(x):
    """Sample kernel for demonstration"""
    kernel_code = """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        in0[idx] = __float2half(1.0f);
    }
    """
    kernel = build_jax("gemm_example", ("float32", "float32"), kernel_code)
    return jax.jit(kernel)(x)
