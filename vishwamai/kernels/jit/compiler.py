import hashlib
import functools
import os
import re
import subprocess
import uuid
from typing import Tuple
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import compile_custom_call
from flax.linen import Module
import optax


@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    return f'{os.path.dirname(os.path.abspath(__file__))}/../include'


@functools.lru_cache(maxsize=None)
def get_nvcc_compiler() -> Tuple[str, str]:
    paths = [f'{os.getenv("CUDA_HOME", "/usr/local/cuda")}/bin/nvcc']
    
    version_pattern = re.compile(r'release (\d+\.\d+)')
    for path in paths:
        if os.path.exists(path):
            match = version_pattern.search(os.popen(f'{path} --version').read())
            if match:
                version = match.group(1)
                return path, version
    raise RuntimeError('Cannot find any available NVCC compiler')


@functools.lru_cache(maxsize=None)
def get_cache_dir():
    return os.path.expanduser('~') + '/.deep_gemm_cache'


def build_jax(name: str, arg_defs: tuple, code: str):
    nvcc_flags = ['-std=c++17', '-shared', '-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda']
    cxx_flags = ['-fPIC', '-O3', '-Wno-deprecated-declarations']
    flags = [*nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
    include_dirs = [get_jit_include_dir()]

    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    so_path = f'{cache_dir}/kernel_{name}.so'
    tmp_so_path = f'{cache_dir}/nvcc.tmp.{uuid.uuid4()}.so'

    command = [get_nvcc_compiler()[0], '-o', tmp_so_path, *flags, *[f'-I{d}' for d in include_dirs]]
    subprocess.check_call(command)
    os.replace(tmp_so_path, so_path)

    return compile_custom_call(so_path)


def jax_kernel(x):
    kernel = build_jax("gemm", ("float32", "float32"), "// CUDA kernel")
    return jax.jit(kernel)(x)
