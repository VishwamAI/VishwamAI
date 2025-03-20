import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Check CUDA version
cuda_version = torch.version.cuda.split('.')
cuda_major = int(cuda_version[0])
cuda_minor = int(cuda_version[1])

# Define source files
sources = [
    'flash_mla.cu',
    'flash_mla_cuda.cpp',
]

# Define include directories
include_dirs = []

# Define NVCC arguments
nvcc_args = [
    '-O3',
    '--use_fast_math',
    '-std=c++14',
    f'-gencode=arch=compute_70,code=sm_70',
    f'-gencode=arch=compute_75,code=sm_75',
    f'-gencode=arch=compute_80,code=sm_80',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
]

# If CUDA version >= 11, add Ampere support
if cuda_major >= 11:
    nvcc_args.extend([
        f'-gencode=arch=compute_86,code=sm_86',
    ])

# Support for advanced hardware if available
if cuda_major >= 12:
    nvcc_args.extend([
        f'-gencode=arch=compute_90,code=sm_90',
    ])

setup(
    name='flash_mla_cuda',
    ext_modules=[
        CUDAExtension(
            name='flash_mla_cuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': nvcc_args,
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)