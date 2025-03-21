import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_cuda_arch_flags():
    """Get CUDA architecture flags for compilation."""
    # Default architectures if not specified
    DEFAULT_CUDA_ARCHS = [
        "7.0",  # Volta: V100
        "7.5",  # Turing: T4, RTX 2080
        "8.0",  # Ampere: A100
        "8.6",  # Ampere: RTX 3090
        "8.9",  # Hopper: H100
    ]
    
    # Use environment variable if set
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return os.environ["TORCH_CUDA_ARCH_LIST"]
        
    return ";".join([f"sm_{arch.replace('.', '')}" for arch in DEFAULT_CUDA_ARCHS])

setup(
    name="flash_mla_cuda",
    ext_modules=[
        CUDAExtension(
            name="flash_mla_cuda",
            sources=[
                "flash_mla_cuda.cpp",
                "flash_mla.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    f"-arch={get_cuda_arch_flags()}",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    "--extended-lambda",
                    "--expt-relaxed-constexpr",
                ]
            },
            include_dirs=["."],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)