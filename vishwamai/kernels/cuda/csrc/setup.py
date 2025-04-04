from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA include paths
def get_cuda_include():
    if "CUDA_HOME" in os.environ:
        cuda_home = os.environ["CUDA_HOME"]
    else:
        cuda_home = "/usr/local/cuda"
    
    return [cuda_home + "/include"]

setup(
    name="flash_mla_cuda",
    ext_modules=[
        CUDAExtension(
            name="flash_mla_cuda",
            sources=[
                "flash_mla_cuda.cpp",
                "flash_mla_kernel.cu",
            ],
            include_dirs=get_cuda_include(),
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-march=native",
                    "-mtune=native"
                ],
                "nvcc": [
                    "-O3",
                    # Optimize specifically for GTX 1650 (Turing)
                    "-gencode=arch=compute_75,code=sm_75",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-maxrregcount=64",  # Optimize register usage for Turing
                    "--gpu-architecture=sm_75",  # Target Turing specifically
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-Xptxas=-O3,-v",  # Aggressive PTX optimization
                    "--compiler-options=-O3,-march=native,-mtune=native",
                    "--disable-warnings",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)