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
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_70,code=sm_70",  # Volta
                    "-gencode=arch=compute_75,code=sm_75",  # Turing
                    "-gencode=arch=compute_80,code=sm_80",  # Ampere
                    "-gencode=arch=compute_86,code=sm_86",  # Ada Lovelace/RTX 4000
                    "--use_fast_math",
                    "--ptxas-options=-v",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)