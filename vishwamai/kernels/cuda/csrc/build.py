import os
import torch
from torch.utils.cpp_extension import load

def build_cuda_extension():
    """Build the CUDA extension with current system's CUDA configuration."""
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source files
    sources = [
        os.path.join(current_dir, "flash_mla_cuda.cpp"),
        os.path.join(current_dir, "flash_mla_kernel.cu")
    ]
    
    # CUDA include paths
    cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
    include_dirs = [os.path.join(cuda_home, "include")]
    
    # Build the extension
    flash_mla_cuda = load(
        name="flash_mla_cuda",
        sources=sources,
        extra_include_paths=include_dirs,
        extra_cuda_cflags=[
            "-O3",
            "--gpu-architecture=sm_75",  # Turing architecture (GTX 1650)
            "--use_fast_math",
            "-maxrregcount=64",
            "--ptxas-options=-v",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
        verbose=True
    )
    
    return flash_mla_cuda

if __name__ == "__main__":
    build_cuda_extension()