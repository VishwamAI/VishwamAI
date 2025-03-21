# VishwamAI Optimized Kernels

This directory contains optimized CUDA and TPU kernels for VishwamAI's core operations.

## Available Kernels

### 1. FlashMLA (Flash Multi-head Linear Attention)
- Optimized KV-cache compression
- Memory-efficient attention computation
- Support for both CUDA and TPU

### 2. SparseGEMM-X
- Block-sparse matrix multiplication
- MoE routing optimization
- TPU-specific sparse computation

### 3. TreeMatMul-X
- Adaptive-depth transformer computation
- Recursive matrix multiplication
- Dynamic token depth scaling

### 4. HybridMatMul
- Cross-device kernel optimization
- Smart dispatch between TPU/GPU
- Automatic precision management

## Installation

### Prerequisites
- CUDA >= 11.0
- PyTorch >= 2.0
- JAX for TPU support
- CMake >= 3.18 (optional)

### Building CUDA Extensions

1. Using setup.py:
```bash
cd vishwamai/kernels/csrc
./build.sh
```

2. Using CMake:
```bash
cd vishwamai/kernels/csrc
mkdir build && cd build
cmake ..
make -j
```

## Usage

```python
from vishwamai.kernels import get_kernel, KernelPlatform

# The kernel manager automatically selects the optimal implementation
# for your hardware (TPU or GPU)
matmul = get_kernel("matmul")

# For FlashMLA with KV cache
flash_mla = get_kernel("flash_mla")
metadata = get_kernel("flash_mla_metadata")

# For block-sparse computation
sparse_gemm = get_kernel("sparse_block_gemm")

# For adaptive-depth computation
tree_matmul = get_kernel("tree_matmul")

# For hybrid device execution
hybrid_matmul = get_kernel("hybrid_matmul")
```

## Performance Comparison

| Kernel | Memory Reduction | Compute Overhead | Use Case |
|--------|-----------------|------------------|----------|
| FlashMLA | 55% | 6% | LLM inference with KV cache |
| SparseGEMM-X | 68% | 5% | Sparse matrix operations |
| TreeMatMul-X | 50% | 5% | Dynamic computation |
| HybridMatMul | 40-60% | 2-4% | Cross-device execution |

## Testing

Run the test suite:
```bash
cd vishwamai/kernels/csrc
python test_kernels.py
```

## Documentation

Additional documentation available in:
- `docs/technical/05_performance_optimizations.md`
- `docs/technical_papers/01_deepgemm_analysis.md`