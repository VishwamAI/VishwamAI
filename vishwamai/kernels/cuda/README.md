# CUDA Kernels for Gemma 3 Knowledge Distillation

This directory contains CUDA-optimized kernels for efficient knowledge distillation of Gemma 3 models. The implementation focuses on high-performance GPU computation with memory efficiency.

## Features

- **Flash Attention Implementation**
  - Memory-efficient attention computation
  - Block-wise processing for large sequences
  - Optimized KV-cache with compression
  - Support for mixed precision (FP16/BF16)

- **Memory Management**
  - Custom memory pool for efficient allocation
  - Memory defragmentation support
  - Smart caching strategies
  - Memory access pattern optimization

- **Performance Optimizations**
  - Tensor Core utilization for matrix operations
  - Multi-stream execution support
  - Operation fusion for reduced memory bandwidth
  - Dynamic kernel selection based on hardware capabilities

## Requirements

- CUDA Toolkit >= 11.0
- GPU with Compute Capability >= 7.0 (Volta architecture or newer)
- C++17 compatible compiler
- CMake >= 3.18

## Components

### Core Components

1. `distillation_kernels.cu`
   - Main implementation of distillation kernels
   - Teacher-student attention computation
   - Loss calculation and backpropagation

2. `kernel_analyzer.cuh`
   - Performance analysis tools
   - Memory access pattern analysis
   - Kernel occupancy optimization

3. `memory_manager.cuh`
   - Custom memory allocator
   - Memory pool management
   - Defragmentation support

4. `flash_kv.py`
   - KV-cache implementation
   - Cache compression strategies
   - Efficient memory layout

### Support Files

- `version.h`: Version information and compatibility checks
- `configs.cuh`: Hardware-specific configurations
- `launch.cuh`: Kernel launch utilities
- `utils.cuh`: Common CUDA utilities

## Usage

### Basic Usage

```cpp
#include "distillation_kernels.cu"
#include "memory_manager.cuh"

// Initialize memory manager
cuda_memory::MemoryPoolConfig pool_config{
    /* initial_pool_size= */ 1ULL << 30,  // 1GB
    /* max_pool_size=    */ 1ULL << 32,   // 4GB
    /* block_size=       */ 1ULL << 20    // 1MB
};
cuda_memory::CUDAMemoryManager::getInstance().initialize(pool_config);

// Create distillation kernels
cuda_kernels::TensorDims dims{
    /* batch_size= */ 32,
    /* seq_length= */ 512,
    /* num_heads=  */ 12,
    /* head_dim=   */ 64
};
cuda_kernels::DistillationKernels kernels;
kernels.initialize(dims);

// Run attention computation
kernels.compute_attention(
    teacher_q, teacher_k, teacher_v,
    student_q, student_k, student_v,
    output, attention_weights,
    temperature
);
```

### Memory Management

```cpp
// Allocate memory with automatic management
cuda_memory::CUDAMemoryPtr<float> buffer(size);

// Memory will be automatically freed when buffer goes out of scope
{
    cuda_memory::CUDAMemoryPtr<float> temp_buffer(size);
    // Use temp_buffer...
} // Memory freed here
```

### Performance Analysis

```cpp
// Create analyzer
cuda_analysis::KernelAnalyzer analyzer;

// Profile kernel execution
auto metrics = analyzer.analyzeKernel(
    kernel_function,
    grid_dim,
    block_dim
);

// Get performance report
std::cout << analyzer.generateReport();
```

## Performance Optimization Tips

1. **Memory Access**
   - Use aligned memory access patterns
   - Leverage shared memory for frequently accessed data
   - Minimize global memory transactions

2. **Kernel Configuration**
   - Choose block sizes that maximize occupancy
   - Use multiple CUDA streams for overlapping execution
   - Enable Tensor Cores when possible

3. **Memory Management**
   - Reuse memory allocations when possible
   - Use the memory pool for frequent allocations/deallocations
   - Monitor fragmentation and trigger defragmentation when needed

4. **Attention Computation**
   - Use block-wise processing for large sequences
   - Implement efficient KV-cache strategies
   - Optimize attention mask operations

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Check memory pool configuration
   - Monitor fragmentation levels
   - Consider enabling defragmentation
   - Reduce batch size or sequence length

2. **Performance Issues**
   - Verify Tensor Core utilization
   - Check kernel occupancy
   - Analyze memory access patterns
   - Profile kernel execution times

3. **Compatibility Issues**
   - Verify CUDA version compatibility
   - Check GPU compute capability
   - Ensure correct driver version

### Debugging Tools

- Use `cuda-memcheck` for memory issues
- Enable profiling with `nvprof` or NSight
- Check kernel analysis reports
- Monitor GPU memory usage

## Contributing

When contributing to the CUDA kernels:

1. Follow the coding style guidelines
2. Add appropriate unit tests
3. Document performance implications
4. Update README with new features
5. Run full test suite before submitting

## License

This code is part of the VishwamAI project and is licensed under the project's terms.
