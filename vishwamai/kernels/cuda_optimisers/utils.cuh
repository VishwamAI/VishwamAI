#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint> // For uintptr_t
#include <cublas_v2.h> // For cuBLAS library

// Macro to check cuBLAS errors
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(err)); \
        } \
    } while (0)

namespace deep_ep {

// Utility functions for GTX 1650 optimizations
namespace utils {

// Calculate optimal grid size based on the number of elements and threads per block
inline __host__ int calcGridSize(int elements, int threads_per_block) {
    return (elements + threads_per_block - 1) / threads_per_block;
}

// Calculate optimal 2D grid size for matrix operations
inline __host__ dim3 calcOptimal2DGrid(int rows, int cols, int threads_x, int threads_y) {
    dim3 grid((cols + threads_x - 1) / threads_x, (rows + threads_y - 1) / threads_y);
    return grid;
}

// Get maximum shared memory size
inline __host__ size_t getMaxSharedMemPerBlock() {
    int dev_id;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    int shared_mem;
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, dev_id));
    return shared_mem;
}

// Check if tensor dimensions are aligned for optimal memory access
inline __host__ bool isMemoryAligned(int dim_size) {
    return (dim_size % EP_MEMORY_ALIGNMENT == 0);
}

// Get warp size for the current device (typically 32)
inline __host__ int getWarpSize() {
    int dev_id;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    int warp_size;
    CUDA_CHECK(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, dev_id));
    return warp_size;
}

// Round up to next multiple of warp size
inline __host__ int roundUpToWarpSize(int val) {
    int warp_size = getWarpSize();
    return ((val + warp_size - 1) / warp_size) * warp_size;
}

// Utility to pad tensor dimensions for optimal performance
inline __host__ int padDimension(int dim_size) {
    int alignment = EP_MEMORY_ALIGNMENT / sizeof(float);
    return ((dim_size + alignment - 1) / alignment) * alignment;
}

// Calculate optimal workspace size for flash attention on GTX 1650
inline __host__ size_t flashAttentionWorkspaceSize(int batch_size, int seq_len, int num_heads, int head_dim) {
    // Basic workspace includes scratch memory for intermediate results and optimization buffers
    // This is specific to GTX 1650's 4GB memory constraint
    const int bytes_per_element = sizeof(half);
    
    // For flash attention, we need:
    // 1. O(N) memory for m and l accumulators (2*batch_size*num_heads*seq_len*sizeof(float))
    // 2. O(BLK) memory for key/value tiles (2*batch_size*num_heads*BLK*head_dim*sizeof(half))
    // where BLK is the block size, typically 64 or 128 for GTX 1650
    const int block_size = 64;
    
    return 2 * batch_size * num_heads * seq_len * sizeof(float) + 
           2 * batch_size * num_heads * block_size * head_dim * bytes_per_element;
}

// Kernels and device functions

// Device function to compute warp-level reduction sum (within a single warp)
template<typename T>
__device__ inline T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = GTX1650_WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function for warp-level maximum
template<typename T>
__device__ inline T warpReduceMax(T val) {
    #pragma unroll
    for (int offset = GTX1650_WARP_SIZE/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// GEMM tuning parameters for GTX 1650
struct GemmTuningParams {
    int block_dim_x;
    int block_dim_y;
    int block_k;
    int thread_tile_m;
    int thread_tile_n;
    bool use_dp4a;  // Use DP4A instructions if available (int8 dot product)
};

// Get optimized GEMM tuning parameters for GTX 1650
inline __host__ GemmTuningParams getOptimalGemmParams(int m, int n, int k) {
    GemmTuningParams params;
    
    // Default parameters
    params.block_dim_x = 16;
    params.block_dim_y = 16;
    params.block_k = 8;
    params.thread_tile_m = 8;
    params.thread_tile_n = 8;
    params.use_dp4a = false;
    
    // Tune based on matrix dimensions
    if (m >= 1024 && n >= 1024) {
        // Large matrices - optimize for throughput
        params.block_dim_x = 32;
        params.block_dim_y = 32;
        params.block_k = 8;
    }
    else if (m <= 128 && n <= 128) {
        // Small matrices - optimize for low latency
        params.block_dim_x = 8;
        params.block_dim_y = 8;
        params.block_k = 8;
        params.thread_tile_m = 4;
        params.thread_tile_n = 4;
    }
    
    // Check if DP4A is worth using (4x4x4 dot product)
    if (k >= 32 && (k % 4 == 0) && GTX1650_COMPUTE_CAPABILITY >= 6.1f) {
        params.use_dp4a = true;
    }
    
    return params;
}

// Convert float to half precision with rounding
__device__ inline half float2half_rn(float f) {
    return __float2half_rn(f);
}

// Convert half precision to float
__device__ inline float half2float(half h) {
    return __half2float(h);
}

// Fast tanh approximation for fp16 on GTX 1650
__device__ inline half fast_tanh_half(half x) {
    half x2 = __hmul(x, x);
    half a = __hadd(x, __hmul(x, x2));
    half b = __hadd(__float2half(1.0f), __hmul(x2, __float2half(0.3f)));
    return __hdiv(a, b);
}

// Fast GELU approximation optimized for GTX 1650
__device__ inline half fast_gelu_half(half x) {
    // GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x^3)))
    const half sqrt_2_over_pi = __float2half(0.7978845608f);
    const half coef = __float2half(0.044715f);
    
    half x_cubed = __hmul(__hmul(x, x), x);
    half inner = __hmul(sqrt_2_over_pi, __hadd(x, __hmul(coef, x_cubed)));
    half tanh_inner = fast_tanh_half(inner);
    
    half one = __float2half(1.0f);
    half half_val = __float2half(0.5f);
    
    return __hmul(half_val, __hmul(x, __hadd(one, tanh_inner)));
}

// Aligned memory copy for better memory throughput
__device__ inline void alignedMemoryCopy(half* dst, const half* src, int size) {
    static_assert(sizeof(int4) == 16, "int4 should be 16 bytes");
    
    // Use vectorized loads for aligned data
    if (((uintptr_t)dst & 15) == 0 && ((uintptr_t)src & 15) == 0) {
        int4* dst_vec = reinterpret_cast<int4*>(dst);
        const int4* src_vec = reinterpret_cast<const int4*>(src);
        
        int vec_elements = size / 8; // int4 has 8 half values
        for (int i = 0; i < vec_elements; i++) {
            dst_vec[i] = src_vec[i];
        }
        
        // Copy remaining elements
        int processed = vec_elements * 8;
        for (int i = processed; i < size; i++) {
            dst[i] = src[i];
        }
    } else {
        // Fallback for unaligned memory
        for (int i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }
}

// Initialize CUDA for optimal performance on GTX 1650
inline __host__ void optimizeForGTX1650() {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    
    // Set cache configuration to prefer shared memory
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    
    // Create a cuBLAS handle before using it
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
    
    // For Turing architecture, enable tensor cores if available
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    if (prop.major >= 7) {
        // Enable tensor cores - use CUBLAS_CHECK for cuBLAS functions
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    }
    
    // Destroy the handle when done
    CUBLAS_CHECK(cublasDestroy(handle));
}

} // namespace utils

// Simple synchronization primitive since we don't have NVSHMEM
template<int kNumRanks>
__device__ void barrier_device(int** task_fifo_ptrs, int head, int rank) {
    // Simple CUDA barrier - just a placeholder since we don't have NVSHMEM
    __syncthreads();
    
    // In a real implementation this would synchronize across multiple GPUs
    // But for GTX 1650 we'll just do a simple barrier within the GPU
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Increment the counter at the head position
        atomicAdd(task_fifo_ptrs[0] + head, 1);
        
        // Wait until all ranks have reached this point
        while (atomicAdd(task_fifo_ptrs[0] + head, 0) < kNumRanks) {
            // Spin wait - but avoid wasting too much memory bandwidth
            __nanosleep(1000);
        }
    }
    
    // Make sure all threads in the block wait for the barrier
    __syncthreads();
}

} // namespace deep_ep

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[1]; \
    attr[0].id = cudaLaunchAttributeCooperative; \
    attr[0].val.cooperative = 1; \
    cfg.attrs = attr; \
    cfg.numAttrs = 1
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif

#define SWITCH_RANKS(case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        default: EP_HOST_ASSERT(false and "Unsupported ranks"); \
    } while (false)

#define SWITCH_RDMA_RANKS(case_macro) \
    switch (num_ranks / NUM_MAX_NVL_PEERS) { \
        case 2: case_macro(2); \
        case 3: case_macro(3); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        case 16: case_macro(16); \
        case 18: case_macro(18); \
        case 20: case_macro(20); \
        default: EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    } while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(dtype, 2); \
        case 4: case_macro(dtype, 4); \
        case 8: case_macro(dtype, 8); \
        default: EP_HOST_ASSERT(false && "Unsupported ranks"); \
    } while (false)

#define SWITCH_TYPES(case_macro) \
    switch (type) { \
        case CUDA_R_16BF: case_macro(nv_bfloat16); \
        case CUDA_R_32F:  case_macro(float); \
        default: EP_HOST_ASSERT(false && "Unsupported type"); \
    } while (false)

#define SWITCH_HIDDEN(case_macro) \
    switch (hidden) { \
        case 2560: case_macro(2560); \
        case 5120: case_macro(5120); \
        case 7168: case_macro(7168); \
        default: EP_HOST_ASSERT(false && "Unsupported hidden"); \
    } while (false)