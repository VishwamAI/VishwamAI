#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

// Define constants for GTX 1650
#define GTX1650_MAX_THREADS_PER_BLOCK 1024
#define GTX1650_MAX_GRID_SIZE 2147483647
#define GTX1650_WARP_SIZE 32
#define GTX1650_MAX_SHARED_MEMORY 49152  // 48KB
#define GTX1650_COMPUTE_CAPABILITY 7.5f  // Turing architecture

// Optimized block sizes for GTX 1650
#define EP_BLOCK_SIZE_X 128
#define EP_BLOCK_SIZE_Y 1
#define EP_BLOCK_SIZE_Z 1

#define EP_DEFAULT_BLOCK_SIZE EP_BLOCK_SIZE_X

// Error checking macros
#define EP_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#define EP_HOST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error("Assertion failed: " #cond); \
        } \
    } while (0)

// CUDA error checking
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t status = (expr); \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(status) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

// Memory alignment for optimal access patterns
#define EP_MEMORY_ALIGNMENT 128

// Simplified team/peer configuration since we don't have NVSHMEM
#define NUM_MAX_NVL_PEERS 8
#define MAX_RANKS 64

// Configuration for kernel launches
#define SETUP_LAUNCH_CONFIG(num_blocks, num_threads, stream_val) \
    LaunchConfig cfg; \
    cfg.num_blocks = num_blocks; \
    cfg.num_threads = num_threads; \
    cfg.stream = stream_val

#define LAUNCH_KERNEL(cfg_ptr, kernel_name, ...) \
    kernel_name<<<cfg_ptr->num_blocks, cfg_ptr->num_threads, 0, cfg_ptr->stream>>>(__VA_ARGS__)

// Simplified version without NVSHMEM
#define SWITCH_RANKS(macro) \
    switch (num_ranks) { \
        case 1: macro(1); \
        case 2: macro(2); \
        case 4: macro(4); \
        case 8: macro(8); \
        case 16: macro(16); \
        case 32: macro(32); \
        case 64: macro(64); \
        default: throw std::runtime_error("Unsupported number of ranks"); \
    }

// Launch configuration struct
struct LaunchConfig {
    int num_blocks;
    int num_threads;
    cudaStream_t stream;
};

// Memory management for GTX 1650 with 4GB VRAM
namespace GTX1650 {
    // Calculate safe batch size based on model size and sequence length
    __host__ int calculateSafeBatchSize(size_t model_size_bytes, int seq_len, int head_dim) {
        // Get available memory
        size_t free_memory, total_memory;
        CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
        
        // Reserve 20% for workspace and other operations
        size_t available_memory = free_memory * 0.8;
        
        // Estimate memory per sequence:
        // - Input embedding: seq_len * head_dim * sizeof(half)
        // - KV cache: 2 * seq_len * head_dim * sizeof(half)
        // - Attention: seq_len * seq_len * sizeof(float) / 4 (with optimizations)
        // - Activations: ~4 * seq_len * head_dim * sizeof(half)
        size_t memory_per_seq = seq_len * head_dim * sizeof(half) * 7 + 
                                (seq_len * seq_len * sizeof(float)) / 4;
        
        // Calculate max batch size
        int max_batch = (available_memory - model_size_bytes) / memory_per_seq;
        return max(1, max_batch);
    }

    // Check if current operation would exceed memory limits
    __host__ bool checkMemoryLimits(size_t required_bytes) {
        size_t free_memory, total_memory;
        CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
        
        // Leave 10% buffer
        return (required_bytes <= free_memory * 0.9);
    }
}

// Simplified dummy structs to replace NVSHMEM functionality for compilation
namespace DummyNVSHMEM {
    typedef int nvshmem_team_t;
    typedef struct {
        int dummy;
    } nvshmemx_init_attr_t;
    
    const int NVSHMEM_TEAM_INVALID = -1;
    const int NVSHMEMX_INIT_WITH_UNIQUEID = 0;
    const int NVSHMEM_TEAM_WORLD = 0;
    
    typedef struct {
        int dummy;
    } nvshmemx_uniqueid_t;
    
    struct nvshmemi_device_host_state_t {
        bool ibgda_is_initialized;
    };
    
    // Dummy device state for compilation
    __device__ nvshmemi_device_host_state_t nvshmemi_device_state_d;
}