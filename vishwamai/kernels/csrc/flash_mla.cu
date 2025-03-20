#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#include "flash_mla.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void compute_metadata_kernel(
    int* seqlens_k,
    int* tile_scheduler_metadata,
    int* num_splits,
    int batch_size,
    int block_size_n,
    int fixed_overhead_num_blocks,
    int num_sm_parts
) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    
    // 1 thread per batch item
    if (tidx >= batch_size) return;
    
    int seqlen_k = seqlens_k[tidx];
    
    // Calculate blocks needed
    int num_blocks = (seqlen_k + block_size_n - 1) / block_size_n;
    
    // Calculate this thread's position in the split
    const int split_idx = bidx * blockDim.x + tidx;
    
    // Store the number of splits at this position
    num_splits[split_idx] = num_blocks + fixed_overhead_num_blocks;
    
    // For the scheduler metadata, we compute:
    // [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]
    // Only compute for thread 0 in this simplified example
    if (tidx == 0) {
        // Just basic metadata setup for demonstration
        tile_scheduler_metadata[0] = 0;                 // begin_idx
        tile_scheduler_metadata[1] = 0;                 // begin_seqlen
        tile_scheduler_metadata[2] = batch_size - 1;    // end_idx
        tile_scheduler_metadata[3] = seqlen_k;          // end_seqlen
        tile_scheduler_metadata[4] = 0;                 // begin_n_split_idx
        // Other fields left as 0
    }
}

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream) {
    // Calculate grid and block dimensions
    dim3 grid(1);  
    dim3 block(params.batch_size);
    
    // Launch kernel
    compute_metadata_kernel<<<grid, block, 0, stream>>>(
        params.seqlens_k_ptr,
        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.batch_size,
        params.block_size_n,
        params.fixed_overhead_num_blocks,
        params.num_sm_parts
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in get_mla_metadata_func: " 
                  << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Common CUDA functions and utilities

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

__device__ __forceinline__ void atomicAddFloat(float* address, float val) {
    atomicAdd(address, val);
}

template <typename T>
__device__ __forceinline__ T block_reduce_max(T val) {
    constexpr int WARP_SIZE = 32;
    // Shared memory for block reduction
    __shared__ T shared[WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(0xffffffff, val, offset);
        val = max(val, other);
    }
    
    // Write result to shared memory
    if (lane == 0) shared[wid] = val;
    
    __syncthreads();
    
    // Final reduction from shared memory
    if (wid == 0) {
        val = (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared[lane] : -INFINITY;
        
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            T other = __shfl_xor_sync(0xffffffff, val, offset);
            val = max(val, other);
        }
    }
    
    // Broadcast to all threads
    val = __shfl_sync(0xffffffff, val, 0, WARP_SIZE);
    
    return val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// General template declaration for run_mha_fwd_splitkv_mla
template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

// Main FlashMLA kernel
template <typename T, int HEAD_DIM>
__global__ void flash_mla_forward_kernel(
    Flash_fwd_mla_params params, 
    int batch_size,
    int seqlen_q,
    int num_heads
) {
    using index_t = typename Flash_fwd_mla_params::index_t;
    
    // Get batch and head indices
    const int bi = blockIdx.z;
    const int hi = blockIdx.y;
    
    // First token position in query
    const int token_idx_q = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid_q = token_idx_q < seqlen_q;
    
    // Early exit if thread is out of bounds
    if (!valid_q || bi >= batch_size || hi >= num_heads) return;
    
    // Load query into registers
    T q_data[HEAD_DIM];
    const T* q_ptr = reinterpret_cast<const T*>(params.q_ptr);
    
    // Calculate q offset
    index_t q_offset = bi * params.q_batch_stride +
                      hi * params.q_head_stride +
                      token_idx_q * params.q_row_stride;
    
    // Load query
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        q_data[i] = q_ptr[q_offset + i];
    }
    
    // Get sequence length for this batch
    int seqlen_k = params.cu_seqlens_k ? params.cu_seqlens_k[bi + 1] - params.cu_seqlens_k[bi] : seqlen_q;
    
    // Compute scaling factor for attention scores
    float scale = params.scale_softmax;
    
    // Shared memory for block-level operations
    __shared__ float s_mean, s_max;
    
    // Initialize accumulation variables
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float softmax_denom = 0.0f;
    
    // Output accumulator (local partial sums)
    T output_accum[HEAD_DIM] = {0};
    
    // Process key-value pairs in blocks to maintain O(1) memory complexity
    for (int block_start_k = 0; block_start_k < seqlen_k; block_start_k += blockDim.x) {
        // Current k position 
        const int token_idx_k = block_start_k + threadIdx.x;
        const bool valid_k = token_idx_k < seqlen_k;
        
        // Apply causal masking - only attend to positions up to the current query position
        const bool causal_mask = params.is_causal ? (token_idx_k <= token_idx_q) : true;
        
        // Block-level processing for keys and values
        __shared__ T k_block[HEAD_DIM][32 + 1];  // +1 for bank conflict avoidance
        __shared__ T v_block[HEAD_DIM][32 + 1];
        
        // Load key into shared memory
        if (valid_k) {
            const T* k_ptr = reinterpret_cast<const T*>(params.k_ptr);
            index_t k_offset = bi * params.k_batch_stride +
                              hi * params.k_head_stride +
                              token_idx_k * params.k_row_stride;
            
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                k_block[i][threadIdx.x] = k_ptr[k_offset + i];
            }
            
            // Load value into shared memory
            const T* v_ptr = reinterpret_cast<const T*>(params.v_ptr);
            index_t v_offset = bi * params.v_batch_stride +
                              hi * params.v_head_stride +
                              token_idx_k * params.v_row_stride;
            
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; i++) {
                v_block[i][threadIdx.x] = v_ptr[v_offset + i];
            }
        }
        
        __syncthreads();
        
        // Compute attention scores and weighted values
        if (valid_q) {
            const int block_size = min(blockDim.x, seqlen_k - block_start_k);
            
            #pragma unroll 4
            for (int ki = 0; ki < block_size; ++ki) {
                const int token_idx_k = block_start_k + ki;
                
                // Skip if beyond causal mask
                if (params.is_causal && token_idx_k > token_idx_q) continue;
                
                // Compute attention score between q and k
                float score = 0.0f;
                #pragma unroll
                for (int i = 0; i < HEAD_DIM; ++i) {
                    score += static_cast<float>(q_data[i]) * static_cast<float>(k_block[i][ki]);
                }
                score *= scale;
                
                // Apply softmax scaling trick for numerical stability
                local_max = max(local_max, score);
                
                // Store score for softmax computation
                float exp_score = expf(score - local_max);
                softmax_denom += exp_score;
                
                // Compute weighted value
                #pragma unroll
                for (int i = 0; i < HEAD_DIM; ++i) {
                    output_accum[i] += exp_score * static_cast<float>(v_block[i][ki]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Normalize by softmax denominator
    if (valid_q) {
        // Output pointer
        T* output_ptr = reinterpret_cast<T*>(params.o_ptr);
        index_t o_offset = bi * params.o_batch_stride +
                          hi * params.o_head_stride +
                          token_idx_q * params.o_row_stride;
        
        // Calculate final values and write to output
        float softmax_denominator = softmax_denom;
        float inv_softmax_denominator = 1.0f / softmax_denominator;
        
        // Write log-sum-exp for backward pass if needed
        float* softmax_lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr);
        if (softmax_lse_ptr != nullptr) {
            softmax_lse_ptr[bi * seqlen_q * num_heads + hi * seqlen_q + token_idx_q] = 
                logf(softmax_denominator) + local_max;
        }
        
        // Write normalized output
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; ++i) {
            output_ptr[o_offset + i] = static_cast<T>(output_accum[i] * inv_softmax_denominator);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Template specialization for float and head dimension 64
template<>
void run_mha_fwd_splitkv_mla<float, 64>(Flash_fwd_mla_params &params, cudaStream_t stream) {
    static constexpr int HEAD_DIM = 64;
    using T = float;
    
    // Extract parameters
    int batch_size = params.b;
    int seqlen_q = params.seqlen_q;
    int num_heads = params.h;
    
    // Calculate grid and block dimensions
    constexpr int THREADS_PER_BLOCK = 32;  // Use one warp per block for simplicity
    dim3 grid(
        (seqlen_q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
        num_heads,
        batch_size
    );
    dim3 block(THREADS_PER_BLOCK);
    
    // Launch kernel
    flash_mla_forward_kernel<T, HEAD_DIM><<<grid, block, 0, stream>>>(
        params,
        batch_size,
        seqlen_q,
        num_heads
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in run_mha_fwd_splitkv_mla: " 
                  << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

// Template specialization for float16 and head dimension 64
template<>
void run_mha_fwd_splitkv_mla<half, 64>(Flash_fwd_mla_params &params, cudaStream_t stream) {
    static constexpr int HEAD_DIM = 64;
    using T = half;
    
    // Extract parameters
    int batch_size = params.b;
    int seqlen_q = params.seqlen_q;
    int num_heads = params.h;
    
    // Calculate grid and block dimensions
    constexpr int THREADS_PER_BLOCK = 32;
    dim3 grid(
        (seqlen_q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
        num_heads,
        batch_size
    );
    dim3 block(THREADS_PER_BLOCK);
    
    // Launch kernel
    flash_mla_forward_kernel<T, HEAD_DIM><<<grid, block, 0, stream>>>(
        params,
        batch_size,
        seqlen_q,
        num_heads
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in run_mha_fwd_splitkv_mla: " 
                  << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

// Additional template specializations for other head dimensions
template<>
void run_mha_fwd_splitkv_mla<float, 128>(Flash_fwd_mla_params &params, cudaStream_t stream) {
    static constexpr int HEAD_DIM = 128;
    using T = float;
    
    // Similar implementation as above, adjusted for HEAD_DIM = 128
    int batch_size = params.b;
    int seqlen_q = params.seqlen_q;
    int num_heads = params.h;
    
    constexpr int THREADS_PER_BLOCK = 32;
    dim3 grid(
        (seqlen_q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
        num_heads,
        batch_size
    );
    dim3 block(THREADS_PER_BLOCK);
    
    flash_mla_forward_kernel<T, HEAD_DIM><<<grid, block, 0, stream>>>(
        params,
        batch_size,
        seqlen_q,
        num_heads
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in run_mha_fwd_splitkv_mla: " 
                  << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

template<>
void run_mha_fwd_splitkv_mla<half, 128>(Flash_fwd_mla_params &params, cudaStream_t stream) {
    static constexpr int HEAD_DIM = 128;
    using T = half;
    
    // Similar implementation as above, adjusted for HEAD_DIM = 128 and half type
    int batch_size = params.b;
    int seqlen_q = params.seqlen_q;
    int num_heads = params.h;
    
    constexpr int THREADS_PER_BLOCK = 32;
    dim3 grid(
        (seqlen_q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
        num_heads,
        batch_size
    );
    dim3 block(THREADS_PER_BLOCK);
    
    flash_mla_forward_kernel<T, HEAD_DIM><<<grid, block, 0, stream>>>(
        params,
        batch_size,
        seqlen_q,
        num_heads
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in run_mha_fwd_splitkv_mla: " 
                  << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

// Add more specializations for other common head dimensions as needed (e.g., 32, 80, etc.)