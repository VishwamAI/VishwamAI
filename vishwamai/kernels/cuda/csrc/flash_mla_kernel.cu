#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "flash_mla.h"

namespace {

template<typename T>
__device__ __forceinline__ T warp_shfl_xor_sync(T var, int laneMask) {
    return __shfl_xor_sync(0xffffffff, var, laneMask);
}

// Compute MLA metadata for tiling/scheduling
__global__ void get_mla_metadata_kernel(
    const int* __restrict__ seqlens_k,
    int* __restrict__ tile_scheduler_metadata,
    int* __restrict__ num_splits,
    const int batch_size,
    const int block_size_n,
    const int fixed_overhead_num_blocks,
    const int num_sm_parts
) {
    if (threadIdx.x == 0) {
        // Initialize metadata array
        for (int i = 0; i < TileSchedulerMetaDataSize; i++) {
            tile_scheduler_metadata[i] = 0;
        }
        
        // Calculate begin/end sequence positions
        int max_seqlen = 0;
        for (int b = 0; b < batch_size; b++) {
            max_seqlen = max(max_seqlen, seqlens_k[b]);
            // Calculate splits based on sequence length and block size
            int num_blocks = (seqlens_k[b] + block_size_n - 1) / block_size_n;
            num_splits[b] = num_blocks + fixed_overhead_num_blocks;
        }
        
        // Store metadata
        tile_scheduler_metadata[0] = 0;                // begin_idx
        tile_scheduler_metadata[1] = 0;                // begin_seqlen
        tile_scheduler_metadata[2] = batch_size - 1;   // end_idx
        tile_scheduler_metadata[3] = max_seqlen;       // end_seqlen
        tile_scheduler_metadata[4] = 0;                // begin_n_split_idx
    }
}

template<typename T, int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_mla_forward_kernel(
    const Flash_fwd_mla_params params,
    const T* __restrict__ q,           // [B, Sq, H, D]
    const T* __restrict__ k,           // [B, Sk, Hk, D] 
    const T* __restrict__ v,           // [B, Sk, Hk, Dv]
    T* __restrict__ out,              // [B, Sq, H, Dv]
    float* __restrict__ softmax_lse,  // [B, H, Sq]
    const int* __restrict__ cu_seqlens_k,
    const int* __restrict__ block_table,
    const int* __restrict__ tile_scheduler_metadata
) {
    using acc_t = float;
    
    // Block indices
    const int batch_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int query_start = blockIdx.z * BLOCK_SIZE;
    
    // Thread indices
    const int thread_id = threadIdx.x;
    const int lane_id = thread_id % WARP_SIZE;
    const int warp_id = thread_id / WARP_SIZE;
    
    // Load sequence lengths
    const int seqlen_k = cu_seqlens_k[batch_id + 1] - cu_seqlens_k[batch_id];
    const int query_end = min(query_start + BLOCK_SIZE, params.seqlen_q);
    
    // Shared memory for K/V tile
    __shared__ T shared_k[BLOCK_SIZE][HEAD_DIM];
    __shared__ T shared_v[BLOCK_SIZE][HEAD_DIM];
    __shared__ float shared_max[BLOCK_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE];
    
    // Registers for accumulating results
    acc_t thread_max[BLOCK_SIZE / WARP_SIZE] = {-INFINITY};
    acc_t thread_sum[BLOCK_SIZE / WARP_SIZE] = {0.0f};
    
    // Process K/V cache in tiles
    for (int tile_start = 0; tile_start < seqlen_k; tile_start += BLOCK_SIZE) {
        const int tile_size = min(BLOCK_SIZE, seqlen_k - tile_start);
        
        // Load K/V tile into shared memory
        for (int i = thread_id; i < tile_size * HEAD_DIM; i += blockDim.x) {
            const int kv_idx = tile_start + (i / HEAD_DIM);
            const int kv_head = head_id / params.h_h_k_ratio;
            const int d = i % HEAD_DIM;
            
            // Get block offset from block table
            const int block_id = block_table[batch_id * params.block_table_batch_stride + kv_idx / params.page_block_size];
            const int offset = (block_id * params.page_block_size + kv_idx % params.page_block_size) * params.k_row_stride;
            
            shared_k[i / HEAD_DIM][d] = k[offset + kv_head * params.k_head_stride + d];
            shared_v[i / HEAD_DIM][d] = v[offset + kv_head * params.v_head_stride + d];
        }
        __syncthreads();
        
        // Process queries
        #pragma unroll
        for (int q_offset = 0; q_offset < BLOCK_SIZE / WARP_SIZE; q_offset++) {
            const int query_idx = query_start + warp_id + q_offset * (blockDim.x / WARP_SIZE);
            if (query_idx >= query_end) continue;
            
            // Load query
            acc_t q_local[HEAD_DIM];
            #pragma unroll
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                const int q_offset = batch_id * params.q_batch_stride + 
                                   query_idx * params.q_row_stride +
                                   head_id * params.q_head_stride + d;
                q_local[d] = static_cast<acc_t>(q[q_offset]);
            }
            
            // Compute attention scores and accumulate max
            #pragma unroll
            for (int k_idx = 0; k_idx < tile_size; k_idx++) {
                acc_t score = 0.0f;
                #pragma unroll
                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                    score += q_local[d] * static_cast<acc_t>(shared_k[k_idx][d]);
                }
                
                // Warp reduce to get final score
                #pragma unroll
                for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
                    score += warp_shfl_xor_sync(score, mask);
                }
                
                score *= params.scale_softmax;
                
                // Apply causal masking if needed
                if (params.is_causal && tile_start + k_idx > query_idx) {
                    score = -INFINITY;
                }
                
                // Update running max
                thread_max[q_offset] = max(thread_max[q_offset], score);
            }
        }
        __syncthreads();
        
        // Second pass: compute softmax and output
        #pragma unroll
        for (int q_offset = 0; q_offset < BLOCK_SIZE / WARP_SIZE; q_offset++) {
            const int query_idx = query_start + warp_id + q_offset * (blockDim.x / WARP_SIZE);
            if (query_idx >= query_end) continue;
            
            const acc_t max_score = thread_max[q_offset];
            acc_t sum = 0.0f;
            acc_t out_local[HEAD_DIM] = {0.0f};
            
            #pragma unroll
            for (int k_idx = 0; k_idx < tile_size; k_idx++) {
                acc_t score = 0.0f;
                #pragma unroll
                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                    score += static_cast<acc_t>(q[batch_id * params.q_batch_stride +
                                                query_idx * params.q_row_stride +
                                                head_id * params.q_head_stride + d]) *
                            static_cast<acc_t>(shared_k[k_idx][d]);
                }
                
                // Warp reduce
                #pragma unroll
                for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
                    score += warp_shfl_xor_sync(score, mask);
                }
                
                score *= params.scale_softmax;
                
                // Apply causal masking
                if (params.is_causal && tile_start + k_idx > query_idx) {
                    score = -INFINITY;
                }
                
                // Compute attention weight
                const acc_t weight = exp(score - max_score);
                sum += weight;
                
                // Accumulate weighted values
                #pragma unroll
                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                    out_local[d] += weight * static_cast<acc_t>(shared_v[k_idx][d]);
                }
            }
            
            // Store results
            if (lane_id == 0) {
                thread_sum[q_offset] = sum;
            }
            
            // Write output
            const acc_t rcp_sum = 1.0f / sum;
            #pragma unroll
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                const int out_idx = batch_id * params.o_batch_stride +
                                  query_idx * params.o_row_stride +
                                  head_id * params.o_head_stride + d;
                out[out_idx] = static_cast<T>(out_local[d] * rcp_sum);
            }
            
            // Store softmax normalization factors
            if (lane_id == 0) {
                softmax_lse[batch_id * params.h * params.seqlen_q +
                           head_id * params.seqlen_q +
                           query_idx] = max_score + log(sum);
            }
        }
        __syncthreads();
    }
}

} // namespace

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream) {
    const dim3 grid(1);
    const dim3 block(32);
    get_mla_metadata_kernel<<<grid, block, 0, stream>>>(
        params.seqlens_k_ptr,
        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.batch_size,
        params.block_size_n,
        params.fixed_overhead_num_blocks,
        params.num_sm_parts
    );
}

template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 128;
    constexpr int THREADS_PER_BLOCK = 256;
    
    const dim3 grid(params.b, params.h, (params.seqlen_q + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 block(THREADS_PER_BLOCK);
    
    flash_mla_forward_kernel<T, BLOCK_SIZE, Headdim><<<grid, block, 0, stream>>>(
        params,
        static_cast<T*>(params.q_ptr),
        static_cast<T*>(params.k_ptr),
        static_cast<T*>(params.v_ptr),
        static_cast<T*>(params.o_ptr),
        static_cast<float*>(params.softmax_lse_ptr),
        params.cu_seqlens_k,
        params.block_table,
        params.tile_scheduler_metadata_ptr
    );
}

// Explicit instantiations
template void run_mha_fwd_splitkv_mla<float, 64>(Flash_fwd_mla_params &params, cudaStream_t stream);
template void run_mha_fwd_splitkv_mla<float, 128>(Flash_fwd_mla_params &params, cudaStream_t stream);
template void run_mha_fwd_splitkv_mla<at::Half, 64>(Flash_fwd_mla_params &params, cudaStream_t stream);
template void run_mha_fwd_splitkv_mla<at::Half, 128>(Flash_fwd_mla_params &params, cudaStream_t stream);