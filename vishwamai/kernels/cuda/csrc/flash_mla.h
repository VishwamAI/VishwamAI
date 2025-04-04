#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int TileSchedulerMetaDataSize = 8;

struct Flash_fwd_mla_params {
    // Batch size and dimensions
    int64_t b;               // Batch size
    int64_t h;               // Number of heads
    int64_t seqlen_q;       // Query sequence length
    int64_t h_h_k_ratio;    // Query heads / Key heads ratio
    
    // Pointers to data
    void* q_ptr;            // Query tensor
    void* k_ptr;            // Key tensor
    void* v_ptr;            // Value tensor
    void* o_ptr;            // Output tensor
    float* softmax_lse_ptr; // Log-sum-exp values
    const int* cu_seqlens_k;// Cumulative sequence lengths
    const int* block_table; // Block table for KV cache
    const int* tile_scheduler_metadata_ptr;
    const int* num_splits_ptr;
    
    // Strides and dimensions
    int64_t q_batch_stride;
    int64_t q_row_stride;
    int64_t q_head_stride;
    int64_t k_batch_stride;
    int64_t k_row_stride;
    int64_t k_head_stride;
    int64_t v_batch_stride;
    int64_t v_row_stride;
    int64_t v_head_stride;
    int64_t o_batch_stride;
    int64_t o_row_stride;
    int64_t o_head_stride;
    
    // Other parameters
    int64_t page_block_size;
    int64_t block_table_batch_stride;
    double scale_softmax;
    bool is_causal;
};

struct Mla_metadata_params {
    const int* seqlens_k_ptr;
    int* tile_scheduler_metadata_ptr;
    int* num_splits_ptr;
    int64_t batch_size;
    int64_t block_size_n;
    int64_t fixed_overhead_num_blocks;
    int64_t num_sm_parts;
};

// CUDA kernel function declarations
template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream);

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);

// Helper macros for input validation
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")