#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "flash_mla.h"

// CUDA forward declarations
void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream);

// Helper to convert Python tensor to C++ pointer
template <typename T>
T* get_ptr(torch::Tensor &tensor) {
    return tensor.data_ptr<T>();
}

// Convert to right data pointer type based on tensor dtype
void* get_tensor_ptr(torch::Tensor &tensor) {
    if (tensor.scalar_type() == at::ScalarType::Half) {
        return static_cast<void*>(get_ptr<at::Half>(tensor));
    } else if (tensor.scalar_type() == at::ScalarType::Float) {
        return static_cast<void*>(get_ptr<float>(tensor));
    } else if (tensor.scalar_type() == at::ScalarType::BFloat16) {
        return static_cast<void*>(get_ptr<at::BFloat16>(tensor));
    } else {
        throw std::runtime_error("Unsupported tensor dtype");
    }
}

// Python wrapper for get_mla_metadata_func
at::Tensor get_mla_metadata(
    at::Tensor seqlens_k,
    int64_t batch_size,
    int64_t block_size_n,
    int64_t fixed_overhead_num_blocks,
    int64_t num_sm_parts
) {
    CHECK_CUDA(seqlens_k);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    Mla_metadata_params params{seqlens_k.data_ptr<int>()};
    get_mla_metadata_func(params, stream);
    
    return seqlens_k;
}

// Python wrapper for run_mha_fwd_splitkv_mla
std::vector<at::Tensor> flash_mla_forward(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor block_table,
    at::Tensor cu_seqlens,
    double scale,
    bool is_causal,
    at::Tensor tile_scheduler_metadata,
    at::Tensor num_splits,
    int64_t head_dim_value
) {
    // Input validation
    CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(v);
    CHECK_CUDA(block_table); CHECK_CUDA(cu_seqlens);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v);
    
    const int batch_size = q.size(0);
    const int seq_len_q = q.size(1);
    const int num_heads = q.size(2);
    const int head_dim = q.size(3);
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(q.dtype())
        .layout(torch::kStrided)
        .device(q.device())
        .requires_grad(q.requires_grad());
    
    auto output = torch::empty({batch_size, seq_len_q, num_heads, head_dim_value}, options);
    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len_q}, options.dtype(torch::kFloat32));
    
    // Create params struct
    Flash_fwd_mla_params params;
    params.b = batch_size;
    params.h = num_heads;
    params.seqlen_q = seq_len_q;
    params.scale_softmax = scale;
    params.is_causal = is_causal;
    // ... set other params ...
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Call CUDA implementation based on dtype
    if (q.dtype() == torch::kFloat32) {
        if (head_dim == 64) {
            run_mha_fwd_splitkv_mla<float, 64>(params, stream);
        } else if (head_dim == 128) {
            run_mha_fwd_splitkv_mla<float, 128>(params, stream);
        }
    } else if (q.dtype() == torch::kFloat16) {
        if (head_dim == 64) {
            run_mha_fwd_splitkv_mla<at::Half, 64>(params, stream);
        } else if (head_dim == 128) {
            run_mha_fwd_splitkv_mla<at::Half, 128>(params, stream);
        }
    }
    
    return {output, softmax_lse};
}

TORCH_LIBRARY(flash_mla_cuda, m) {
    m.def("flash_mla_forward", flash_mla_forward);
    m.def("get_mla_metadata", get_mla_metadata);
}