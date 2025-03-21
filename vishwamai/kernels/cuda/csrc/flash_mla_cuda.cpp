#include <torch/extension.h>
#include "flash_mla.h"
#include <vector>

// CUDA forward declarations
void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);

template<typename T, int Headdim>
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
torch::Tensor get_mla_metadata(
    torch::Tensor seqlens_k,
    int64_t batch_size,
    int64_t block_size_n,
    int64_t fixed_overhead_num_blocks,
    int64_t num_sm_parts
) {
    // Create output tensors
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(seqlens_k.device());
    auto tile_scheduler_metadata = torch::zeros({TileSchedulerMetaDataSize}, options);
    auto num_splits = torch::zeros({batch_size}, options);
    
    // Set up parameters for the CUDA function
    Mla_metadata_params params;
    params.seqlens_k_ptr = get_ptr<int>(seqlens_k);
    params.tile_scheduler_metadata_ptr = get_ptr<int>(tile_scheduler_metadata);
    params.num_splits_ptr = get_ptr<int>(num_splits);
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    
    // Get CUDA stream from current context
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Call the CUDA function
    get_mla_metadata_func(params, stream);
    
    // Return the computed metadata and num_splits
    return std::move(torch::stack({tile_scheduler_metadata, num_splits}, 0));
}

// Python wrapper for run_mha_fwd_splitkv_mla
std::vector<torch::Tensor> flash_mla_forward(
    torch::Tensor q,           // [b, seqlen_q, h, d]
    torch::Tensor k,           // [b, seqlen_k, h_k, k_d]
    torch::Tensor v,           // [b, seqlen_k, h_k, v_d]
    torch::Tensor block_table, // [batch_size, max_blocks_per_seq]
    torch::Tensor cu_seqlens_k,
    float scale_softmax,
    bool is_causal,
    torch::Tensor tile_scheduler_metadata,
    torch::Tensor num_splits,
    int num_sm_parts = 6
) {
    // Check input tensors
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(block_table);
    CHECK_INPUT(cu_seqlens_k);
    CHECK_INPUT(tile_scheduler_metadata);
    CHECK_INPUT(num_splits);
    
    // Extract dimensions
    int64_t batch_size = q.size(0);
    int64_t seqlen_q = q.size(1);
    int64_t num_heads = q.size(2);
    int64_t head_dim = q.size(3);
    int64_t num_heads_k = k.size(2);
    
    // Validate dimensions
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
               "FlashMLA only supports head dimensions 64 and 128");
    
    // Prepare output tensor
    auto out = torch::empty_like(q);
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, 
                                   torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    
    // Set up Flash_fwd_mla_params
    Flash_fwd_mla_params params;
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.d = head_dim;
    params.d_v = v.size(3);
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = 1;  // Default for now
    params.is_causal = is_causal;
    params.scale_softmax = scale_softmax;
    params.scale_softmax_log2 = log2f(scale_softmax);
    
    // Set up pointers
    params.q_ptr = get_tensor_ptr(q);
    params.k_ptr = get_tensor_ptr(k);
    params.v_ptr = get_tensor_ptr(v);
    params.o_ptr = get_tensor_ptr(out);
    params.softmax_lse_ptr = get_ptr<float>(softmax_lse);
    params.cu_seqlens_k = get_ptr<int>(cu_seqlens_k);
    
    // Set up strides
    // PyTorch default is [b, seqlen, h, d]
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    
    params.q_row_stride = q.stride(1);
    params.k_row_stride = k.stride(1);
    params.v_row_stride = v.stride(1);
    params.o_row_stride = out.stride(1);
    
    params.q_head_stride = q.stride(2);
    params.k_head_stride = k.stride(2);
    params.v_head_stride = v.stride(2);
    params.o_head_stride = out.stride(2);
    
    // Block table for KV-cache
    params.block_table = get_ptr<int>(block_table);
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = 64;  // Default block size
    
    // Scheduler metadata
    params.tile_scheduler_metadata_ptr = get_ptr<int>(tile_scheduler_metadata);
    params.num_sm_parts = num_sm_parts;
    params.num_splits_ptr = get_ptr<int>(num_splits);
    
    // Set up accumulators
    params.softmax_lseaccum_ptr = nullptr;  // Not used in forward pass
    params.oaccum_ptr = nullptr;            // Not used in forward pass
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Dispatch based on type and head dimension
    if (q.scalar_type() == at::ScalarType::Float) {
        if (head_dim == 64) {
            run_mha_fwd_splitkv_mla<float, 64>(params, stream);
        } else if (head_dim == 128) {
            run_mha_fwd_splitkv_mla<float, 128>(params, stream);
        } else {
            throw std::runtime_error("Unsupported head dimension for FlashMLA");
        }
    } else if (q.scalar_type() == at::ScalarType::Half) {
        if (head_dim == 64) {
            run_mha_fwd_splitkv_mla<at::Half, 64>(params, stream);
        } else if (head_dim == 128) {
            run_mha_fwd_splitkv_mla<at::Half, 128>(params, stream);
        } else {
            throw std::runtime_error("Unsupported head dimension for FlashMLA");
        }
    } else {
        throw std::runtime_error("Unsupported dtype for FlashMLA");
    }
    
    return {out, softmax_lse};
}

TORCH_LIBRARY(flash_mla_cuda, m) {
    m.def("get_mla_metadata", get_mla_metadata);
    m.def("flash_mla_forward", flash_mla_forward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_mla_metadata", &get_mla_metadata, 
          "Generate metadata for MLA scheduling",
          py::arg("seqlens_k"),
          py::arg("batch_size"),
          py::arg("block_size_n"),
          py::arg("fixed_overhead_num_blocks"),
          py::arg("num_sm_parts"));
          
    m.def("flash_mla_forward", &flash_mla_forward,
          "Forward pass for Flash Multi-head Linear Attention",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("block_table"),
          py::arg("cu_seqlens_k"),
          py::arg("scale_softmax"),
          py::arg("is_causal"),
          py::arg("tile_scheduler_metadata"),
          py::arg("num_splits"),
          py::arg("num_sm_parts") = 6);
}