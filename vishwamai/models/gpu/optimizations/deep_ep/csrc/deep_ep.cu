/*
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "deep_ep.h"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constants for dispatch kernel
namespace {
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
}

namespace deep_ep {
namespace kernels {

template<typename T>
__global__ void dispatch_kernel(
    const T* input,
    const int* expert_indices,
    const T* expert_weights,
    T* output,
    const int num_tokens,
    const int hidden_dim,
    const int num_experts
) {
    // Get thread block index and size
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hidden_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (token_idx >= num_tokens || hidden_idx >= hidden_dim) return;
    
    // Get expert assignment
    const int expert_idx = expert_indices[token_idx];
    const T weight = expert_weights[token_idx];
    
    // Calculate output offset for expert
    const size_t output_idx = expert_idx * num_tokens * hidden_dim + 
                             token_idx * hidden_dim + hidden_idx;
                             
    // Load and scale input value
    const T input_val = input[token_idx * hidden_dim + hidden_idx];
    output[output_idx] = input_val * weight;
}

template<typename T>
__global__ void combine_kernel(
    const T* expert_outputs,
    const int* expert_indices,
    const T* expert_weights,
    T* output,
    const int num_tokens,
    const int hidden_dim,
    const int num_experts
) {
    // Get indices
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hidden_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (token_idx >= num_tokens || hidden_idx >= hidden_dim) return;
    
    // Accumulate expert outputs
    T acc = T(0);
    #pragma unroll
    for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        if (expert_indices[token_idx * num_experts + expert_idx] >= 0) {
            const size_t expert_offset = expert_idx * num_tokens * hidden_dim +
                                       token_idx * hidden_dim + hidden_idx;
            const T weight = expert_weights[token_idx * num_experts + expert_idx];
            acc += expert_outputs[expert_offset] * weight;
        }
    }
    
    // Write final output
    output[token_idx * hidden_dim + hidden_idx] = acc;
}

} // namespace kernels
} // namespace deep_ep

// Template instantiations
template __global__ void deep_ep::kernels::dispatch_kernel<float>(
    const float*, const int*, const float*, float*, const int, const int, const int);
template __global__ void deep_ep::kernels::combine_kernel<float>(
    const float*, const int*, const float*, float*, const int, const int, const int);

#endif // __CUDACC__

extern "C" {

cudaError_t deep_ep_init_buffer(DeepEPParams* params) {
    cudaError_t err = cudaSuccess;
    
    if (params->hidden_bytes > 0) {
        err = cudaMalloc(&params->hidden_buffer, params->hidden_bytes);
        if (err != cudaSuccess) return err;
    }
    if (params->nvl_bytes > 0) {
        err = cudaMalloc(&params->nvlink_buffer, params->nvl_bytes);
        if (err != cudaSuccess) return err;
    }
    if (params->rdma_bytes > 0) {
        err = cudaMalloc(&params->rdma_buffer, params->rdma_bytes);
        if (err != cudaSuccess) return err;
    }
    
    // Create CUDA stream if not provided
    if (params->stream == nullptr) {
        err = cudaStreamCreate(&params->stream);
    }
    
    return err;
}

cudaError_t deep_ep_free_buffer(DeepEPParams* params) {
    cudaError_t err = cudaSuccess;
    
    if (params->hidden_buffer) {
        err = cudaFree(params->hidden_buffer);
        if (err != cudaSuccess) return err;
    }
    if (params->nvlink_buffer) {
        err = cudaFree(params->nvlink_buffer);
        if (err != cudaSuccess) return err;
    }
    if (params->rdma_buffer) {
        err = cudaFree(params->rdma_buffer);
        if (err != cudaSuccess) return err;
    }
    
    // Destroy stream if we created it
    if (params->stream != nullptr) {
        err = cudaStreamDestroy(params->stream);
    }
    
    return err;
}

cudaError_t deep_ep_dispatch(
    void* input,
    void* expert_indices, 
    void* expert_weights,
    void* output,
    DeepEPParams* params
) {
#if defined(__CUDACC__)
    // Calculate grid and block dimensions
    dim3 block_dim(WARP_SIZE, WARP_SIZE/2);
    dim3 grid_dim(
        (params->num_tokens + block_dim.x - 1) / block_dim.x,
        (params->hidden_dim + block_dim.y - 1) / block_dim.y
    );
    
    // Launch dispatch kernel
    deep_ep::kernels::dispatch_kernel<float><<<grid_dim, block_dim, 0, params->stream>>>(
        static_cast<float*>(input),
        static_cast<int*>(expert_indices),
        static_cast<float*>(expert_weights),
        static_cast<float*>(output),
        params->num_tokens,
        params->hidden_dim,
        params->num_experts
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    if (params->event != nullptr) {
        err = cudaEventRecord(params->event, params->stream);
    }
    
    return err;
#else
    return cudaErrorNotSupported;
#endif
}

cudaError_t deep_ep_combine(
    void* expert_outputs,
    void* expert_indices,
    void* expert_weights,
    void* output,
    DeepEPParams* params
) {
#if defined(__CUDACC__)
    // Calculate grid and block dimensions
    dim3 block_dim(WARP_SIZE, WARP_SIZE/2);
    dim3 grid_dim(
        (params->num_tokens + block_dim.x - 1) / block_dim.x,
        (params->hidden_dim + block_dim.y - 1) / block_dim.y
    );
    
    // Launch combine kernel
    deep_ep::kernels::combine_kernel<float><<<grid_dim, block_dim, 0, params->stream>>>(
        static_cast<float*>(expert_outputs),
        static_cast<int*>(expert_indices),
        static_cast<float*>(expert_weights),
        static_cast<float*>(output),
        params->num_tokens,
        params->hidden_dim,
        params->num_experts
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    if (params->event != nullptr) {
        err = cudaEventRecord(params->event, params->stream);
    }
    
    return err;
#else
    return cudaErrorNotSupported;
#endif
}

} // extern "C"
