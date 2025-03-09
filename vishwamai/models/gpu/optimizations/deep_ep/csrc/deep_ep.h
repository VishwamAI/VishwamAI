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

#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

struct DeepEPParams {
    // Buffer parameters
    void* hidden_buffer;
    void* nvlink_buffer;
    void* rdma_buffer;
    size_t hidden_bytes;
    size_t nvl_bytes;
    size_t rdma_bytes;

    // Expert dispatch parameters
    int num_experts;
    int num_tokens;
    int hidden_dim;
    bool async_dispatch;

    // Stream and event management
    cudaStream_t stream;
    cudaEvent_t event;
};

// Buffer management
cudaError_t deep_ep_init_buffer(DeepEPParams* params);
cudaError_t deep_ep_free_buffer(DeepEPParams* params);

// Expert dispatch/combine operations
cudaError_t deep_ep_dispatch(
    void* input,
    void* expert_indices,
    void* expert_weights,
    void* output,
    DeepEPParams* params
);

cudaError_t deep_ep_combine(
    void* expert_outputs,
    void* expert_indices,
    void* expert_weights,
    void* output,
    DeepEPParams* params
);

#ifdef __cplusplus
}
#endif
