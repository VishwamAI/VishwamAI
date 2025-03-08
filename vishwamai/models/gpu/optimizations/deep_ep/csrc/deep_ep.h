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
