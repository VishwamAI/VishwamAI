#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

// Define missing types and macros
typedef uint32_t __be32;
typedef uint64_t __be64;

#define HtoBE32(x) __byte_perm(x, 0, 0x0123)
#define HtoBE64(x) __brevll(x)

#define memory_fence_cta() __threadfence_block()
#define st_na_relaxed(ptr, val) atomicAdd((unsigned int*)(ptr), val)
#define st_na_release(ptr, val) atomicExch((unsigned int*)(ptr), val)

#define EP_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#define GTX1650_WARP_SIZE 32
#define GTX1650_MAX_SHARED_MEMORY 49152  // 48 KB shared memory for GTX 1650

namespace deep_ep {

// Define NVSHMEM-like structures (simplified for compatibility)
struct nvshmemi_ibgda_device_key_t {
    uint32_t key;
    uint64_t next_addr;
};

struct nvshmemi_ibgda_device_qp_t {
    uint32_t qpn;
    struct {
        uint64_t *wqe;
        uint64_t *dbrec;
        uint64_t *bf;
        uint16_t nwqes;
        uint64_t prod_idx;
        uint64_t resv_head;
        uint64_t ready_head;
    } tx_wq;
    struct {
        void *buf;
        uint32_t lkey;
    } ibuf;
    struct {
        uint64_t prod_idx;
        int post_send_lock;
    } mvars;
};

struct nvshmemi_ibgda_device_state_t {
    uint64_t log2_cumem_granularity;
    struct {
        nvshmemi_ibgda_device_key_t *lkeys;
        nvshmemi_ibgda_device_key_t *rkeys;
    } constmem;
    struct {
        nvshmemi_ibgda_device_key_t *rkeys;
        nvshmemi_ibgda_device_qp_t *rcs;
    } globalmem;
    int num_rc_per_pe;
};

// Global state (simplified for compatibility)
__device__ nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;

// Helper functions
__device__ static __forceinline__
nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;
}

__device__ static __forceinline__
nvshmemi_ibgda_device_qp_t* ibgda_get_rc(int pe, int id) {
    auto state = ibgda_get_state();
    const auto num_rc_per_pe = state->num_rc_per_pe;
    return &state->globalmem.rcs[pe * num_rc_per_pe + id % num_rc_per_pe];
}

__device__ static __forceinline__
void ibgda_lock_acquire(int *lock) {
    while (atomicCAS(lock, 0, 1) == 1);
    memory_fence_cta();
}

__device__ static __forceinline__
void ibgda_lock_release(int *lock) {
    memory_fence_cta();
    st_na_relaxed(lock, 0);
}

// Flash Attention Kernel (optimized for GTX 1650)
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K>
__global__ void flash_attention_kernel(
    const half* __restrict__ query,      // [batch_size, num_heads, seq_len_q, head_dim]
    const half* __restrict__ key,        // [batch_size, num_heads, seq_len_k, head_dim]
    const half* __restrict__ value,      // [batch_size, num_heads, seq_len_k, head_dim]
    half* __restrict__ output,           // [batch_size, num_heads, seq_len_q, head_dim]
    const float scale,                   // Scale factor (1/sqrt(head_dim))
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim,
    const bool causal = true             // Whether to use causal attention
) {
    extern __shared__ half smem[];
    half* q_smem = smem;
    half* k_smem = q_smem + BLOCK_SIZE_M * BLOCK_SIZE_K;
    half* v_smem = k_smem + BLOCK_SIZE_K * BLOCK_SIZE_N;

    float O_tile[BLOCK_SIZE_M][BLOCK_SIZE_K] = {0.0f};
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_tile_idx = blockIdx.z;

    int q_offset = batch_idx * (num_heads * seq_len_q * head_dim) +
                   head_idx * (seq_len_q * head_dim) +
                   q_tile_idx * BLOCK_SIZE_M * head_dim;

    int kv_offset = batch_idx * (num_heads * seq_len_k * head_dim) +
                    head_idx * (seq_len_k * head_dim);

    // Load Q tile into shared memory
    for (int i = threadIdx.x; i < BLOCK_SIZE_M * BLOCK_SIZE_K; i += blockDim.x) {
        int m = i / BLOCK_SIZE_K;
        int k = i % BLOCK_SIZE_K;
        if (q_tile_idx * BLOCK_SIZE_M + m < seq_len_q && k < head_dim) {
            q_smem[m * BLOCK_SIZE_K + k] = query[q_offset + m * head_dim + k];
        } else {
            q_smem[m * BLOCK_SIZE_K + k] = 0;
        }
    }
    __syncthreads();

    // Process K, V tiles
    for (int k_tile_idx = 0; k_tile_idx < (seq_len_k + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++k_tile_idx) {
        if (causal && q_tile_idx * BLOCK_SIZE_M < k_tile_idx * BLOCK_SIZE_N) continue;

        // Load K, V tiles to shared memory
        for (int i = threadIdx.x; i < BLOCK_SIZE_K * BLOCK_SIZE_N; i += blockDim.x) {
            int k = i / BLOCK_SIZE_N;
            int n = i % BLOCK_SIZE_N;
            int k_idx = k_tile_idx * BLOCK_SIZE_N + n;
            if (k < head_dim && k_idx < seq_len_k) {
                k_smem[k * BLOCK_SIZE_N + n] = key[kv_offset + k_idx * head_dim + k];
                v_smem[n * BLOCK_SIZE_K + k] = value[kv_offset + k_idx * head_dim + k];
            } else {
                k_smem[k * BLOCK_SIZE_N + n] = 0;
                v_smem[n * BLOCK_SIZE_K + k] = 0;
            }
        }
        __syncthreads();

        // Compute attention for this tile
        for (int m = threadIdx.x / BLOCK_SIZE_K; m < BLOCK_SIZE_M; m += blockDim.x / BLOCK_SIZE_K) {
            int k = threadIdx.x % BLOCK_SIZE_K;
            if (q_tile_idx * BLOCK_SIZE_M + m >= seq_len_q) continue;

            float s[BLOCK_SIZE_N] = {0.0f};
            float m_cur = m_prev;
            float l = 0.0f;

            for (int n = 0; n < BLOCK_SIZE_N; ++n) {
                int k_idx = k_tile_idx * BLOCK_SIZE_N + n;
                if (k_idx >= seq_len_k || (causal && q_tile_idx * BLOCK_SIZE_M + m < k_idx)) continue;

                float qk = 0.0f;
                for (int d = 0; d < BLOCK_SIZE_K; d += 8) {
                    qk += __half2float(q_smem[m * BLOCK_SIZE_K + (k + d) % BLOCK_SIZE_K]) *
                          __half2float(k_smem[(k + d) % BLOCK_SIZE_K * BLOCK_SIZE_N + n]);
                }
                s[n] = qk * scale;
                m_cur = max(m_cur, s[n]);
            }

            for (int n = 0; n < BLOCK_SIZE_N; ++n) {
                int k_idx = k_tile_idx * BLOCK_SIZE_N + n;
                if (k_idx >= seq_len_k || (causal && q_tile_idx * BLOCK_SIZE_M + m < k_idx)) continue;

                float exp_s = exp(s[n] - m_cur);
                l += exp_s;
                float o_contribution = exp_s * __half2float(v_smem[n * BLOCK_SIZE_K + k]);
                O_tile[m][k] = O_tile[m][k] * exp(m_prev - m_cur) + o_contribution;
            }

            m_prev = m_cur;
            l_prev = l_prev * exp(m_prev - m_cur) + l;
        }
        __syncthreads();
    }

    // Normalize and write output
    for (int m = threadIdx.x / BLOCK_SIZE_K; m < BLOCK_SIZE_M; m += blockDim.x / BLOCK_SIZE_K) {
        int k = threadIdx.x % BLOCK_SIZE_K;
        if (q_tile_idx * BLOCK_SIZE_M + m < seq_len_q && k < head_dim) {
            float o = O_tile[m][k] / l_prev;
            output[q_offset + m * head_dim + k] = __float2half(o);
        }
    }
}

} // namespace deep_ep