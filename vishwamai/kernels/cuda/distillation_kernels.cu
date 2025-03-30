#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <vector>

#include "memory_manager.cuh"
#include "kernel_analyzer.cuh"
#include "configs.cuh"

namespace cg = cooperative_groups;

namespace deep_ep {
namespace cuda_kernels {

// Constants for GPU-specific optimizations
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Tensor dimensions for distillation
struct TensorDims {
    int batch_size;
    int seq_length;
    int num_heads;
    int head_dim;
};

// Helper functions for thread/block indexing
__device__ inline int get_thread_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int get_block_idx() {
    return blockIdx.x;
}

// CUDA kernel for teacher-student attention computation
template<typename T>
__global__ void compute_teacher_student_attention(
    const T* __restrict__ teacher_q,
    const T* __restrict__ teacher_k,
    const T* __restrict__ teacher_v,
    const T* __restrict__ student_q,
    const T* __restrict__ student_k,
    const T* __restrict__ student_v,
    T* __restrict__ output,
    T* __restrict__ attention_weights,
    const float temperature,
    const TensorDims dims,
    const bool use_flash = true
) {
    // Get thread indices
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const auto block = cg::this_thread_block();
    
    // Shared memory for intermediate computations
    extern __shared__ char shared_mem[];
    T* shared_q = reinterpret_cast<T*>(shared_mem);
    T* shared_k = shared_q + dims.head_dim;
    T* shared_scores = shared_k + dims.head_dim;
    
    // Calculate base indices
    const int batch_idx = bidx / (dims.num_heads * dims.seq_length);
    const int head_idx = (bidx / dims.seq_length) % dims.num_heads;
    const int seq_idx = bidx % dims.seq_length;
    
    // Load query vectors into shared memory
    if (tidx < dims.head_dim) {
        const int q_idx = ((batch_idx * dims.num_heads + head_idx) * 
                          dims.seq_length + seq_idx) * dims.head_dim + tidx;
        shared_q[tidx] = teacher_q[q_idx];
    }
    block.sync();
    
    // Compute attention scores
    const float scale = rsqrtf(static_cast<float>(dims.head_dim));
    
    for (int k_idx = tidx; k_idx < dims.seq_length; k_idx += block.size()) {
        // Load key vectors
        if (k_idx < dims.head_dim) {
            const int key_idx = ((batch_idx * dims.num_heads + head_idx) * 
                               dims.seq_length + k_idx) * dims.head_dim + tidx;
            shared_k[tidx] = teacher_k[key_idx];
        }
        block.sync();
        
        // Compute attention score
        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < dims.head_dim; d++) {
            score += static_cast<float>(shared_q[d]) * 
                    static_cast<float>(shared_k[d]);
        }
        score *= scale / temperature;
        
        // Store score
        if (tidx < dims.seq_length) {
            const int score_idx = ((batch_idx * dims.num_heads + head_idx) * 
                                 dims.seq_length + seq_idx) * dims.seq_length + k_idx;
            attention_weights[score_idx] = static_cast<T>(score);
        }
    }
    block.sync();
    
    // Apply softmax
    if (tidx < dims.seq_length) {
        // Find max for numerical stability
        float max_score = -INFINITY;
        #pragma unroll
        for (int i = 0; i < dims.seq_length; i++) {
            const int idx = ((batch_idx * dims.num_heads + head_idx) * 
                           dims.seq_length + seq_idx) * dims.seq_length + i;
            max_score = max(max_score, static_cast<float>(attention_weights[idx]));
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < dims.seq_length; i++) {
            const int idx = ((batch_idx * dims.num_heads + head_idx) * 
                           dims.seq_length + seq_idx) * dims.seq_length + i;
            const float val = expf(static_cast<float>(attention_weights[idx]) - max_score);
            attention_weights[idx] = static_cast<T>(val);
            sum += val;
        }
        
        // Normalize
        const float inv_sum = 1.0f / sum;
        #pragma unroll
        for (int i = 0; i < dims.seq_length; i++) {
            const int idx = ((batch_idx * dims.num_heads + head_idx) * 
                           dims.seq_length + seq_idx) * dims.seq_length + i;
            attention_weights[idx] = static_cast<T>(
                static_cast<float>(attention_weights[idx]) * inv_sum
            );
        }
    }
    block.sync();
    
    // Compute output
    if (tidx < dims.head_dim) {
        float out_val = 0.0f;
        #pragma unroll
        for (int i = 0; i < dims.seq_length; i++) {
            const int attn_idx = ((batch_idx * dims.num_heads + head_idx) * 
                                dims.seq_length + seq_idx) * dims.seq_length + i;
            const int v_idx = ((batch_idx * dims.num_heads + head_idx) * 
                             dims.seq_length + i) * dims.head_dim + tidx;
            out_val += static_cast<float>(attention_weights[attn_idx]) * 
                      static_cast<float>(teacher_v[v_idx]);
        }
        
        const int out_idx = ((batch_idx * dims.num_heads + head_idx) * 
                           dims.seq_length + seq_idx) * dims.head_dim + tidx;
        output[out_idx] = static_cast<T>(out_val);
    }
}

// CUDA kernel for attention loss computation
template<typename T>
__global__ void compute_attention_loss(
    const T* __restrict__ teacher_attn,
    const T* __restrict__ student_attn,
    T* __restrict__ loss,
    const float temperature,
    const TensorDims dims
) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    
    // Calculate indices
    const int batch_idx = bidx / (dims.num_heads * dims.seq_length);
    const int head_idx = (bidx / dims.seq_length) % dims.num_heads;
    const int seq_idx = bidx % dims.seq_length;
    
    // Compute KL divergence loss
    if (tidx < dims.seq_length) {
        const int base_idx = ((batch_idx * dims.num_heads + head_idx) * 
                            dims.seq_length + seq_idx) * dims.seq_length;
        
        float loss_val = 0.0f;
        const T* teacher_row = teacher_attn + base_idx;
        const T* student_row = student_attn + base_idx;
        
        #pragma unroll
        for (int i = 0; i < dims.seq_length; i++) {
            const float t_val = static_cast<float>(teacher_row[i]);
            const float s_val = static_cast<float>(student_row[i]);
            if (t_val > 0.0f) {
                loss_val += t_val * (logf(t_val + 1e-10f) - logf(s_val + 1e-10f));
            }
        }
        
        // Scale loss by temperature
        loss_val *= (temperature * temperature);
        
        // Store loss
        const int loss_idx = (batch_idx * dims.num_heads + head_idx) * 
                            dims.seq_length + seq_idx;
        loss[loss_idx] = static_cast<T>(loss_val);
    }
}

// Launch configuration helper
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    cudaStream_t stream;
};

// Helper function to get launch configuration
LaunchConfig get_launch_config(
    const TensorDims& dims,
    size_t shared_mem_per_block = 0
) {
    LaunchConfig config;
    
    // Calculate block size
    config.block.x = min(dims.head_dim, MAX_THREADS_PER_BLOCK);
    config.block.y = 1;
    config.block.z = 1;
    
    // Calculate grid size
    const int total_blocks = dims.batch_size * dims.num_heads * dims.seq_length;
    config.grid.x = total_blocks;
    config.grid.y = 1;
    config.grid.z = 1;
    
    // Set shared memory size
    config.shared_mem = shared_mem_per_block;
    
    // Use default stream
    config.stream = 0;
    
    return config;
}

// Class to manage distillation kernels
class DistillationKernels {
public:
    DistillationKernels() = default;
    
    // Initialize kernels
    void initialize(const TensorDims& dims) {
        dims_ = dims;
        
        // Initialize memory manager
        cuda_memory::MemoryPoolConfig pool_config{
            /* initial_pool_size= */ 1ull << 30,  // 1GB
            /* max_pool_size=    */ 1ull << 32,   // 4GB
            /* block_size=       */ 1ull << 20,   // 1MB
            /* growth_factor=    */ 1.5f,
            /* enable_defrag=    */ true,
            /* defrag_threshold= */ 0.3f
        };
        cuda_memory::CUDAMemoryManager::getInstance().initialize(pool_config);
        
        // Create analyzer
        analyzer_ = std::make_unique<cuda_analysis::KernelAnalyzer>();
    }
    
    // Compute teacher-student attention
    template<typename T>
    void compute_attention(
        const T* teacher_q,
        const T* teacher_k,
        const T* teacher_v,
        const T* student_q,
        const T* student_k,
        const T* student_v,
        T* output,
        T* attention_weights,
        float temperature,
        cudaStream_t stream = 0
    ) {
        // Get launch configuration
        const size_t shared_mem_size = 
            2 * dims_.head_dim * sizeof(T) + 
            dims_.seq_length * sizeof(float);
            
        auto config = get_launch_config(dims_, shared_mem_size);
        config.stream = stream;
        
        // Profile kernel
        analyzer_->profileKernelLaunch(
            "compute_teacher_student_attention",
            compute_teacher_student_attention<T>,
            teacher_q, teacher_k, teacher_v,
            student_q, student_k, student_v,
            output, attention_weights,
            temperature, dims_, true
        );
    }
    
    // Compute attention loss
    template<typename T>
    void compute_loss(
        const T* teacher_attn,
        const T* student_attn,
        T* loss,
        float temperature,
        cudaStream_t stream = 0
    ) {
        auto config = get_launch_config(dims_);
        config.stream = stream;
        
        // Profile kernel
        analyzer_->profileKernelLaunch(
            "compute_attention_loss",
            compute_attention_loss<T>,
            teacher_attn, student_attn,
            loss, temperature, dims_
        );
    }
    
    // Get performance metrics
    cuda_analysis::KernelMetrics get_metrics(const std::string& kernel_name) {
        return analyzer_->analyzeKernel(
            kernel_name,
            get_launch_config(dims_).grid,
            get_launch_config(dims_).block
        );
    }

private:
    TensorDims dims_;
    std::unique_ptr<cuda_analysis::KernelAnalyzer> analyzer_;
};

}  // namespace cuda_kernels
}  // namespace deep_ep
