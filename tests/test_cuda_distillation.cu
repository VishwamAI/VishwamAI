#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <memory>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../vishwamai/kernels/cuda/distillation_kernels.cu"
#include "../vishwamai/kernels/cuda/memory_manager.cuh"
#include "../vishwamai/kernels/cuda/kernel_analyzer.cuh"

using namespace deep_ep::cuda_kernels;
using namespace deep_ep::cuda_memory;

class CUDADistillationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        
        // Set test dimensions
        dims = {
            /* batch_size= */ 4,
            /* seq_length= */ 256,
            /* num_heads=  */ 12,
            /* head_dim=   */ 64
        };
        
        // Initialize memory manager
        MemoryPoolConfig pool_config{
            /* initial_pool_size= */ 1ull << 30,  // 1GB
            /* max_pool_size=    */ 1ull << 32,   // 4GB
            /* block_size=       */ 1ull << 20,   // 1MB
            /* growth_factor=    */ 1.5f,
            /* enable_defrag=    */ true,
            /* defrag_threshold= */ 0.3f
        };
        CUDAMemoryManager::getInstance().initialize(pool_config);
        
        // Initialize kernels
        kernels = std::make_unique<DistillationKernels>();
        kernels->initialize(dims);
        
        // Allocate test data
        allocateTestData();
    }

    void TearDown() override {
        // Free test data
        freeTestData();
    }
    
    void allocateTestData() {
        const size_t qkv_size = dims.batch_size * dims.num_heads * 
                               dims.seq_length * dims.head_dim;
        const size_t attn_size = dims.batch_size * dims.num_heads * 
                                dims.seq_length * dims.seq_length;
        
        // Allocate host memory
        h_teacher_q.resize(qkv_size);
        h_teacher_k.resize(qkv_size);
        h_teacher_v.resize(qkv_size);
        h_student_q.resize(qkv_size);
        h_student_k.resize(qkv_size);
        h_student_v.resize(qkv_size);
        h_output.resize(qkv_size);
        h_attention.resize(attn_size);
        h_loss.resize(dims.batch_size * dims.num_heads * dims.seq_length);
        
        // Initialize with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (size_t i = 0; i < qkv_size; ++i) {
            h_teacher_q[i] = dist(gen);
            h_teacher_k[i] = dist(gen);
            h_teacher_v[i] = dist(gen);
            h_student_q[i] = h_teacher_q[i] + dist(gen) * 0.1f;
            h_student_k[i] = h_teacher_k[i] + dist(gen) * 0.1f;
            h_student_v[i] = h_teacher_v[i] + dist(gen) * 0.1f;
        }
        
        // Allocate device memory
        d_teacher_q = CUDAMemoryPtr<float>(qkv_size);
        d_teacher_k = CUDAMemoryPtr<float>(qkv_size);
        d_teacher_v = CUDAMemoryPtr<float>(qkv_size);
        d_student_q = CUDAMemoryPtr<float>(qkv_size);
        d_student_k = CUDAMemoryPtr<float>(qkv_size);
        d_student_v = CUDAMemoryPtr<float>(qkv_size);
        d_output = CUDAMemoryPtr<float>(qkv_size);
        d_attention = CUDAMemoryPtr<float>(attn_size);
        d_loss = CUDAMemoryPtr<float>(dims.batch_size * dims.num_heads * dims.seq_length);
        
        // Copy data to device
        ASSERT_EQ(cudaMemcpy(d_teacher_q.get(), h_teacher_q.data(), 
                            qkv_size * sizeof(float), 
                            cudaMemcpyHostToDevice), 
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_teacher_k.get(), h_teacher_k.data(),
                            qkv_size * sizeof(float),
                            cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_teacher_v.get(), h_teacher_v.data(),
                            qkv_size * sizeof(float),
                            cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_student_q.get(), h_student_q.data(),
                            qkv_size * sizeof(float),
                            cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_student_k.get(), h_student_k.data(),
                            qkv_size * sizeof(float),
                            cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_student_v.get(), h_student_v.data(),
                            qkv_size * sizeof(float),
                            cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
    
    void freeTestData() {
        // Device memory is automatically freed by CUDAMemoryPtr destructors
    }
    
    // Helper function to compute CPU reference attention
    void computeReferenceAttention(
        float temperature,
        std::vector<float>& ref_output,
        std::vector<float>& ref_attention
    ) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));
        
        for (int b = 0; b < dims.batch_size; ++b) {
            for (int h = 0; h < dims.num_heads; ++h) {
                for (int q = 0; q < dims.seq_length; ++q) {
                    // Compute attention scores
                    std::vector<float> scores(dims.seq_length);
                    float max_score = -std::numeric_limits<float>::infinity();
                    
                    for (int k = 0; k < dims.seq_length; ++k) {
                        float score = 0.0f;
                        for (int d = 0; d < dims.head_dim; ++d) {
                            const int q_idx = ((b * dims.num_heads + h) * 
                                             dims.seq_length + q) * 
                                             dims.head_dim + d;
                            const int k_idx = ((b * dims.num_heads + h) * 
                                             dims.seq_length + k) * 
                                             dims.head_dim + d;
                            score += h_teacher_q[q_idx] * h_teacher_k[k_idx];
                        }
                        score *= scale / temperature;
                        scores[k] = score;
                        max_score = std::max(max_score, score);
                    }
                    
                    // Apply softmax
                    float sum = 0.0f;
                    for (int k = 0; k < dims.seq_length; ++k) {
                        scores[k] = std::exp(scores[k] - max_score);
                        sum += scores[k];
                    }
                    
                    const float inv_sum = 1.0f / sum;
                    for (int k = 0; k < dims.seq_length; ++k) {
                        const int attn_idx = ((b * dims.num_heads + h) * 
                                            dims.seq_length + q) * 
                                            dims.seq_length + k;
                        ref_attention[attn_idx] = scores[k] * inv_sum;
                    }
                    
                    // Compute weighted sum
                    for (int d = 0; d < dims.head_dim; ++d) {
                        float weighted_sum = 0.0f;
                        for (int k = 0; k < dims.seq_length; ++k) {
                            const int v_idx = ((b * dims.num_heads + h) * 
                                             dims.seq_length + k) * 
                                             dims.head_dim + d;
                            weighted_sum += ref_attention[((b * dims.num_heads + h) * 
                                                         dims.seq_length + q) * 
                                                         dims.seq_length + k] * 
                                          h_teacher_v[v_idx];
                        }
                        const int out_idx = ((b * dims.num_heads + h) * 
                                           dims.seq_length + q) * 
                                           dims.head_dim + d;
                        ref_output[out_idx] = weighted_sum;
                    }
                }
            }
        }
    }

    TensorDims dims;
    std::unique_ptr<DistillationKernels> kernels;
    
    // Host data
    std::vector<float> h_teacher_q;
    std::vector<float> h_teacher_k;
    std::vector<float> h_teacher_v;
    std::vector<float> h_student_q;
    std::vector<float> h_student_k;
    std::vector<float> h_student_v;
    std::vector<float> h_output;
    std::vector<float> h_attention;
    std::vector<float> h_loss;
    
    // Device data
    CUDAMemoryPtr<float> d_teacher_q;
    CUDAMemoryPtr<float> d_teacher_k;
    CUDAMemoryPtr<float> d_teacher_v;
    CUDAMemoryPtr<float> d_student_q;
    CUDAMemoryPtr<float> d_student_k;
    CUDAMemoryPtr<float> d_student_v;
    CUDAMemoryPtr<float> d_output;
    CUDAMemoryPtr<float> d_attention;
    CUDAMemoryPtr<float> d_loss;
};

TEST_F(CUDADistillationTest, TestTeacherStudentAttention) {
    const float temperature = 2.0f;
    
    // Run CUDA kernel
    kernels->compute_attention(
        d_teacher_q.get(),
        d_teacher_k.get(),
        d_teacher_v.get(),
        d_student_q.get(),
        d_student_k.get(),
        d_student_v.get(),
        d_output.get(),
        d_attention.get(),
        temperature
    );
    
    // Copy results back to host
    const size_t qkv_size = dims.batch_size * dims.num_heads * 
                           dims.seq_length * dims.head_dim;
    const size_t attn_size = dims.batch_size * dims.num_heads * 
                            dims.seq_length * dims.seq_length;
                            
    ASSERT_EQ(cudaMemcpy(h_output.data(), d_output.get(),
                        qkv_size * sizeof(float),
                        cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_attention.data(), d_attention.get(),
                        attn_size * sizeof(float),
                        cudaMemcpyDeviceToHost),
              cudaSuccess);
    
    // Compute reference results
    std::vector<float> ref_output(qkv_size);
    std::vector<float> ref_attention(attn_size);
    computeReferenceAttention(temperature, ref_output, ref_attention);
    
    // Verify results
    const float epsilon = 1e-4f;
    for (size_t i = 0; i < qkv_size; ++i) {
        EXPECT_NEAR(h_output[i], ref_output[i], epsilon)
            << "Output mismatch at index " << i;
    }
    
    for (size_t i = 0; i < attn_size; ++i) {
        EXPECT_NEAR(h_attention[i], ref_attention[i], epsilon)
            << "Attention weight mismatch at index " << i;
    }
    
    // Check performance metrics
    auto metrics = kernels->get_metrics("compute_teacher_student_attention");
    EXPECT_GT(metrics.occupancy, 0.5f)
        << "Low GPU occupancy: " << metrics.occupancy;
    EXPECT_GT(metrics.memory_throughput, 100.0f)
        << "Low memory throughput: " << metrics.memory_throughput << " GB/s";
}

TEST_F(CUDADistillationTest, TestAttentionLoss) {
    const float temperature = 2.0f;
    
    // Run CUDA kernel
    kernels->compute_loss(
        d_attention.get(),
        d_attention.get(),  // Use same attention for testing
        d_loss.get(),
        temperature
    );
    
    // Copy results back to host
    const size_t loss_size = dims.batch_size * dims.num_heads * dims.seq_length;
    ASSERT_EQ(cudaMemcpy(h_loss.data(), d_loss.get(),
                        loss_size * sizeof(float),
                        cudaMemcpyDeviceToHost),
              cudaSuccess);
    
    // Verify loss is zero when using same attention
    const float epsilon = 1e-5f;
    for (size_t i = 0; i < loss_size; ++i) {
        EXPECT_NEAR(h_loss[i], 0.0f, epsilon)
            << "Non-zero loss at index " << i;
    }
    
    // Check performance metrics
    auto metrics = kernels->get_metrics("compute_attention_loss");
    EXPECT_GT(metrics.occupancy, 0.5f)
        << "Low GPU occupancy: " << metrics.occupancy;
}

TEST_F(CUDADistillationTest, TestMemoryManager) {
    // Get initial memory stats
    auto initial_stats = CUDAMemoryManager::getInstance().getStats();
    
    // Allocate test memory
    const size_t test_size = 1 << 20;  // 1MB
    void* ptr1 = CUDAMemoryManager::getInstance().allocate(test_size);
    void* ptr2 = CUDAMemoryManager::getInstance().allocate(test_size);
    
    // Check memory stats after allocation
    auto alloc_stats = CUDAMemoryManager::getInstance().getStats();
    EXPECT_GE(alloc_stats.total_allocated, initial_stats.total_allocated + 2 * test_size)
        << "Memory allocation tracking error";
    
    // Free memory
    CUDAMemoryManager::getInstance().free(ptr1);
    CUDAMemoryManager::getInstance().free(ptr2);
    
    // Check memory stats after free
    auto final_stats = CUDAMemoryManager::getInstance().getStats();
    EXPECT_LE(final_stats.total_allocated, initial_stats.total_allocated + test_size)
        << "Memory not properly freed";
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
