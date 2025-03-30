#pragma once

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>

#include "configs.cuh"
#include "utils.cuh"

namespace deep_ep {
namespace cuda_analysis {

struct KernelMetrics {
    float occupancy;                // Achieved occupancy
    float sm_efficiency;           // SM efficiency percentage
    float memory_throughput;       // Memory throughput in GB/s
    float memory_utilization;      // Memory utilization percentage
    float cache_hit_rate;          // L2 cache hit rate
    float warp_efficiency;         // Warp execution efficiency
    size_t shared_memory_used;     // Shared memory usage in bytes
    size_t register_per_thread;    // Registers used per thread
};

class KernelAnalyzer {
public:
    KernelAnalyzer() {
        // Initialize CUDA events for timing
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));
    }

    ~KernelAnalyzer() {
        CUDA_CHECK(cudaEventDestroy(start_event_));
        CUDA_CHECK(cudaEventDestroy(stop_event_));
    }

    // Profile kernel launch
    template<typename F, typename... Args>
    float profileKernelLaunch(const std::string& kernel_name, F kernel_func, Args... args) {
        // Start NVTX range for visual profiling
        nvtxRangePushA(kernel_name.c_str());
        
        // Record start time
        CUDA_CHECK(cudaEventRecord(start_event_));
        
        // Launch kernel
        kernel_func(args...);
        
        // Record stop time
        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
        
        // Calculate elapsed time
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
        
        // End NVTX range
        nvtxRangePop();
        
        // Store metrics
        kernel_timings_[kernel_name].push_back(elapsed_ms);
        
        return elapsed_ms;
    }

    // Analyze kernel occupancy
    template<typename KernelFunc>
    KernelMetrics analyzeKernel(
        KernelFunc kernel,
        dim3 grid_dim,
        dim3 block_dim,
        size_t shared_mem = 0
    ) {
        KernelMetrics metrics;
        
        // Get device properties
        cudaDeviceProp device_props;
        CUDA_CHECK(cudaGetDeviceProperties(&device_props, 0));
        
        // Calculate theoretical occupancy
        int max_blocks_per_sm;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            kernel,
            block_dim.x * block_dim.y * block_dim.z,
            shared_mem
        ));
        
        // Calculate achieved occupancy
        metrics.occupancy = static_cast<float>(max_blocks_per_sm) * 
                          static_cast<float>(block_dim.x * block_dim.y * block_dim.z) /
                          static_cast<float>(device_props.maxThreadsPerMultiProcessor);
        
        // Get register usage
        cudaFuncAttributes func_attrs;
        CUDA_CHECK(cudaFuncGetAttributes(&func_attrs, kernel));
        metrics.register_per_thread = func_attrs.numRegs;
        
        // Store shared memory usage
        metrics.shared_memory_used = shared_mem;
        
        return metrics;
    }

    // Memory analysis
    struct MemoryMetrics {
        size_t total_memory;
        size_t free_memory;
        size_t peak_memory;
        std::vector<size_t> allocation_sizes;
    };

    MemoryMetrics analyzeMemoryUsage() {
        MemoryMetrics metrics;
        
        // Get current memory usage
        CUDA_CHECK(cudaMemGetInfo(
            &metrics.free_memory,
            &metrics.total_memory
        ));
        
        // Calculate peak memory usage
        metrics.peak_memory = metrics.total_memory - metrics.free_memory;
        
        return metrics;
    }

    // Analyze memory access patterns
    template<typename T>
    void analyzeMemoryPatterns(
        const T* data,
        size_t size,
        size_t stride,
        bool async = false
    ) {
        std::vector<float> access_times;
        cudaStream_t stream = async ? cuda_stream_ : 0;
        
        // Measure sequential access
        float seq_time = measureMemoryAccess(
            data, size, stride, stream, false
        );
        
        // Measure strided access
        float strided_time = measureMemoryAccess(
            data, size, stride, stream, true
        );
        
        // Store results
        memory_patterns_["sequential"].push_back(seq_time);
        memory_patterns_["strided"].push_back(strided_time);
    }

    // Get kernel timing statistics
    struct TimingStats {
        float mean;
        float min;
        float max;
        float stddev;
    };

    TimingStats getKernelStats(const std::string& kernel_name) {
        const auto& timings = kernel_timings_[kernel_name];
        if (timings.empty()) {
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }
        
        // Calculate statistics
        float sum = 0.0f;
        float min = timings[0];
        float max = timings[0];
        
        for (float t : timings) {
            sum += t;
            min = std::min(min, t);
            max = std::max(max, t);
        }
        
        float mean = sum / timings.size();
        
        // Calculate standard deviation
        float variance_sum = 0.0f;
        for (float t : timings) {
            float diff = t - mean;
            variance_sum += diff * diff;
        }
        float stddev = std::sqrt(variance_sum / timings.size());
        
        return {mean, min, max, stddev};
    }

    // Generate analysis report
    std::string generateReport() const {
        std::string report = "CUDA Kernel Analysis Report\n";
        report += "========================\n\n";
        
        // Add kernel timing statistics
        report += "Kernel Timing Statistics:\n";
        for (const auto& kernel : kernel_timings_) {
            report += "  " + kernel.first + ":\n";
            auto stats = getStatsForKernel(kernel.second);
            report += "    Mean: " + std::to_string(stats.mean) + " ms\n";
            report += "    Min:  " + std::to_string(stats.min) + " ms\n";
            report += "    Max:  " + std::to_string(stats.max) + " ms\n";
            report += "    StdDev: " + std::to_string(stats.stddev) + " ms\n\n";
        }
        
        // Add memory access patterns
        report += "Memory Access Patterns:\n";
        for (const auto& pattern : memory_patterns_) {
            report += "  " + pattern.first + ":\n";
            auto stats = getStatsForPattern(pattern.second);
            report += "    Mean Access Time: " + std::to_string(stats.mean) + " ms\n";
            report += "    Bandwidth: " + std::to_string(calculateBandwidth(stats.mean)) + " GB/s\n\n";
        }
        
        return report;
    }

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    cudaStream_t cuda_stream_;
    
    // Storage for metrics
    std::unordered_map<std::string, std::vector<float>> kernel_timings_;
    std::unordered_map<std::string, std::vector<float>> memory_patterns_;

    // Helper to measure memory access time
    template<typename T>
    float measureMemoryAccess(
        const T* data,
        size_t size,
        size_t stride,
        cudaStream_t stream,
        bool is_strided
    ) {
        float elapsed_ms;
        
        // Create events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Record start time
        CUDA_CHECK(cudaEventRecord(start, stream));
        
        // Launch memory access kernel
        if (is_strided) {
            measureStridedAccess<<<grid_dim_, block_dim_, 0, stream>>>(
                data, size, stride
            );
        } else {
            measureSequentialAccess<<<grid_dim_, block_dim_, 0, stream>>>(
                data, size
            );
        }
        
        // Record stop time
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        
        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        return elapsed_ms;
    }

    // CUDA kernels for memory access measurement
    template<typename T>
    __global__ void measureSequentialAccess(const T* data, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            // Prevent compiler optimization
            volatile T value = data[idx];
        }
    }

    template<typename T>
    __global__ void measureStridedAccess(const T* data, size_t size, size_t stride) {
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
        if (idx < size) {
            // Prevent compiler optimization
            volatile T value = data[idx];
        }
    }

    // Helper functions for statistics
    static TimingStats getStatsForKernel(const std::vector<float>& timings) {
        TimingStats stats;
        if (timings.empty()) return stats;
        
        stats.mean = std::accumulate(timings.begin(), timings.end(), 0.0f) / timings.size();
        stats.min = *std::min_element(timings.begin(), timings.end());
        stats.max = *std::max_element(timings.begin(), timings.end());
        
        float variance = 0.0f;
        for (float t : timings) {
            variance += (t - stats.mean) * (t - stats.mean);
        }
        stats.stddev = std::sqrt(variance / timings.size());
        
        return stats;
    }

    // Calculate memory bandwidth
    static float calculateBandwidth(float time_ms, size_t bytes_transferred) {
        return (bytes_transferred / (1024.0f * 1024.0f * 1024.0f)) / (time_ms / 1000.0f);
    }

    // Default grid and block dimensions
    const dim3 grid_dim_{256, 1, 1};
    const dim3 block_dim_{256, 1, 1};
};

}  // namespace cuda_analysis
}  // namespace deep_ep
