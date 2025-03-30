#pragma once

#include <string>
#include <cuda_runtime.h>

namespace deep_ep {
namespace cuda {

// Version information
constexpr int MAJOR_VERSION = 1;
constexpr int MINOR_VERSION = 0;
constexpr int PATCH_VERSION = 0;

// Minimum required CUDA version
constexpr int MIN_CUDA_MAJOR = 11;
constexpr int MIN_CUDA_MINOR = 0;

// Minimum required compute capability
constexpr int MIN_COMPUTE_CAPABILITY_MAJOR = 7;
constexpr int MIN_COMPUTE_CAPABILITY_MINOR = 0;

// Feature flags
constexpr bool ENABLE_TENSOR_CORES = true;
constexpr bool ENABLE_COOPERATIVE_GROUPS = true;
constexpr bool ENABLE_DYNAMIC_PARALLELISM = true;

// Version string
inline std::string get_version_string() {
    return std::to_string(MAJOR_VERSION) + "." +
           std::to_string(MINOR_VERSION) + "." +
           std::to_string(PATCH_VERSION);
}

// CUDA version check
inline bool check_cuda_version() {
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    int major = runtime_version / 1000;
    int minor = (runtime_version % 1000) / 10;
    
    return (major > MIN_CUDA_MAJOR) ||
           (major == MIN_CUDA_MAJOR && minor >= MIN_CUDA_MINOR);
}

// Device capability check
inline bool check_device_capability(int device = 0) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    return (props.major > MIN_COMPUTE_CAPABILITY_MAJOR) ||
           (props.major == MIN_COMPUTE_CAPABILITY_MAJOR && 
            props.minor >= MIN_COMPUTE_CAPABILITY_MINOR);
}

// Feature support checks
struct DeviceFeatures {
    bool tensor_cores_supported;
    bool cooperative_groups_supported;
    bool dynamic_parallelism_supported;
    int max_shared_memory;
    int max_threads_per_block;
    int max_registers_per_block;
    int warp_size;
    
    static DeviceFeatures query(int device = 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        
        DeviceFeatures features;
        features.tensor_cores_supported = 
            props.major >= 7 || (props.major == 7 && props.minor >= 0);
        features.cooperative_groups_supported = 
            props.cooperativeLaunch && props.cooperativeMultiDeviceLaunch;
        features.dynamic_parallelism_supported = 
            props.major >= 3 || (props.major == 3 && props.minor >= 5);
        features.max_shared_memory = props.sharedMemPerBlock;
        features.max_threads_per_block = props.maxThreadsPerBlock;
        features.max_registers_per_block = props.regsPerBlock;
        features.warp_size = props.warpSize;
        
        return features;
    }
    
    std::string to_string() const {
        return "Device Features:\n"
               "  - Tensor Cores: " + std::string(tensor_cores_supported ? "Yes" : "No") + "\n"
               "  - Cooperative Groups: " + std::string(cooperative_groups_supported ? "Yes" : "No") + "\n"
               "  - Dynamic Parallelism: " + std::string(dynamic_parallelism_supported ? "Yes" : "No") + "\n"
               "  - Max Shared Memory: " + std::to_string(max_shared_memory) + " bytes\n"
               "  - Max Threads/Block: " + std::to_string(max_threads_per_block) + "\n"
               "  - Max Registers/Block: " + std::to_string(max_registers_per_block) + "\n"
               "  - Warp Size: " + std::to_string(warp_size);
    }
};

// Runtime feature configuration
struct RuntimeConfig {
    bool use_tensor_cores;
    bool use_cooperative_groups;
    bool use_dynamic_parallelism;
    int target_block_size;
    int shared_memory_limit;
    
    static RuntimeConfig create_default() {
        auto features = DeviceFeatures::query();
        return {
            /* use_tensor_cores= */ ENABLE_TENSOR_CORES && features.tensor_cores_supported,
            /* use_cooperative_groups= */ ENABLE_COOPERATIVE_GROUPS && features.cooperative_groups_supported,
            /* use_dynamic_parallelism= */ ENABLE_DYNAMIC_PARALLELISM && features.dynamic_parallelism_supported,
            /* target_block_size= */ 256,  // Default block size
            /* shared_memory_limit= */ features.max_shared_memory
        };
    }
    
    std::string to_string() const {
        return "Runtime Configuration:\n"
               "  - Using Tensor Cores: " + std::string(use_tensor_cores ? "Yes" : "No") + "\n"
               "  - Using Cooperative Groups: " + std::string(use_cooperative_groups ? "Yes" : "No") + "\n"
               "  - Using Dynamic Parallelism: " + std::string(use_dynamic_parallelism ? "Yes" : "No") + "\n"
               "  - Target Block Size: " + std::to_string(target_block_size) + "\n"
               "  - Shared Memory Limit: " + std::to_string(shared_memory_limit) + " bytes";
    }
};

// Error checking and reporting
class CudaError : public std::runtime_error {
public:
    CudaError(const char* msg, cudaError_t err)
        : std::runtime_error(std::string(msg) + ": " + 
                           cudaGetErrorString(err))
        , error_(err)
    {}
    
    cudaError_t error() const { return error_; }
    
private:
    cudaError_t error_;
};

// Version compatibility check
inline void ensure_compatibility() {
    // Check CUDA version
    if (!check_cuda_version()) {
        throw CudaError(
            "Unsupported CUDA version. Required: " +
            std::to_string(MIN_CUDA_MAJOR) + "." +
            std::to_string(MIN_CUDA_MINOR),
            cudaErrorInvalidConfiguration
        );
    }
    
    // Check device capability
    if (!check_device_capability()) {
        throw CudaError(
            "Unsupported device compute capability. Required: " +
            std::to_string(MIN_COMPUTE_CAPABILITY_MAJOR) + "." +
            std::to_string(MIN_COMPUTE_CAPABILITY_MINOR),
            cudaErrorInvalidConfiguration
        );
    }
}

}  // namespace cuda
}  // namespace deep_ep
