#pragma once

#include <string>
#include <stdexcept>
#include <sstream>

#include "configs.cuh"

namespace deep_ep {

// Custom exception class for CUDA errors
class EPException : public std::runtime_error {
public:
    EPException(const std::string& domain, const std::string& file, int line, const std::string& message)
        : std::runtime_error(formatMessage(domain, file, line, message)) {}

private:
    static std::string formatMessage(const std::string& domain, const std::string& file, int line, const std::string& message) {
        std::ostringstream oss;
        oss << "[" << domain << " Error] " << file << ":" << line << " - " << message;
        return oss.str();
    }
};

// CUDA error checking function
inline void checkCudaError(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        throw EPException("CUDA", file, line, cudaGetErrorString(e));
    }
}

// Macro for checking CUDA errors with file and line info
#define EP_CUDA_CHECK(expr) checkCudaError(expr, __FILE__, __LINE__)

// Memory allocation error checking
inline void* checkCudaMalloc(size_t size, const char* file, int line) {
    void* ptr = nullptr;
    cudaError_t status = cudaMalloc(&ptr, size);
    if (status != cudaSuccess) {
        throw EPException("CUDA Memory", file, line, 
                         std::string("Failed to allocate ") + std::to_string(size) + 
                         " bytes: " + cudaGetErrorString(status));
    }
    return ptr;
}

// Macro for checking CUDA memory allocation
#define EP_CUDA_MALLOC(size) checkCudaMalloc(size, __FILE__, __LINE__)

// Check for kernel launch errors
inline void checkKernelLaunch(const char* kernel_name, const char* file, int line) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw EPException("CUDA Kernel", file, line,
                         std::string("Failed to launch kernel ") + kernel_name + 
                         ": " + cudaGetErrorString(status));
    }
}

// Macro for checking kernel launches
#define EP_CHECK_KERNEL_LAUNCH(kernel_name) checkKernelLaunch(#kernel_name, __FILE__, __LINE__)

// GTX 1650 specific memory checker
inline void checkGTX1650Memory(size_t required_bytes, const char* file, int line) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    if (required_bytes > free * 0.9) {
        std::ostringstream oss;
        oss << "Not enough memory on GTX 1650: required " 
            << (required_bytes / (1024.0 * 1024.0)) << " MB, but only " 
            << (free / (1024.0 * 1024.0)) << " MB available out of " 
            << (total / (1024.0 * 1024.0)) << " MB total";
        
        throw EPException("GTX1650 Memory", file, line, oss.str());
    }
}

// Macro for GTX 1650 memory checking
#define EP_CHECK_GTX1650_MEMORY(bytes) checkGTX1650Memory(bytes, __FILE__, __LINE__)

} // namespace deep_ep