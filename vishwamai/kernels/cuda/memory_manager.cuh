#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>

#include "configs.cuh"
#include "kernel_analyzer.cuh"

namespace deep_ep {
namespace cuda_memory {

// Memory block metadata
struct MemoryBlock {
    void* ptr;               // Pointer to allocated memory
    size_t size;            // Size in bytes
    bool in_use;            // Whether block is currently in use
    cudaStream_t stream;    // Associated CUDA stream
    size_t last_access;     // Timestamp of last access
};

// Memory pool configuration
struct MemoryPoolConfig {
    size_t initial_pool_size;     // Initial size of memory pool
    size_t max_pool_size;         // Maximum size of memory pool
    size_t block_size;            // Size of memory blocks
    float growth_factor;          // Pool growth factor
    bool enable_defrag;           // Enable defragmentation
    size_t defrag_threshold;      // Memory fragmentation threshold
};

class CUDAMemoryManager {
public:
    // Singleton instance
    static CUDAMemoryManager& getInstance() {
        static CUDAMemoryManager instance;
        return instance;
    }

    // Initialize memory manager
    void initialize(const MemoryPoolConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
        
        // Initialize memory pool
        size_t total_memory, free_memory;
        CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
        
        pool_size_ = std::min(config.initial_pool_size, free_memory * 0.8);
        CUDA_CHECK(cudaMalloc(&pool_base_, pool_size_));
        
        // Initialize first block
        blocks_.push_back({
            pool_base_,
            pool_size_,
            false,
            0,
            0
        });
        
        // Create memory analyzer
        analyzer_ = std::make_unique<cuda_analysis::KernelAnalyzer>();
    }

    // Allocate memory
    void* allocate(
        size_t size,
        cudaStream_t stream = 0,
        bool async = false
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Align size to memory alignment requirements
        size = alignSize(size);
        
        // Try to find existing block
        for (auto& block : blocks_) {
            if (!block.in_use && block.size >= size) {
                // Split block if significantly larger
                if (block.size > size * 2 && block.size - size > config_.block_size) {
                    MemoryBlock new_block{
                        static_cast<char*>(block.ptr) + size,
                        block.size - size,
                        false,
                        stream,
                        0
                    };
                    block.size = size;
                    blocks_.push_back(new_block);
                }
                
                block.in_use = true;
                block.stream = stream;
                block.last_access = getCurrentTimestamp();
                
                // Track allocation
                allocations_[block.ptr] = &block;
                
                return block.ptr;
            }
        }
        
        // Need to grow pool
        if (shouldGrowPool(size)) {
            growPool(size);
            return allocate(size, stream, async);
        }
        
        // Try defragmentation
        if (config_.enable_defrag && shouldDefragment()) {
            defragment();
            return allocate(size, stream, async);
        }
        
        throw std::runtime_error("Out of memory");
    }

    // Free memory
    void free(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            throw std::runtime_error("Invalid pointer");
        }
        
        // Mark block as free
        it->second->in_use = false;
        allocations_.erase(it);
        
        // Merge adjacent free blocks
        mergeBlocks();
    }

    // Get memory usage statistics
    struct MemoryStats {
        size_t total_allocated;
        size_t total_free;
        size_t largest_free_block;
        size_t num_blocks;
        float fragmentation;
    };

    MemoryStats getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        MemoryStats stats{0, 0, 0, blocks_.size(), 0.0f};
        
        for (const auto& block : blocks_) {
            if (block.in_use) {
                stats.total_allocated += block.size;
            } else {
                stats.total_free += block.size;
                stats.largest_free_block = std::max(
                    stats.largest_free_block,
                    block.size
                );
            }
        }
        
        // Calculate fragmentation
        if (stats.total_free > 0) {
            stats.fragmentation = 1.0f - 
                (float)stats.largest_free_block / (float)stats.total_free;
        }
        
        return stats;
    }

    // Prefetch memory to GPU
    void prefetch(
        void* ptr,
        size_t size,
        cudaStream_t stream = 0
    ) {
        CUDA_CHECK(cudaMemPrefetchAsync(ptr, size, 0, stream));
    }

    // Memory access analysis
    template<typename T>
    void analyzeAccess(
        const T* data,
        size_t size,
        size_t stride,
        bool async = false
    ) {
        if (analyzer_) {
            analyzer_->analyzeMemoryPatterns(data, size, stride, async);
        }
    }

    // Cleanup
    ~CUDAMemoryManager() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_base_) {
            CUDA_CHECK(cudaFree(pool_base_));
            pool_base_ = nullptr;
        }
    }

private:
    // Private constructor for singleton
    CUDAMemoryManager() : pool_base_(nullptr), pool_size_(0) {}
    
    // Helper functions
    size_t alignSize(size_t size) const {
        return (size + EP_MEMORY_ALIGNMENT - 1) & ~(EP_MEMORY_ALIGNMENT - 1);
    }

    bool shouldGrowPool(size_t required_size) const {
        return pool_size_ < config_.max_pool_size &&
               pool_size_ + required_size <= config_.max_pool_size;
    }

    void growPool(size_t min_size) {
        size_t growth = std::max(
            min_size,
            static_cast<size_t>(pool_size_ * config_.growth_factor)
        );
        
        // Allocate new memory
        void* new_base;
        CUDA_CHECK(cudaMalloc(&new_base, pool_size_ + growth));
        
        // Copy existing memory
        CUDA_CHECK(cudaMemcpy(
            new_base,
            pool_base_,
            pool_size_,
            cudaMemcpyDeviceToDevice
        ));
        
        // Update pointers
        size_t offset = static_cast<char*>(new_base) - 
                       static_cast<char*>(pool_base_);
        
        for (auto& block : blocks_) {
            block.ptr = static_cast<char*>(block.ptr) + offset;
        }
        
        // Free old memory
        CUDA_CHECK(cudaFree(pool_base_));
        
        // Update pool information
        pool_base_ = new_base;
        
        // Add new block
        blocks_.push_back({
            static_cast<char*>(pool_base_) + pool_size_,
            growth,
            false,
            0,
            0
        });
        
        pool_size_ += growth;
    }

    bool shouldDefragment() const {
        MemoryStats stats = getStats();
        return stats.fragmentation > config_.defrag_threshold;
    }

    void defragment() {
        // Sort blocks by address
        std::sort(blocks_.begin(), blocks_.end(),
                 [](const MemoryBlock& a, const MemoryBlock& b) {
                     return a.ptr < b.ptr;
                 });
        
        // Compact memory
        void* current = pool_base_;
        for (auto& block : blocks_) {
            if (block.in_use) {
                if (block.ptr != current) {
                    // Move memory to new location
                    CUDA_CHECK(cudaMemcpy(
                        current,
                        block.ptr,
                        block.size,
                        cudaMemcpyDeviceToDevice
                    ));
                    
                    // Update pointer
                    void* old_ptr = block.ptr;
                    block.ptr = current;
                    
                    // Update allocations map
                    allocations_[block.ptr] = &block;
                    allocations_.erase(old_ptr);
                }
                current = static_cast<char*>(current) + block.size;
            }
        }
        
        // Create single free block at end
        size_t used_size = static_cast<char*>(current) - 
                          static_cast<char*>(pool_base_);
        
        blocks_.erase(
            std::remove_if(
                blocks_.begin(),
                blocks_.end(),
                [](const MemoryBlock& block) { return !block.in_use; }
            ),
            blocks_.end()
        );
        
        blocks_.push_back({
            current,
            pool_size_ - used_size,
            false,
            0,
            0
        });
    }

    void mergeBlocks() {
        // Sort blocks by address
        std::sort(blocks_.begin(), blocks_.end(),
                 [](const MemoryBlock& a, const MemoryBlock& b) {
                     return a.ptr < b.ptr;
                 });
        
        // Merge adjacent free blocks
        for (size_t i = 0; i < blocks_.size() - 1; ) {
            if (!blocks_[i].in_use && !blocks_[i + 1].in_use) {
                blocks_[i].size += blocks_[i + 1].size;
                blocks_.erase(blocks_.begin() + i + 1);
            } else {
                ++i;
            }
        }
    }

    size_t getCurrentTimestamp() const {
        return std::chrono::steady_clock::now()
               .time_since_epoch()
               .count();
    }

    // Member variables
    void* pool_base_;
    size_t pool_size_;
    MemoryPoolConfig config_;
    std::vector<MemoryBlock> blocks_;
    std::unordered_map<void*, MemoryBlock*> allocations_;
    std::unique_ptr<cuda_analysis::KernelAnalyzer> analyzer_;
    mutable std::mutex mutex_;
};

// Helper class for automatic memory management
template<typename T>
class CUDAMemoryPtr {
public:
    CUDAMemoryPtr(size_t size, cudaStream_t stream = 0)
        : size_(size)
        , stream_(stream)
    {
        ptr_ = static_cast<T*>(
            CUDAMemoryManager::getInstance().allocate(
                size * sizeof(T),
                stream
            )
        );
    }
    
    ~CUDAMemoryPtr() {
        if (ptr_) {
            CUDAMemoryManager::getInstance().free(ptr_);
        }
    }
    
    // No copy
    CUDAMemoryPtr(const CUDAMemoryPtr&) = delete;
    CUDAMemoryPtr& operator=(const CUDAMemoryPtr&) = delete;
    
    // Move operations
    CUDAMemoryPtr(CUDAMemoryPtr&& other)
        : ptr_(other.ptr_)
        , size_(other.size_)
        , stream_(other.stream_)
    {
        other.ptr_ = nullptr;
    }
    
    CUDAMemoryPtr& operator=(CUDAMemoryPtr&& other) {
        if (this != &other) {
            if (ptr_) {
                CUDAMemoryManager::getInstance().free(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            stream_ = other.stream_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    // Access
    T* get() const { return ptr_; }
    size_t size() const { return size_; }
    cudaStream_t stream() const { return stream_; }
    
    // Operators
    T& operator[](size_t index) { return ptr_[index]; }
    const T& operator[](size_t index) const { return ptr_[index]; }
    
    // Explicit conversion
    explicit operator T*() const { return ptr_; }

private:
    T* ptr_;
    size_t size_;
    cudaStream_t stream_;
};

}  // namespace cuda_memory
}  // namespace deep_ep
