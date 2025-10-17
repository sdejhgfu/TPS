#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <atomic>

namespace hpie {

// Memory page size (4KB for optimal cache line utilization)
constexpr size_t PAGE_SIZE = 4096;
constexpr size_t CACHE_LINE_SIZE = 64;

// Memory allocation strategy
enum class AllocationStrategy {
    FIRST_FIT,
    BEST_FIT,
    WORST_FIT,
    BUDDY_SYSTEM
};

// Memory statistics
struct MemoryStats {
    size_t total_memory;
    size_t used_memory;
    size_t free_memory;
    size_t fragmented_memory;
    size_t num_allocations;
    size_t num_deallocations;
    double fragmentation_ratio;
    size_t cache_misses;
    size_t cache_hits;
};

// Memory allocation header
struct MemoryBlock {
    void* ptr;
    size_t size;
    size_t alignment;
    bool is_free;
    MemoryBlock* next;
    MemoryBlock* prev;
    uint32_t magic_number; // For debugging
};

// Memory pool for fast allocations
class MemoryPool {
public:
    explicit MemoryPool(size_t pool_size, size_t block_size);
    ~MemoryPool();

    void* Allocate();
    void Deallocate(void* ptr);
    bool Contains(void* ptr) const;
    size_t GetFreeBlocks() const { return free_blocks_; }
    size_t GetTotalBlocks() const { return total_blocks_; }

private:
    void* pool_memory_;
    size_t pool_size_;
    size_t block_size_;
    size_t total_blocks_;
    size_t free_blocks_;
    void* free_list_head_;
    std::mutex mutex_;
};

// Page manager for large allocations
class PageManager {
public:
    explicit PageManager(size_t max_pages);
    ~PageManager();

    void* AllocatePages(size_t num_pages);
    void DeallocatePages(void* ptr, size_t num_pages);
    void* AllocatePage();
    void DeallocatePage(void* ptr);
    
    size_t GetUsedPages() const { return used_pages_; }
    size_t GetFreePages() const { return total_pages_ - used_pages_; }

private:
    std::vector<void*> pages_;
    std::vector<bool> page_used_;
    size_t total_pages_;
    size_t used_pages_;
    std::mutex mutex_;
};

// Main memory manager
class MemoryManager {
public:
    explicit MemoryManager(size_t max_memory = SIZE_MAX);
    ~MemoryManager();

    // Core allocation functions
    void* Allocate(size_t size, size_t alignment = alignof(max_align_t));
    void Deallocate(void* ptr);
    void* Reallocate(void* ptr, size_t new_size);

    // Memory management
    void Defragment();
    void Clear();
    void SetAllocationStrategy(AllocationStrategy strategy);

    // Statistics and monitoring
    MemoryStats GetStats() const;
    size_t GetUsedMemory() const { return used_memory_; }
    size_t GetTotalMemory() const { return total_memory_; }
    
    // Memory optimization
    void OptimizeMemoryLayout();
    void PrefetchMemory(void* ptr, size_t size);
    void FlushCache(void* ptr, size_t size);

    // Alignment utilities
    static size_t AlignUp(size_t size, size_t alignment);
    static bool IsAligned(void* ptr, size_t alignment);

private:
    // Memory pools for different sizes
    std::vector<std::unique_ptr<MemoryPool>> small_pools_;
    std::unique_ptr<PageManager> page_manager_;
    
    // Allocation strategy
    AllocationStrategy strategy_;
    
    // Memory tracking
    std::unordered_map<void*, MemoryBlock> allocations_;
    std::vector<MemoryBlock*> free_blocks_;
    
    // Statistics
    std::atomic<size_t> total_memory_;
    std::atomic<size_t> used_memory_;
    std::atomic<size_t> num_allocations_;
    std::atomic<size_t> num_deallocations_;
    std::atomic<size_t> cache_hits_;
    std::atomic<size_t> cache_misses_;
    
    // Thread safety
    mutable std::mutex mutex_;
    
    // Internal methods
    void* AllocateFromPool(size_t size);
    void* AllocateFromHeap(size_t size, size_t alignment);
    MemoryBlock* FindFreeBlock(size_t size);
    void SplitBlock(MemoryBlock* block, size_t size);
    void MergeFreeBlocks();
    
    // Buddy system implementation
    void* BuddyAllocate(size_t size);
    void BuddyDeallocate(void* ptr, size_t size);
    size_t GetBuddySize(size_t size);
    
    // Memory prefetching and optimization
    void PrefetchBlock(const MemoryBlock& block);
    void OptimizeBlockLayout(MemoryBlock* block);
};

// RAII wrapper for automatic memory management
template<typename T>
class MemoryGuard {
public:
    explicit MemoryGuard(MemoryManager* manager, size_t count = 1)
        : manager_(manager), ptr_(nullptr), count_(count) {
        if (manager_) {
            ptr_ = static_cast<T*>(manager_->Allocate(count * sizeof(T), alignof(T)));
        }
    }

    ~MemoryGuard() {
        if (manager_ && ptr_) {
            manager_->Deallocate(ptr_);
        }
    }

    T* Get() const { return ptr_; }
    T& operator[](size_t index) { return ptr_[index]; }
    const T& operator[](size_t index) const { return ptr_[index]; }

    // Disable copy
    MemoryGuard(const MemoryGuard&) = delete;
    MemoryGuard& operator=(const MemoryGuard&) = delete;

    // Enable move
    MemoryGuard(MemoryGuard&& other) noexcept
        : manager_(other.manager_), ptr_(other.ptr_), count_(other.count_) {
        other.manager_ = nullptr;
        other.ptr_ = nullptr;
    }

    MemoryGuard& operator=(MemoryGuard&& other) noexcept {
        if (this != &other) {
            if (manager_ && ptr_) {
                manager_->Deallocate(ptr_);
            }
            manager_ = other.manager_;
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.manager_ = nullptr;
            other.ptr_ = nullptr;
        }
        return *this;
    }

private:
    MemoryManager* manager_;
    T* ptr_;
    size_t count_;
};

} // namespace hpie

