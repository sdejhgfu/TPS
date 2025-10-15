#include "memory_manager.h"
#include "../utils/logger.h"
#include <algorithm>
#include <cstring>
#include <immintrin.h>

namespace hpie {

// MemoryPool implementation
MemoryPool::MemoryPool(size_t pool_size, size_t block_size)
    : pool_size_(pool_size), block_size_(block_size) {
    
    // Allocate aligned memory for the pool
    pool_memory_ = aligned_alloc(CACHE_LINE_SIZE, pool_size);
    if (!pool_memory_) {
        throw std::bad_alloc();
    }
    
    total_blocks_ = pool_size / block_size;
    free_blocks_ = total_blocks_;
    
    // Initialize free list
    free_list_head_ = pool_memory_;
    char* ptr = static_cast<char*>(pool_memory_);
    
    for (size_t i = 0; i < total_blocks_ - 1; ++i) {
        *reinterpret_cast<void**>(ptr) = ptr + block_size_;
        ptr += block_size_;
    }
    *reinterpret_cast<void**>(ptr) = nullptr;
    
    Logger::Info("Memory pool created: %zu blocks of %zu bytes each",
                 total_blocks_, block_size_);
}

MemoryPool::~MemoryPool() {
    if (pool_memory_) {
        free(pool_memory_);
    }
}

void* MemoryPool::Allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!free_list_head_) {
        return nullptr;
    }
    
    void* block = free_list_head_;
    free_list_head_ = *reinterpret_cast<void**>(block);
    free_blocks_--;
    
    return block;
}

void MemoryPool::Deallocate(void* ptr) {
    if (!Contains(ptr)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    *reinterpret_cast<void**>(ptr) = free_list_head_;
    free_list_head_ = ptr;
    free_blocks_++;
}

bool MemoryPool::Contains(void* ptr) const {
    return ptr >= pool_memory_ && 
           ptr < static_cast<char*>(pool_memory_) + pool_size_;
}

// PageManager implementation
PageManager::PageManager(size_t max_pages)
    : total_pages_(max_pages), used_pages_(0) {
    
    pages_.reserve(max_pages);
    page_used_.reserve(max_pages);
    
    // Pre-allocate all pages
    for (size_t i = 0; i < max_pages; ++i) {
        void* page = aligned_alloc(PAGE_SIZE, PAGE_SIZE);
        if (!page) {
            throw std::bad_alloc();
        }
        pages_.push_back(page);
        page_used_.push_back(false);
    }
    
    Logger::Info("Page manager created with %zu pages", max_pages);
}

PageManager::~PageManager() {
    for (void* page : pages_) {
        if (page) {
            free(page);
        }
    }
}

void* PageManager::AllocatePages(size_t num_pages) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find contiguous free pages
    for (size_t i = 0; i <= total_pages_ - num_pages; ++i) {
        bool all_free = true;
        for (size_t j = 0; j < num_pages; ++j) {
            if (page_used_[i + j]) {
                all_free = false;
                break;
            }
        }
        
        if (all_free) {
            // Mark pages as used
            for (size_t j = 0; j < num_pages; ++j) {
                page_used_[i + j] = true;
            }
            used_pages_ += num_pages;
            return pages_[i];
        }
    }
    
    return nullptr;
}

void PageManager::DeallocatePages(void* ptr, size_t num_pages) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find the page index
    auto it = std::find(pages_.begin(), pages_.end(), ptr);
    if (it == pages_.end()) {
        return;
    }
    
    size_t index = std::distance(pages_.begin(), it);
    
    // Mark pages as free
    for (size_t i = 0; i < num_pages && index + i < total_pages_; ++i) {
        page_used_[index + i] = false;
    }
    used_pages_ -= num_pages;
}

void* PageManager::AllocatePage() {
    return AllocatePages(1);
}

void PageManager::DeallocatePage(void* ptr) {
    DeallocatePages(ptr, 1);
}

// MemoryManager implementation
MemoryManager::MemoryManager(size_t max_memory)
    : strategy_(AllocationStrategy::BUDDY_SYSTEM),
      total_memory_(0),
      used_memory_(0),
      num_allocations_(0),
      num_deallocations_(0),
      cache_hits_(0),
      cache_misses_(0) {
    
    // Initialize small memory pools for common sizes
    small_pools_.emplace_back(std::make_unique<MemoryPool>(1024 * 1024, 64));   // 64-byte blocks
    small_pools_.emplace_back(std::make_unique<MemoryPool>(2 * 1024 * 1024, 256)); // 256-byte blocks
    small_pools_.emplace_back(std::make_unique<MemoryPool>(4 * 1024 * 1024, 1024)); // 1KB blocks
    
    // Initialize page manager
    size_t max_pages = max_memory / PAGE_SIZE;
    page_manager_ = std::make_unique<PageManager>(max_pages);
    
    total_memory_ = max_memory;
    
    Logger::Info("Memory manager initialized with %zu MB total memory",
                 max_memory / (1024 * 1024));
}

MemoryManager::~MemoryManager() {
    // Clean up all allocations
    for (auto& pair : allocations_) {
        free(pair.first);
    }
}

void* MemoryManager::Allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (size == 0) {
        return nullptr;
    }
    
    size_t aligned_size = AlignUp(size, alignment);
    
    // Try to allocate from small pools first
    void* ptr = AllocateFromPool(aligned_size);
    if (ptr) {
        num_allocations_++;
        used_memory_ += aligned_size;
        cache_hits_++;
        return ptr;
    }
    
    // Fall back to heap allocation
    ptr = AllocateFromHeap(aligned_size, alignment);
    if (ptr) {
        // Track allocation
        MemoryBlock block;
        block.ptr = ptr;
        block.size = aligned_size;
        block.alignment = alignment;
        block.is_free = false;
        block.magic_number = 0xDEADBEEF;
        allocations_[ptr] = block;
        
        num_allocations_++;
        used_memory_ += aligned_size;
        cache_misses_++;
    }
    
    return ptr;
}

void MemoryManager::Deallocate(void* ptr) {
    if (!ptr) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if it's from a pool
    for (auto& pool : small_pools_) {
        if (pool->Contains(ptr)) {
            pool->Deallocate(ptr);
            num_deallocations_++;
            return;
        }
    }
    
    // Check tracked allocations
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        used_memory_ -= it->second.size;
        allocations_.erase(it);
        free(ptr);
        num_deallocations_++;
    }
}

void* MemoryManager::Reallocate(void* ptr, size_t new_size) {
    if (!ptr) {
        return Allocate(new_size);
    }
    
    if (new_size == 0) {
        Deallocate(ptr);
        return nullptr;
    }
    
    // Find current allocation
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        // Not tracked, allocate new and copy
        void* new_ptr = Allocate(new_size);
        if (new_ptr && ptr) {
            std::memcpy(new_ptr, ptr, std::min(new_size, it->second.size));
        }
        return new_ptr;
    }
    
    size_t current_size = it->second.size;
    if (new_size <= current_size) {
        // Shrink allocation
        it->second.size = new_size;
        used_memory_ -= (current_size - new_size);
        return ptr;
    }
    
    // Grow allocation
    void* new_ptr = Allocate(new_size);
    if (new_ptr) {
        std::memcpy(new_ptr, ptr, current_size);
        Deallocate(ptr);
    }
    
    return new_ptr;
}

void* MemoryManager::AllocateFromPool(size_t size) {
    // Find appropriate pool
    for (auto& pool : small_pools_) {
        if (size <= pool->GetTotalBlocks() * 64) { // Approximate check
            void* ptr = pool->Allocate();
            if (ptr) {
                return ptr;
            }
        }
    }
    return nullptr;
}

void* MemoryManager::AllocateFromHeap(size_t size, size_t alignment) {
    // Use aligned allocation for large blocks
    if (size >= PAGE_SIZE) {
        size_t num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
        void* ptr = page_manager_->AllocatePages(num_pages);
        if (ptr) {
            return ptr;
        }
    }
    
    // Fall back to standard aligned allocation
    return aligned_alloc(alignment, size);
}

void MemoryManager::Defragment() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Merge adjacent free blocks
    MergeFreeBlocks();
    
    // Compact memory by moving allocated blocks
    std::vector<MemoryBlock*> blocks_to_move;
    for (auto& pair : allocations_) {
        blocks_to_move.push_back(&pair.second);
    }
    
    // Sort by address
    std::sort(blocks_to_move.begin(), blocks_to_move.end(),
              [](const MemoryBlock* a, const MemoryBlock* b) {
                  return a->ptr < b->ptr;
              });
    
    // Move blocks to reduce fragmentation
    void* current_ptr = nullptr;
    for (auto* block : blocks_to_move) {
        if (block->ptr != current_ptr) {
            // Move block to current_ptr
            std::memmove(current_ptr, block->ptr, block->size);
            block->ptr = current_ptr;
            current_ptr = static_cast<char*>(current_ptr) + block->size;
        }
    }
    
    Logger::Info("Memory defragmentation completed");
}

void MemoryManager::MergeFreeBlocks() {
    // Implementation for merging adjacent free blocks
    // This would involve updating the free list and combining blocks
}

void MemoryManager::SetAllocationStrategy(AllocationStrategy strategy) {
    std::lock_guard<std::mutex> lock(mutex_);
    strategy_ = strategy;
    Logger::Info("Allocation strategy set to %d", static_cast<int>(strategy));
}

MemoryStats MemoryManager::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    MemoryStats stats;
    stats.total_memory = total_memory_;
    stats.used_memory = used_memory_;
    stats.free_memory = total_memory_ - used_memory_;
    stats.num_allocations = num_allocations_;
    stats.num_deallocations = num_deallocations_;
    stats.cache_hits = cache_hits_;
    stats.cache_misses = cache_misses_;
    
    // Calculate fragmentation
    size_t largest_free_block = 0;
    for (const auto& pool : small_pools_) {
        largest_free_block = std::max(largest_free_block, 
                                     pool->GetFreeBlocks() * 64);
    }
    
    stats.fragmented_memory = stats.free_memory - largest_free_block;
    stats.fragmentation_ratio = stats.free_memory > 0 ? 
        static_cast<double>(stats.fragmented_memory) / stats.free_memory : 0.0;
    
    return stats;
}

void MemoryManager::OptimizeMemoryLayout() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Prefetch frequently accessed memory regions
    for (const auto& pair : allocations_) {
        PrefetchBlock(pair.second);
    }
    
    Logger::Info("Memory layout optimized");
}

void MemoryManager::PrefetchMemory(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }
    
    // Prefetch memory into cache
    char* char_ptr = static_cast<char*>(ptr);
    for (size_t i = 0; i < size; i += CACHE_LINE_SIZE) {
        __builtin_prefetch(char_ptr + i, 0, 3); // Read, temporal locality
    }
}

void MemoryManager::FlushCache(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }
    
    // Flush cache lines
    char* char_ptr = static_cast<char*>(ptr);
    for (size_t i = 0; i < size; i += CACHE_LINE_SIZE) {
        _mm_clflush(char_ptr + i);
    }
}

void MemoryManager::PrefetchBlock(const MemoryBlock& block) {
    PrefetchMemory(block.ptr, block.size);
}

size_t MemoryManager::AlignUp(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

bool MemoryManager::IsAligned(void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

void MemoryManager::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clear all allocations
    for (auto& pair : allocations_) {
        free(pair.first);
    }
    allocations_.clear();
    
    // Reset statistics
    used_memory_ = 0;
    num_allocations_ = 0;
    num_deallocations_ = 0;
    
    Logger::Info("Memory manager cleared");
}

} // namespace hpie

