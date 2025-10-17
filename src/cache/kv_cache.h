#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <chrono>

#include "../memory/memory_manager.h"

namespace hpie {

// Cache replacement policy enum (must be defined before CacheConfig)
enum class CacheReplacementPolicy {
    LRU,        // Least Recently Used
    LFU,        // Least Frequently Used
    FIFO,       // First In First Out
    RANDOM,     // Random replacement
    ADAPTIVE    // Adaptive based on access patterns
};

// Cache configuration
struct CacheConfig {
    size_t max_entries;
    size_t max_memory_bytes;
    float eviction_threshold;
    bool enable_prefetch;
    bool enable_compression;
    CacheReplacementPolicy replacement_policy;
};

// Key-Value cache entry
struct CacheEntry {
    std::vector<uint32_t> key;        // Input token sequence
    std::vector<float> value;         // Computed hidden states
    std::chrono::steady_clock::time_point last_access;
    std::atomic<uint32_t> access_count;
    size_t memory_size;
    bool is_valid;
    
    CacheEntry() : access_count(0), memory_size(0), is_valid(true) {
        last_access = std::chrono::steady_clock::now();
    }
    
    void UpdateAccess() {
        last_access = std::chrono::steady_clock::now();
        access_count.fetch_add(1, std::memory_order_relaxed);
    }
};

// Attention cache entry for storing attention weights
struct AttentionCacheEntry {
    std::vector<float> attention_weights;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    size_t sequence_length;
    std::chrono::steady_clock::time_point last_access;
    
    AttentionCacheEntry() : sequence_length(0) {
        last_access = std::chrono::steady_clock::now();
    }
};

// Main Key-Value cache implementation
class KVCache {
public:
    explicit KVCache(size_t max_context_length, MemoryManager* memory_manager = nullptr);
    ~KVCache();

    // Core cache operations
    bool Store(const std::vector<uint32_t>& key, const std::vector<float>& value);
    std::vector<float> Retrieve(const std::vector<uint32_t>& key);
    bool Contains(const std::vector<uint32_t>& key) const;
    void Remove(const std::vector<uint32_t>& key);
    void Clear();

    // Prefill-specific operations
    void StorePrefill(size_t sequence_length, const std::vector<float>& hidden_states);
    std::vector<float> RetrievePrefill(size_t sequence_length);
    
    // Decode-specific operations
    void Update(size_t position, uint32_t token_id);
    std::vector<float> GetCachedOutput(size_t position);
    
    // Cache management
    void Evict();
    void Optimize();
    void Defragment();
    
    // Statistics
    size_t GetSize() const { return cache_entries_.size(); }
    size_t GetMemoryUsage() const { return memory_usage_; }
    float GetHitRate() const;
    size_t GetHitCount() const { return hit_count_; }
    size_t GetMissCount() const { return miss_count_; }
    
    // Configuration
    void SetConfig(const CacheConfig& config);
    CacheConfig GetConfig() const { return config_; }

private:
    // Cache storage
    std::unordered_map<std::string, std::unique_ptr<CacheEntry>> cache_entries_;
    std::unordered_map<std::string, std::unique_ptr<AttentionCacheEntry>> attention_cache_;
    
    // Memory management
    MemoryManager* memory_manager_;
    size_t memory_usage_;
    size_t max_memory_;
    
    // Statistics
    std::atomic<size_t> hit_count_;
    std::atomic<size_t> miss_count_;
    
    // Configuration
    CacheConfig config_;
    
    // Thread safety
    mutable std::mutex cache_mutex_;
    mutable std::mutex attention_mutex_;
    
    // Internal methods
    std::string HashKey(const std::vector<uint32_t>& key) const;
    void EvictLRU();
    void EvictLFU();
    void EvictRandom();
    void EvictAdaptive();
    
    // Memory optimization
    bool CompressEntry(CacheEntry& entry);
    bool DecompressEntry(CacheEntry& entry);
    void PrefetchEntry(const std::string& key);
    
    // Attention cache management
    void StoreAttentionWeights(size_t seq_len, const std::vector<float>& weights);
    std::vector<float> RetrieveAttentionWeights(size_t seq_len);
};

// Attention cache for storing computed attention matrices
class AttentionCache {
public:
    explicit AttentionCache(size_t max_sequence_length, MemoryManager* memory_manager = nullptr);
    ~AttentionCache();

    // Attention operations
    bool StoreAttention(size_t layer, size_t sequence_length, 
                       const std::vector<float>& attention_weights);
    std::vector<float> RetrieveAttention(size_t layer, size_t sequence_length);
    
    // Key-Value cache operations
    bool StoreKV(size_t layer, size_t position, 
                const std::vector<float>& key, const std::vector<float>& value);
    std::pair<std::vector<float>, std::vector<float>> RetrieveKV(size_t layer, size_t position);
    
    // Cache management
    void Clear();
    void EvictOldEntries();
    void OptimizeForSequenceLength(size_t target_length);
    
    // Statistics
    size_t GetSize() const { return attention_entries_.size(); }
    float GetHitRate() const;
    size_t GetMemoryUsage() const { return memory_usage_; }

private:
    struct AttentionEntry {
        size_t layer;
        size_t sequence_length;
        std::vector<float> weights;
        std::vector<float> keys;
        std::vector<float> values;
        std::chrono::steady_clock::time_point last_access;
        size_t memory_size;
    };
    
    std::unordered_map<std::string, std::unique_ptr<AttentionEntry>> attention_entries_;
    MemoryManager* memory_manager_;
    size_t memory_usage_;
    size_t max_sequence_length_;
    
    std::atomic<size_t> hit_count_;
    std::atomic<size_t> miss_count_;
    
    mutable std::mutex mutex_;
    
    std::string HashKey(size_t layer, size_t sequence_length) const;
    void EvictLRU();
};

// Cache prefetcher for proactive loading
class CachePrefetcher {
public:
    explicit CachePrefetcher(KVCache* kv_cache, AttentionCache* attention_cache);
    ~CachePrefetcher();

    // Prefetching strategies
    void PrefetchNextToken(const std::vector<uint32_t>& current_sequence);
    void PrefetchAttentionPattern(size_t layer, size_t sequence_length);
    void PrefetchBasedOnHistory();
    
    // Prediction-based prefetching
    void UpdatePredictionModel(const std::vector<uint32_t>& sequence, 
                              const std::vector<uint32_t>& next_tokens);
    std::vector<std::vector<uint32_t>> PredictNextSequences(
        const std::vector<uint32_t>& current_sequence);

private:
    KVCache* kv_cache_;
    AttentionCache* attention_cache_;
    
    // Prediction model
    std::unordered_map<std::string, std::vector<uint32_t>> prediction_model_;
    std::mutex prediction_mutex_;
    
    void BuildPredictionModel(const std::vector<std::vector<uint32_t>>& sequences);
    float CalculateSequenceSimilarity(const std::vector<uint32_t>& seq1, 
                                     const std::vector<uint32_t>& seq2);
};

// Cache performance monitor
class CacheMonitor {
public:
    CacheMonitor() = default;
    ~CacheMonitor() = default;

    // Monitoring
    void RecordCacheHit(const std::string& cache_type, size_t key_size, size_t value_size);
    void RecordCacheMiss(const std::string& cache_type, size_t key_size);
    void RecordEviction(const std::string& cache_type, size_t memory_freed);
    
    // Performance metrics
    struct CacheMetrics {
        size_t total_hits;
        size_t total_misses;
        float hit_rate;
        size_t total_memory_used;
        size_t eviction_count;
        double average_access_time;
        std::chrono::microseconds total_access_time;
    };
    
    CacheMetrics GetMetrics(const std::string& cache_type) const;
    void ResetMetrics(const std::string& cache_type = "");

private:
    struct CacheStats {
        std::atomic<size_t> hits{0};
        std::atomic<size_t> misses{0};
        std::atomic<size_t> evictions{0};
        std::atomic<size_t> memory_used{0};
        std::atomic<std::chrono::microseconds> total_access_time{std::chrono::microseconds{0}};
        std::atomic<size_t> access_count{0};
    };
    
    std::unordered_map<std::string, CacheStats> cache_stats_;
    mutable std::mutex stats_mutex_;
    
    std::chrono::steady_clock::time_point start_time_;
};

// Cache utilities
class CacheUtils {
public:
    // Memory optimization
    static size_t CalculateOptimalCacheSize(size_t available_memory, 
                                          size_t model_size,
                                          size_t batch_size);
    
    // Compression utilities
    static std::vector<uint8_t> CompressFloatVector(const std::vector<float>& data);
    static std::vector<float> DecompressFloatVector(const std::vector<uint8_t>& compressed);
    
    // Cache key generation
    static std::string GenerateCacheKey(const std::vector<uint32_t>& tokens);
    static std::string GenerateCacheKey(size_t layer, size_t position, size_t sequence_length);
    
    // Performance analysis
    static void AnalyzeCachePerformance(const std::vector<std::string>& access_patterns);
    static std::vector<size_t> FindOptimalCacheSizes(const std::vector<size_t>& sequence_lengths);
    
    // Cache warming
    static void WarmCache(KVCache* cache, const std::vector<std::vector<uint32_t>>& sequences);
    static void WarmAttentionCache(AttentionCache* cache, 
                                  const std::vector<std::pair<size_t, size_t>>& layer_seq_pairs);
};

} // namespace hpie

