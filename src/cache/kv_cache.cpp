#include "kv_cache.h"
#include "../utils/logger.h"
#include "../utils/timer.h"
#include <algorithm>
#include <functional>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <zlib.h> // For compression

namespace hpie {

// KVCache implementation
KVCache::KVCache(size_t max_context_length, MemoryManager* memory_manager)
    : memory_manager_(memory_manager),
      memory_usage_(0),
      max_memory_(max_context_length * 1024 * 1024), // 1MB per context length
      hit_count_(0),
      miss_count_(0) {
    
    config_.max_entries = max_context_length * 4; // 4x context length
    config_.max_memory_bytes = max_memory_;
    config_.eviction_threshold = 0.8f;
    config_.enable_prefetch = true;
    config_.enable_compression = true;
    config_.replacement_policy = CacheReplacementPolicy::ADAPTIVE;
    
    Logger::Info("KV Cache initialized with %zu MB max memory", 
                 max_memory_ / (1024 * 1024));
}

KVCache::~KVCache() {
    Clear();
}

std::string KVCache::HashKey(const std::vector<uint32_t>& key) const {
    std::hash<uint32_t> hasher;
    size_t hash_value = 0;
    
    for (uint32_t token : key) {
        hash_value ^= hasher(token) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
    }
    
    std::stringstream ss;
    ss << std::hex << hash_value;
    return ss.str();
}

bool KVCache::Store(const std::vector<uint32_t>& key, const std::vector<float>& value) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    if (key.empty() || value.empty()) {
        return false;
    }
    
    std::string hash = HashKey(key);
    
    // Check if entry already exists
    auto it = cache_entries_.find(hash);
    if (it != cache_entries_.end()) {
        // Update existing entry
        it->second->value = value;
        it->second->UpdateAccess();
        return true;
    }
    
    // Create new entry
    auto entry = std::make_unique<CacheEntry>();
    entry->key = key;
    entry->value = value;
    entry->memory_size = key.size() * sizeof(uint32_t) + value.size() * sizeof(float);
    
    // Check memory limits
    if (memory_usage_ + entry->memory_size > config_.max_memory_bytes) {
        Evict();
    }
    
    // Compress if enabled
    if (config_.enable_compression) {
        CompressEntry(*entry);
    }
    
    // Save memory size before move
    size_t entry_size = entry->memory_size;
    
    cache_entries_[hash] = std::move(entry);
    memory_usage_ += entry_size;
    
    return true;
}

std::vector<float> KVCache::Retrieve(const std::vector<uint32_t>& key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string hash = HashKey(key);
    auto it = cache_entries_.find(hash);
    
    if (it != cache_entries_.end()) {
        it->second->UpdateAccess();
        hit_count_++;
        
        // Decompress if needed
        if (config_.enable_compression && it->second->memory_size != it->second->value.size() * sizeof(float)) {
            DecompressEntry(*it->second);
        }
        
        return it->second->value;
    }
    
    miss_count_++;
    return {};
}

bool KVCache::Contains(const std::vector<uint32_t>& key) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    std::string hash = HashKey(key);
    return cache_entries_.find(hash) != cache_entries_.end();
}

void KVCache::Remove(const std::vector<uint32_t>& key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string hash = HashKey(key);
    auto it = cache_entries_.find(hash);
    
    if (it != cache_entries_.end()) {
        memory_usage_ -= it->second->memory_size;
        cache_entries_.erase(it);
    }
}

void KVCache::Clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    std::lock_guard<std::mutex> attention_lock(attention_mutex_);
    
    cache_entries_.clear();
    attention_cache_.clear();
    memory_usage_ = 0;
    
    Logger::Info("Cache cleared");
}

void KVCache::StorePrefill(size_t sequence_length, const std::vector<float>& hidden_states) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string key = "prefill_" + std::to_string(sequence_length);
    
    auto entry = std::make_unique<CacheEntry>();
    entry->key = {static_cast<uint32_t>(sequence_length)};
    entry->value = hidden_states;
    entry->memory_size = hidden_states.size() * sizeof(float);
    
    cache_entries_[key] = std::move(entry);
    memory_usage_ += entry->memory_size;
}

std::vector<float> KVCache::RetrievePrefill(size_t sequence_length) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string key = "prefill_" + std::to_string(sequence_length);
    auto it = cache_entries_.find(key);
    
    if (it != cache_entries_.end()) {
        it->second->UpdateAccess();
        hit_count_++;
        return it->second->value;
    }
    
    miss_count_++;
    return {};
}

void KVCache::Update(size_t position, uint32_t token_id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string key = "decode_" + std::to_string(position);
    
    auto it = cache_entries_.find(key);
    if (it != cache_entries_.end()) {
        // Update existing decode cache entry
        it->second->key = {static_cast<uint32_t>(position), token_id};
        it->second->UpdateAccess();
    } else {
        // Create new decode cache entry
        auto entry = std::make_unique<CacheEntry>();
        entry->key = {static_cast<uint32_t>(position), token_id};
        entry->value = {}; // Will be populated when computed
        entry->memory_size = sizeof(uint32_t) * 2;
        
        cache_entries_[key] = std::move(entry);
        memory_usage_ += entry->memory_size;
    }
}

std::vector<float> KVCache::GetCachedOutput(size_t position) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string key = "decode_" + std::to_string(position);
    auto it = cache_entries_.find(key);
    
    if (it != cache_entries_.end()) {
        it->second->UpdateAccess();
        hit_count_++;
        return it->second->value;
    }
    
    miss_count_++;
    return {};
}

void KVCache::Evict() {
    switch (config_.replacement_policy) {
        case CacheReplacementPolicy::LRU:
            EvictLRU();
            break;
        case CacheReplacementPolicy::LFU:
            EvictLFU();
            break;
        case CacheReplacementPolicy::RANDOM:
            EvictRandom();
            break;
        case CacheReplacementPolicy::ADAPTIVE:
            EvictAdaptive();
            break;
        default:
            EvictLRU();
            break;
    }
}

void KVCache::EvictLRU() {
    if (cache_entries_.empty()) {
        return;
    }
    
    auto oldest_it = cache_entries_.begin();
    auto oldest_time = oldest_it->second->last_access;
    
    for (auto it = cache_entries_.begin(); it != cache_entries_.end(); ++it) {
        if (it->second->last_access < oldest_time) {
            oldest_time = it->second->last_access;
            oldest_it = it;
        }
    }
    
    memory_usage_ -= oldest_it->second->memory_size;
    cache_entries_.erase(oldest_it);
    
    Logger::Debug("Evicted LRU cache entry, freed %zu bytes", oldest_it->second->memory_size);
}

void KVCache::EvictLFU() {
    if (cache_entries_.empty()) {
        return;
    }
    
    auto least_frequent_it = cache_entries_.begin();
    uint32_t min_access_count = least_frequent_it->second->access_count.load();
    
    for (auto it = cache_entries_.begin(); it != cache_entries_.end(); ++it) {
        if (it->second->access_count.load() < min_access_count) {
            min_access_count = it->second->access_count.load();
            least_frequent_it = it;
        }
    }
    
    memory_usage_ -= least_frequent_it->second->memory_size;
    cache_entries_.erase(least_frequent_it);
    
    Logger::Debug("Evicted LFU cache entry, freed %zu bytes", least_frequent_it->second->memory_size);
}

void KVCache::EvictRandom() {
    if (cache_entries_.empty()) {
        return;
    }
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, cache_entries_.size() - 1);
    
    size_t random_index = dis(gen);
    auto it = cache_entries_.begin();
    std::advance(it, random_index);
    
    memory_usage_ -= it->second->memory_size;
    cache_entries_.erase(it);
    
    Logger::Debug("Evicted random cache entry, freed %zu bytes", it->second->memory_size);
}

void KVCache::EvictAdaptive() {
    // Adaptive eviction based on access patterns and memory pressure
    float memory_pressure = static_cast<float>(memory_usage_) / config_.max_memory_bytes;
    
    if (memory_pressure > 0.9f) {
        // High memory pressure - evict aggressively
        EvictLFU();
        if (memory_pressure > 0.95f) {
            EvictLRU(); // Double eviction for critical pressure
        }
    } else if (memory_pressure > config_.eviction_threshold) {
        // Normal eviction threshold
        EvictLRU();
    } else {
        // Low pressure - evict least recently used with low frequency
        EvictLRU();
    }
}

void KVCache::Optimize() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Remove invalid entries
    for (auto it = cache_entries_.begin(); it != cache_entries_.end();) {
        if (!it->second->is_valid) {
            memory_usage_ -= it->second->memory_size;
            it = cache_entries_.erase(it);
        } else {
            ++it;
        }
    }
    
    // Defragment if needed
    if (memory_usage_ > config_.max_memory_bytes * 0.8f) {
        Defragment();
    }
    
    Logger::Info("Cache optimized, %zu entries remaining", cache_entries_.size());
}

void KVCache::Defragment() {
    // Implementation would involve compacting memory and updating pointers
    // For now, we'll just evict some entries to free up space
    size_t target_memory = config_.max_memory_bytes * 0.6f; // Target 60% usage
    
    while (memory_usage_ > target_memory && !cache_entries_.empty()) {
        EvictLRU();
    }
    
    Logger::Info("Cache defragmented, memory usage: %zu MB", 
                 memory_usage_ / (1024 * 1024));
}

bool KVCache::CompressEntry(CacheEntry& entry) {
    if (entry.value.empty()) {
        return false;
    }
    
    // Simple compression using zlib
    uLongf compressed_size = compressBound(entry.value.size() * sizeof(float));
    std::vector<uint8_t> compressed(compressed_size);
    
    int result = compress(
        compressed.data(), &compressed_size,
        reinterpret_cast<const Bytef*>(entry.value.data()),
        entry.value.size() * sizeof(float)
    );
    
    if (result == Z_OK && compressed_size < entry.value.size() * sizeof(float)) {
        entry.value.resize(compressed_size / sizeof(float));
        std::memcpy(entry.value.data(), compressed.data(), compressed_size);
        entry.memory_size = compressed_size;
        return true;
    }
    
    return false;
}

bool KVCache::DecompressEntry(CacheEntry& entry) {
    if (entry.value.empty()) {
        return false;
    }
    
    // Estimate original size (this is simplified)
    uLongf original_size = entry.value.size() * sizeof(float) * 4; // Rough estimate
    std::vector<float> decompressed(original_size / sizeof(float));
    
    int result = uncompress(
        reinterpret_cast<Bytef*>(decompressed.data()), &original_size,
        reinterpret_cast<const Bytef*>(entry.value.data()),
        entry.memory_size
    );
    
    if (result == Z_OK) {
        entry.value = std::move(decompressed);
        entry.memory_size = original_size;
        return true;
    }
    
    return false;
}

float KVCache::GetHitRate() const {
    size_t total_requests = hit_count_.load() + miss_count_.load();
    return total_requests > 0 ? static_cast<float>(hit_count_.load()) / total_requests : 0.0f;
}

void KVCache::SetConfig(const CacheConfig& config) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    config_ = config;
    max_memory_ = config.max_memory_bytes;
    
    // Adjust cache if new limits are smaller
    if (memory_usage_ > max_memory_) {
        Evict();
    }
    
    Logger::Info("Cache configuration updated");
}

void KVCache::PrefetchEntry(const std::string& key) {
    (void)key;  // Suppress unused warning
    // Implementation would involve prefetching based on prediction
    // For now, this is a placeholder
}

// AttentionCache implementation
AttentionCache::AttentionCache(size_t max_sequence_length, MemoryManager* memory_manager)
    : memory_manager_(memory_manager),
      memory_usage_(0),
      max_sequence_length_(max_sequence_length),
      hit_count_(0),
      miss_count_(0) {
    
    Logger::Info("Attention cache initialized for max sequence length %zu", 
                 max_sequence_length);
}

AttentionCache::~AttentionCache() {
    Clear();
}

bool AttentionCache::StoreAttention(size_t layer, size_t sequence_length,
                                   const std::vector<float>& attention_weights) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string key = HashKey(layer, sequence_length);
    
    auto entry = std::make_unique<AttentionEntry>();
    entry->layer = layer;
    entry->sequence_length = sequence_length;
    entry->weights = attention_weights;
    entry->memory_size = attention_weights.size() * sizeof(float);
    entry->last_access = std::chrono::steady_clock::now();
    
    attention_entries_[key] = std::move(entry);
    memory_usage_ += entry->memory_size;
    
    return true;
}

std::vector<float> AttentionCache::RetrieveAttention(size_t layer, size_t sequence_length) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string key = HashKey(layer, sequence_length);
    auto it = attention_entries_.find(key);
    
    if (it != attention_entries_.end()) {
        it->second->last_access = std::chrono::steady_clock::now();
        hit_count_++;
        return it->second->weights;
    }
    
    miss_count_++;
    return {};
}

std::string AttentionCache::HashKey(size_t layer, size_t sequence_length) const {
    return std::to_string(layer) + "_" + std::to_string(sequence_length);
}

void AttentionCache::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    attention_entries_.clear();
    memory_usage_ = 0;
}

float AttentionCache::GetHitRate() const {
    size_t total_requests = hit_count_.load() + miss_count_.load();
    return total_requests > 0 ? static_cast<float>(hit_count_.load()) / total_requests : 0.0f;
}

void AttentionCache::EvictOldEntries() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto cutoff_time = now - std::chrono::minutes(10); // Evict entries older than 10 minutes
    
    for (auto it = attention_entries_.begin(); it != attention_entries_.end();) {
        if (it->second->last_access < cutoff_time) {
            memory_usage_ -= it->second->memory_size;
            it = attention_entries_.erase(it);
        } else {
            ++it;
        }
    }
}

// CacheUtils implementation
std::string CacheUtils::GenerateCacheKey(const std::vector<uint32_t>& tokens) {
    std::stringstream ss;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) ss << "_";
        ss << tokens[i];
    }
    return ss.str();
}

std::string CacheUtils::GenerateCacheKey(size_t layer, size_t position, size_t sequence_length) {
    return std::to_string(layer) + "_" + std::to_string(position) + "_" + std::to_string(sequence_length);
}

void CacheUtils::WarmCache(KVCache* cache, const std::vector<std::vector<uint32_t>>& sequences) {
    for (const auto& sequence : sequences) {
        // Generate dummy values for warming
        std::vector<float> dummy_values(sequence.size() * 512, 1.0f); // Assuming 512 hidden size
        cache->Store(sequence, dummy_values);
    }
    
    Logger::Info("Cache warmed with %zu sequences", sequences.size());
}

} // namespace hpie
