#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <random>
#include <cmath>
#include <algorithm>
#include <immintrin.h>  // For AVX2/AVX512 intrinsics

namespace tps {

// Performance constants optimized for 2-socket, 8-core, 32GB DDR5 system
constexpr int VOCAB_SIZE = 2048;
constexpr int HIDDEN_SIZE = 128;
constexpr int NUM_LAYERS = 4;
constexpr int NUM_HEADS = 4;
constexpr int HEAD_DIM = HIDDEN_SIZE / NUM_HEADS;
constexpr int MAX_SEQ_LEN = 64;
constexpr int CACHE_SIZE = 256;

// Performance statistics
struct PerformanceStats {
    int generatedTokens = 0;
    double elapsedSeconds = 0.0;
    double tps = 0.0;
    size_t memoryUsed = 0;
    double cacheHitRatio = 0.0;
};

// Simple tokenizer for fast word-based tokenization
class Tokenizer {
public:
    Tokenizer();
    
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode(int token) const;
    
    int getVocabSize() const { return VOCAB_SIZE; }
    int getEOSToken() const { return 0; }

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> reverse_vocab_;
    
    void buildVocab();
    int getTokenId(const std::string& word) const;
};

// High-performance memory manager with alignment
class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();
    
    void* allocate(size_t size, size_t alignment = 64);
    void deallocate(void* ptr);
    void reset();
    
    size_t getTotalAllocated() const { return total_allocated_; }
    size_t getPeakUsage() const { return peak_usage_; }

private:
    struct Block {
        void* ptr;
        size_t size;
        bool free;
    };
    
    std::vector<Block> blocks_;
    size_t total_allocated_ = 0;
    size_t peak_usage_ = 0;
    
    void* findFreeBlock(size_t size);
};

// Efficient cache for KV pairs
class KVCache {
public:
    KVCache();
    ~KVCache();
    
    void initialize(size_t max_sequences, size_t max_tokens);
    void clear();
    void clearSequence(int sequence_id);
    
    void store(int sequence_id, int layer, int pos, const float* k, const float* v);
    void retrieve(int sequence_id, int layer, int start_pos, int length, 
                  float* k_out, float* v_out) const;
    
    bool isSequenceCached(int sequence_id) const;
    size_t getMemoryUsage() const;
    double getHitRatio() const;
    
    void resetStats();
    size_t getCacheHits() const { return cache_hits_; }
    size_t getCacheMisses() const { return cache_misses_; }

private:
    struct CacheEntry {
        int sequence_id;
        int layer;
        int pos;
        std::vector<float> k_data;
        std::vector<float> v_data;
    };
    
    std::unordered_map<int, std::vector<CacheEntry>> cache_;
    size_t max_sequences_;
    size_t max_tokens_per_sequence_;
    size_t memory_limit_;
    size_t current_memory_usage_;
    
    mutable size_t cache_hits_ = 0;
    mutable size_t cache_misses_ = 0;
    
    void evictLRU();
};

// Lightweight transformer model
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    bool initialize();
    void shutdown();
    
    std::string generate(const std::string& prompt, int max_tokens = 100);
    PerformanceStats runBatch(const std::vector<std::string>& prompts);
    
    PerformanceStats getLastStats() const { return last_stats_; }
    size_t getMemoryUsage() const;

private:
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<KVCache> kv_cache_;
    
    // Model weights (small for performance)
    std::vector<std::vector<float>> token_embeddings_;
    std::vector<std::vector<float>> position_embeddings_;
    std::vector<std::vector<std::vector<float>>> attention_weights_;
    std::vector<std::vector<std::vector<float>>> ffn_weights_;
    std::vector<float> output_projection_;
    
    // Performance tracking
    PerformanceStats last_stats_;
    
    // Random number generator for sampling
    mutable std::mt19937 rng_;
    
    // Internal methods
    void initializeWeights();
    void forwardPass(const std::vector<int>& tokens, std::vector<float>& logits, int sequence_id = 0);
    void transformerLayer(int layer, const std::vector<float>& input, std::vector<float>& output, int sequence_id = 0);
    void multiHeadAttention(int layer, const std::vector<float>& input, std::vector<float>& output, int sequence_id = 0);
    void feedForward(int layer, const std::vector<float>& input, std::vector<float>& output);
    
    // Optimized operations
    void matmul(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int m, int n, int k);
    void matmulAVX2(const float* a, const float* b, float* c, int m, int n, int k);
    void layerNorm(const std::vector<float>& input, std::vector<float>& output);
    void relu(std::vector<float>& x);
    void softmax(std::vector<float>& x);
    void applyRotaryEmbeddings(std::vector<float>& q, std::vector<float>& k, int seq_len);
    
    // Sampling
    int sampleToken(const std::vector<float>& logits, float temperature = 0.8f);
    
    // SIMD optimizations
    void addVectorsAVX2(const float* a, const float* b, float* c, int n);
    void scaleVectorAVX2(float* x, float scale, int n);
};

// Benchmarking utilities
class Benchmark {
public:
    static std::vector<std::string> getStandardPrompts();
    static PerformanceStats runBenchmark(InferenceEngine& engine, 
                                        const std::vector<std::string>& prompts);
    static void printResults(const PerformanceStats& stats);
    static bool validatePerformance(const PerformanceStats& stats, double target_tps = 30.0);
};

} // namespace tps
