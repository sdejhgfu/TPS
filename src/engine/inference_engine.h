#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

#include "../memory/memory_manager.h"
#include "../cache/kv_cache.h"
#include "../quantization/quantizer.h"
#include "tokenizer.h"
#include "model_loader.h"
#include "sampler.h"

namespace hpie {

struct InferenceConfig {
    size_t max_context_length = 4096;
    size_t max_new_tokens = 512;
    size_t batch_size = 1;
    size_t num_threads = 0; // 0 = auto-detect
    size_t max_memory_mb = 0; // 0 = unlimited
    bool enable_caching = true;
    bool enable_quantization = true;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    bool use_simd = true;
    bool enable_prefill_optimization = true;
    bool use_lightweight_model = false; // Use reduced model for testing
};

struct Token {
    uint32_t id;
    std::string text;
    float probability;
    float logit;
};

struct InferenceStats {
    std::chrono::microseconds prefill_time;
    std::chrono::microseconds decode_time;
    std::chrono::microseconds total_time;
    size_t tokens_generated;
    size_t memory_used;
    float cache_hit_rate;
    double tokens_per_second;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const InferenceConfig& config = InferenceConfig{});
    ~InferenceEngine();

    // Model loading and initialization
    bool LoadModel(const std::string& model_path);
    bool IsModelLoaded() const { return model_loaded_; }

    // Inference methods
    std::vector<Token> Generate(const std::string& prompt);
    std::vector<Token> Generate(const std::vector<uint32_t>& token_ids);
    
    // Batch inference
    std::vector<std::vector<Token>> GenerateBatch(
        const std::vector<std::string>& prompts);

    // Performance monitoring
    InferenceStats GetLastStats() const { return last_stats_; }
    void ResetStats();

    // Configuration
    void UpdateConfig(const InferenceConfig& config);
    const InferenceConfig& GetConfig() const { return config_; }

    // Memory and cache management
    void ClearCache();
    size_t GetCacheSize() const;
    void OptimizeMemory();

private:
    // Core components
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<Quantizer> quantizer_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<Sampler> sampler_;

    // Configuration
    InferenceConfig config_;
    
    // State
    bool model_loaded_;
    std::vector<float> hidden_states_;
    std::vector<float> attention_weights_;
    
    // Threading
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> should_stop_;
    std::mutex inference_mutex_;

    // Performance tracking
    mutable InferenceStats last_stats_;
    mutable std::mutex stats_mutex_;

    // Internal methods
    bool InitializeComponents();
    void InitializeThreading();
    void ShutdownThreading();
    
    std::vector<Token> GenerateInternal(
        const std::vector<uint32_t>& input_tokens);
    
    void PrefillPhase(const std::vector<uint32_t>& input_tokens);
    Token DecodePhase(const std::vector<float>& logits);
    
    void UpdateStats(const InferenceStats& stats);
    
    // SIMD optimization helpers
    void OptimizedMatrixMultiply(
        const float* a, const float* b, float* c,
        size_t m, size_t n, size_t k);
    
    void OptimizedAttention(
        const float* query, const float* key, const float* value,
        float* output, size_t seq_len, size_t hidden_dim);
};

} // namespace hpie

