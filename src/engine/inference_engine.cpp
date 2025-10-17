#include "inference_engine.h"
#include "../utils/logger.h"
#include "../utils/timer.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <immintrin.h> // For SIMD intrinsics

namespace hpie {

InferenceEngine::InferenceEngine(const InferenceConfig& config)
    : config_(config), model_loaded_(false), should_stop_(false) {
    if (!InitializeComponents()) {
        Logger::Error("Failed to initialize inference engine components");
    }
}

InferenceEngine::~InferenceEngine() {
    ShutdownThreading();
}

bool InferenceEngine::InitializeComponents() {
    try {
        // Initialize memory manager
        memory_manager_ = std::make_unique<MemoryManager>(config_.max_context_length * 2);
        
        // Initialize cache
        if (config_.enable_caching) {
            kv_cache_ = std::make_unique<KVCache>(
                config_.max_context_length,
                memory_manager_.get()
            );
        }
        
        // Initialize quantizer (use DynamicQuantizer as default)
        if (config_.enable_quantization) {
            quantizer_ = std::make_unique<DynamicQuantizer>();
        }
        
        // Initialize tokenizer
        tokenizer_ = std::make_unique<Tokenizer>();
        
        // Initialize model loader
        model_loader_ = std::make_unique<ModelLoader>(memory_manager_.get());
        
        // Initialize sampler
        sampler_ = std::make_unique<Sampler>(
            config_.temperature,
            config_.top_p,
            config_.top_k
        );
        
        // Initialize threading
        InitializeThreading();
        
        Logger::Info("Inference engine components initialized successfully");
        return true;
    } catch (const std::exception& e) {
        Logger::Error("Failed to initialize components: %s", e.what());
        return false;
    }
}

void InferenceEngine::InitializeThreading() {
    if (config_.num_threads == 0) {
        config_.num_threads = std::thread::hardware_concurrency();
    }
    
    worker_threads_.reserve(config_.num_threads);
    Logger::Info("Initialized %zu worker threads", config_.num_threads);
}

void InferenceEngine::ShutdownThreading() {
    should_stop_ = true;
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

bool InferenceEngine::LoadModel(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    Timer timer;
    Logger::Info("Loading model from: %s", model_path.c_str());
    
    if (!model_loader_->LoadModel(model_path)) {
        Logger::Error("Failed to load model from %s", model_path.c_str());
        return false;
    }
    
    if (!tokenizer_->Load(model_path + "/tokenizer.json")) {
        Logger::Error("Failed to load tokenizer");
        return false;
    }
    
    // Allocate memory for hidden states and attention weights
    size_t hidden_size = model_loader_->GetHiddenSize();
    hidden_states_.resize(config_.max_context_length * hidden_size);
    attention_weights_.resize(config_.max_context_length * config_.max_context_length);
    
    model_loaded_ = true;
    
    auto load_time = timer.Elapsed();
    Logger::Info("Model loaded successfully in %.2f ms", 
                 load_time.count() / 1000.0);
    
    return true;
}

std::vector<Token> InferenceEngine::Generate(const std::string& prompt) {
    if (!model_loaded_) {
        Logger::Error("Model not loaded");
        return {};
    }
    
    // Tokenize input
    auto input_tokens = tokenizer_->Encode(prompt);
    if (input_tokens.empty()) {
        Logger::Error("Failed to tokenize input");
        return {};
    }
    
    return GenerateInternal(input_tokens);
}

std::vector<Token> InferenceEngine::Generate(
    const std::vector<uint32_t>& token_ids) {
    if (!model_loaded_) {
        Logger::Error("Model not loaded");
        return {};
    }
    
    return GenerateInternal(token_ids);
}

std::vector<Token> InferenceEngine::GenerateInternal(
    const std::vector<uint32_t>& input_tokens) {
    
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    Timer total_timer;
    InferenceStats stats = {};
    
    // Prefill phase
    Timer prefill_timer;
    PrefillPhase(input_tokens);
    stats.prefill_time = prefill_timer.Elapsed();
    
    // Decode phase
    Timer decode_timer;
    std::vector<Token> generated_tokens;
    generated_tokens.reserve(config_.max_new_tokens);
    
    size_t current_length = input_tokens.size();
    
    for (size_t i = 0; i < config_.max_new_tokens; ++i) {
        // Get logits for next token
        auto logits = model_loader_->GetLogits(current_length);
        
        // Sample next token
        auto sampled_token = DecodePhase(logits);
        generated_tokens.push_back(sampled_token);
        
        // Update KV cache
        if (kv_cache_) {
            kv_cache_->Update(current_length, sampled_token.id);
        }
        
        current_length++;
        
        // Check for EOS token
        if (sampled_token.id == tokenizer_->GetEOSTokenId()) {
            break;
        }
    }
    
    stats.decode_time = decode_timer.Elapsed();
    stats.total_time = total_timer.Elapsed();
    stats.tokens_generated = generated_tokens.size();
    stats.memory_used = memory_manager_->GetUsedMemory();
    stats.cache_hit_rate = kv_cache_ ? kv_cache_->GetHitRate() : 0.0f;
    
    // Calculate TPS
    if (stats.total_time.count() > 0) {
        stats.tokens_per_second = 
            static_cast<double>(stats.tokens_generated) * 1000000.0 / 
            stats.total_time.count();
    }
    
    UpdateStats(stats);
    
    Logger::Info("Generated %zu tokens in %.2f ms (%.1f TPS)",
                 stats.tokens_generated,
                 stats.total_time.count() / 1000.0,
                 stats.tokens_per_second);
    
    return generated_tokens;
}

void InferenceEngine::PrefillPhase(const std::vector<uint32_t>& input_tokens) {
    size_t seq_len = input_tokens.size();
    size_t hidden_size = model_loader_->GetHiddenSize();
    
    // Process input tokens through embedding layer
    for (size_t i = 0; i < seq_len; ++i) {
        auto embedding = model_loader_->GetEmbedding(input_tokens[i]);
        
        // Copy to hidden states
        std::memcpy(&hidden_states_[i * hidden_size],
                   embedding.data(),
                   hidden_size * sizeof(float));
    }
    
    // Process through transformer layers
    for (size_t layer = 0; layer < model_loader_->GetNumLayers(); ++layer) {
        // Self-attention
        if (config_.use_simd) {
            OptimizedAttention(
                &hidden_states_[0],
                &hidden_states_[0], // Same for self-attention
                &hidden_states_[0],
                &attention_weights_[0],
                seq_len,
                hidden_size
            );
        } else {
            // Standard attention computation
            model_loader_->ComputeAttention(layer, hidden_states_, attention_weights_);
        }
        
        // Feed-forward network
        model_loader_->ComputeFeedForward(layer, hidden_states_);
        
        // Layer normalization and residual connections
        model_loader_->ApplyLayerNorm(layer, hidden_states_);
    }
    
    // Update KV cache with computed keys and values
    if (kv_cache_) {
        kv_cache_->StorePrefill(input_tokens.size(), hidden_states_);
    }
}

Token InferenceEngine::DecodePhase(const std::vector<float>& logits) {
    // Apply sampling strategy
    auto sampled_id = sampler_->Sample(logits);
    
    // Get token text
    auto token_text = tokenizer_->Decode({sampled_id});
    
    // Calculate probability (simplified)
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float probability = std::exp(logits[sampled_id] - max_logit);
    
    return {
        .id = sampled_id,
        .text = token_text,
        .probability = probability,
        .logit = logits[sampled_id]
    };
}

void InferenceEngine::OptimizedMatrixMultiply(
    const float* a, const float* b, float* c,
    size_t m, size_t n, size_t k) {
    
    // Use SIMD for optimized matrix multiplication
    // #pragma omp parallel for  // Disabled: OpenMP not available
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            
            for (size_t l = 0; l < k; ++l) {
                __m256 va = _mm256_broadcast_ss(&a[i * k + l]);
                __m256 vb = _mm256_loadu_ps(&b[l * n + j]);
                sum = _mm256_fmadd_ps(va, vb, sum);
            }
            
            _mm256_storeu_ps(&c[i * n + j], sum);
        }
    }
}

void InferenceEngine::OptimizedAttention(
    const float* query, const float* key, const float* value,
    float* output, size_t seq_len, size_t hidden_dim) {
    
    // Optimized attention computation using SIMD
    const float scale = 1.0f / std::sqrt(static_cast<float>(hidden_dim));
    
    // Suppress unused parameter warning
    (void)value;
    
    // #pragma omp parallel for  // Disabled: OpenMP not available
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; j += 8) {
            // __m256 qv = _mm256_loadu_ps(&query[i * hidden_dim]);  // Unused
            __m256 score_sum = _mm256_setzero_ps();
            
            for (size_t k = 0; k < hidden_dim; k += 8) {
                __m256 qk = _mm256_loadu_ps(&query[i * hidden_dim + k]);
                __m256 kv = _mm256_loadu_ps(&key[j * hidden_dim + k]);
                score_sum = _mm256_fmadd_ps(qk, kv, score_sum);
            }
            
            // Apply scaling and store attention weights
            __m256 scaled = _mm256_mul_ps(score_sum, _mm256_set1_ps(scale));
            _mm256_storeu_ps(&output[i * seq_len + j], scaled);
        }
    }
}

std::vector<std::vector<Token>> InferenceEngine::GenerateBatch(
    const std::vector<std::string>& prompts) {
    
    std::vector<std::vector<Token>> results;
    results.reserve(prompts.size());
    
    // Process batch in parallel
    // #pragma omp parallel for  // Disabled: OpenMP not available
    for (size_t i = 0; i < prompts.size(); ++i) {
        results[i] = Generate(prompts[i]);
    }
    
    return results;
}

void InferenceEngine::UpdateConfig(const InferenceConfig& config) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    config_ = config;
    
    // Update sampler parameters
    if (sampler_) {
        sampler_->UpdateParameters(config.temperature, config.top_p, config.top_k);
    }
    
    Logger::Info("Configuration updated");
}

void InferenceEngine::ClearCache() {
    if (kv_cache_) {
        kv_cache_->Clear();
    }
    Logger::Info("Cache cleared");
}

size_t InferenceEngine::GetCacheSize() const {
    return kv_cache_ ? kv_cache_->GetSize() : 0;
}

void InferenceEngine::OptimizeMemory() {
    if (memory_manager_) {
        memory_manager_->Defragment();
    }
    Logger::Info("Memory optimized");
}

void InferenceEngine::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    last_stats_ = InferenceStats{};
}

void InferenceEngine::UpdateStats(const InferenceStats& stats) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    last_stats_ = stats;
}

} // namespace hpie

