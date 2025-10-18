#include "sampler.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace hpie {

Sampler::Sampler(float temperature, float top_p, int top_k)
    : temperature_(temperature), top_p_(top_p), top_k_(top_k),
      rng_(12345) {  // Fixed seed for reproducible results
}

void Sampler::SetSeed(uint32_t seed) {
    rng_.seed(seed);
}

uint32_t Sampler::Sample(const std::vector<float>& logits) {
    if (logits.empty()) {
        return 0;
    }
    
    // Apply temperature scaling
    auto scaled_logits = ApplyTemperature(logits);
    
    // Apply softmax to get probabilities
    auto probs = Softmax(scaled_logits);
    
    // Use nucleus sampling (top-p) as default
    return NucleusSample(scaled_logits);
}

void Sampler::UpdateParameters(float temperature, float top_p, int top_k) {
    temperature_ = std::max(0.1f, temperature);
    top_p_ = std::max(0.1f, std::min(1.0f, top_p));
    top_k_ = std::max(1, top_k);
}

uint32_t Sampler::GreedySample(const std::vector<float>& logits) {
    auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<uint32_t>(std::distance(logits.begin(), max_it));
}

uint32_t Sampler::RandomSample(const std::vector<float>& logits) {
    auto probs = Softmax(logits);
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float random_value = dist(rng_);
    
    float cumulative = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (random_value <= cumulative) {
            return static_cast<uint32_t>(i);
        }
    }
    
    return static_cast<uint32_t>(probs.size() - 1);
}

uint32_t Sampler::TopKSample(const std::vector<float>& logits, int k) {
    k = std::min(k, static_cast<int>(logits.size()));
    
    // Get top-k indices
    auto top_k_indices = GetTopKIndices(logits, k);
    
    // Create mask for top-k tokens
    std::vector<float> masked_logits = logits;
    for (size_t i = 0; i < masked_logits.size(); ++i) {
        if (std::find(top_k_indices.begin(), top_k_indices.end(), i) == top_k_indices.end()) {
            masked_logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
    
    // Sample from masked distribution
    return RandomSample(masked_logits);
}

uint32_t Sampler::TopPSample(const std::vector<float>& logits, float p) {
    // Sort logits in descending order
    std::vector<std::pair<float, size_t>> logit_indices;
    for (size_t i = 0; i < logits.size(); ++i) {
        logit_indices.emplace_back(logits[i], i);
    }
    
    std::sort(logit_indices.begin(), logit_indices.end(),
              [](const auto& a, const auto& b) {
                  return a.first > b.first;
              });
    
    // Apply softmax to get probabilities
    auto probs = Softmax(logits);
    
    // Find cumulative probability threshold
    float cumulative = 0.0f;
    size_t cutoff = 0;
    
    for (size_t i = 0; i < logit_indices.size(); ++i) {
        cumulative += probs[logit_indices[i].second];
        if (cumulative >= p) {
            cutoff = i + 1;
            break;
        }
    }
    
    // Create mask for top-p tokens
    std::vector<float> masked_logits = logits;
    for (size_t i = cutoff; i < logit_indices.size(); ++i) {
        masked_logits[logit_indices[i].second] = -std::numeric_limits<float>::infinity();
    }
    
    // Sample from masked distribution
    return RandomSample(masked_logits);
}

uint32_t Sampler::NucleusSample(const std::vector<float>& logits) {
    return TopPSample(logits, top_p_);
}

std::vector<float> Sampler::ApplyTemperature(const std::vector<float>& logits) {
    std::vector<float> scaled_logits = logits;
    
    if (temperature_ > 0.0f) {
        for (auto& logit : scaled_logits) {
            logit /= temperature_;
        }
    }
    
    return scaled_logits;
}

std::vector<float> Sampler::Softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    
    // Find maximum for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (auto& prob : probs) {
            prob /= sum;
        }
    }
    
    return probs;
}

std::vector<size_t> Sampler::GetTopKIndices(const std::vector<float>& values, int k) {
    std::vector<std::pair<float, size_t>> value_indices;
    for (size_t i = 0; i < values.size(); ++i) {
        value_indices.emplace_back(values[i], i);
    }
    
    // Sort in descending order
    std::sort(value_indices.begin(), value_indices.end(),
              [](const auto& a, const auto& b) {
                  return a.first > b.first;
              });
    
    // Extract top-k indices
    std::vector<size_t> top_k_indices;
    for (int i = 0; i < k && i < static_cast<int>(value_indices.size()); ++i) {
        top_k_indices.push_back(value_indices[i].second);
    }
    
    return top_k_indices;
}

} // namespace hpie
