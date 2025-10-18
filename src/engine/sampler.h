#pragma once

#include <vector>
#include <random>

namespace hpie {

class Sampler {
public:
    Sampler(float temperature = 0.7f, float top_p = 0.9f, int top_k = 50);
    ~Sampler() = default;

    // Sampling methods
    uint32_t Sample(const std::vector<float>& logits);
    
    // Parameter updates
    void UpdateParameters(float temperature, float top_p, int top_k);
    void SetSeed(uint32_t seed);  // For reproducible results
    
    // Sampling strategies
    uint32_t GreedySample(const std::vector<float>& logits);
    uint32_t RandomSample(const std::vector<float>& logits);
    uint32_t TopKSample(const std::vector<float>& logits, int k);
    uint32_t TopPSample(const std::vector<float>& logits, float p);
    uint32_t NucleusSample(const std::vector<float>& logits);

private:
    float temperature_;
    float top_p_;
    int top_k_;
    
    std::mt19937 rng_;
    
    // Utility functions
    std::vector<float> ApplyTemperature(const std::vector<float>& logits);
    std::vector<float> Softmax(const std::vector<float>& logits);
    std::vector<size_t> GetTopKIndices(const std::vector<float>& values, int k);
    float GetCumulativeProbability(const std::vector<float>& probs, float threshold);
};

} // namespace hpie
