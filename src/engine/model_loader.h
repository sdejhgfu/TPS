#pragma once

#include <vector>
#include <string>
#include <memory>

namespace hpie {

class MemoryManager;

class ModelLoader {
public:
    explicit ModelLoader(MemoryManager* memory_manager);
    ~ModelLoader();

    // Model loading
    bool LoadModel(const std::string& model_path);
    bool IsModelLoaded() const { return model_loaded_; }

    // Model information
    size_t GetHiddenSize() const { return hidden_size_; }
    size_t GetNumLayers() const { return num_layers_; }
    size_t GetNumHeads() const { return num_heads_; }
    size_t GetVocabSize() const { return vocab_size_; }
    size_t GetMaxSequenceLength() const { return max_sequence_length_; }

    // Embedding operations
    std::vector<float> GetEmbedding(uint32_t token_id);
    std::vector<float> GetLogits(size_t sequence_length);

    // Transformer operations
    void ComputeAttention(size_t layer, const std::vector<float>& input, 
                         std::vector<float>& attention_weights);
    void ComputeFeedForward(size_t layer, std::vector<float>& hidden_states);
    void ApplyLayerNorm(size_t layer, std::vector<float>& hidden_states);

    // Memory management
    void OptimizeMemoryLayout();
    size_t GetModelSize() const { return model_size_; }

private:
    MemoryManager* memory_manager_;
    bool model_loaded_;
    
    // Model architecture parameters
    size_t hidden_size_;
    size_t num_layers_;
    size_t num_heads_;
    size_t vocab_size_;
    size_t max_sequence_length_;
    size_t model_size_;
    
    // Model weights (simplified representation)
    std::vector<float> embeddings_;
    std::vector<std::vector<float>> attention_weights_;
    std::vector<std::vector<float>> feed_forward_weights_;
    std::vector<std::vector<float>> layer_norm_weights_;
    std::vector<float> output_weights_;
    
    // Internal methods
    bool LoadWeights(const std::string& weights_path);
    void InitializeRandomWeights();
    void AllocateModelMemory();
    size_t CalculateModelSize() const;
};

} // namespace hpie
