#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace hpie {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Model loading
    bool Load(const std::string& tokenizer_path);
    bool IsLoaded() const { return loaded_; }

    // Tokenization
    std::vector<uint32_t> Encode(const std::string& text);
    std::string Decode(const std::vector<uint32_t>& tokens);
    
    // Token utilities
    uint32_t GetEOSTokenId() const { return eos_token_id_; }
    uint32_t GetPADTokenId() const { return pad_token_id_; }
    size_t GetVocabSize() const { return vocab_size_; }

private:
    bool loaded_ = false;
    uint32_t eos_token_id_ = 2;
    uint32_t pad_token_id_ = 0;
    size_t vocab_size_ = 50257;
    
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::unordered_map<uint32_t, std::string> id_to_token_;
    
    void InitializeDefaultVocab();
};

} // namespace hpie
