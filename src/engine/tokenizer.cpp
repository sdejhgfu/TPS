#include "tokenizer.h"
#include "../utils/logger.h"
#include <fstream>
#include <sstream>

namespace hpie {

bool Tokenizer::Load(const std::string& tokenizer_path) {
    // For this implementation, we'll use a simplified tokenizer
    // In a real implementation, you would load the actual tokenizer files
    
    InitializeDefaultVocab();
    loaded_ = true;
    
    Logger::Info("Tokenizer loaded with %zu vocabulary entries", vocab_size_);
    return true;
}

void Tokenizer::InitializeDefaultVocab() {
    // Simplified vocabulary initialization
    // In practice, this would load from the actual tokenizer files
    
    token_to_id_.clear();
    id_to_token_.clear();
    
    // Add special tokens
    token_to_id_["<|endoftext|>"] = eos_token_id_;
    id_to_token_[eos_token_id_] = "<|endoftext|>";
    
    token_to_id_["<|pad|>"] = pad_token_id_;
    id_to_token_[pad_token_id_] = "<|pad|>";
    
    // Add basic ASCII characters
    for (int i = 32; i < 127; ++i) {
        std::string token(1, static_cast<char>(i));
        uint32_t id = static_cast<uint32_t>(i);
        token_to_id_[token] = id;
        id_to_token_[id] = token;
    }
    
    // Add some common words and subwords
    std::vector<std::string> common_tokens = {
        " the", " and", " of", " to", " a", " in", " is", " it", " you", " that",
        " he", " was", " for", " on", " are", " as", " with", " his", " they",
        " at", " be", " this", " have", " from", " or", " one", " had", " by",
        " word", " but", " not", " what", " all", " were", " we", " when",
        " your", " can", " said", " there", " each", " which", " she", " do",
        " how", " their", " if", " will", " up", " other", " about", " out",
        " many", " then", " them", " these", " so", " some", " her", " would",
        " make", " like", " into", " him", " time", " has", " two", " more",
        " go", " no", " way", " could", " my", " than", " first", " been",
        " call", " who", " oil", " sit", " now", " find", " long", " down",
        " day", " did", " get", " come", " made", " may", " part"
    };
    
    for (size_t i = 0; i < common_tokens.size(); ++i) {
        uint32_t id = static_cast<uint32_t>(127 + i);
        token_to_id_[common_tokens[i]] = id;
        id_to_token_[id] = common_tokens[i];
    }
    
    vocab_size_ = token_to_id_.size();
}

std::vector<uint32_t> Tokenizer::Encode(const std::string& text) {
    if (!loaded_) {
        Logger::Error("Tokenizer not loaded");
        return {};
    }
    
    std::vector<uint32_t> tokens;
    
    // Simple tokenization - split by whitespace and map to IDs
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // Try to find exact match first
        auto it = token_to_id_.find(word);
        if (it != token_to_id_.end()) {
            tokens.push_back(it->second);
        } else {
            // Try to find partial matches (subword tokenization)
            bool found = false;
            size_t start = 0;
            
            while (start < word.length()) {
                size_t best_match_len = 0;
                uint32_t best_match_id = 0;
                
                // Find longest matching subword
                for (const auto& pair : token_to_id_) {
                    const std::string& token = pair.first;
                    if (word.substr(start).starts_with(token) && 
                        token.length() > best_match_len) {
                        best_match_len = token.length();
                        best_match_id = pair.second;
                    }
                }
                
                if (best_match_len > 0) {
                    tokens.push_back(best_match_id);
                    start += best_match_len;
                    found = true;
                } else {
                    // Character-level fallback
                    tokens.push_back(static_cast<uint32_t>(word[start]));
                    start++;
                }
            }
        }
    }
    
    // Add EOS token
    tokens.push_back(eos_token_id_);
    
    return tokens;
}

std::string Tokenizer::Decode(const std::vector<uint32_t>& tokens) {
    if (!loaded_) {
        Logger::Error("Tokenizer not loaded");
        return "";
    }
    
    std::string result;
    
    for (uint32_t token_id : tokens) {
        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            result += it->second;
        } else {
            // Fallback for unknown tokens
            result += "<UNK>";
        }
    }
    
    return result;
}

} // namespace hpie
