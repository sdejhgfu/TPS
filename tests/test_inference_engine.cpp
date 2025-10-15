#include "../src/engine/inference_engine.h"
#include "../src/utils/logger.h"
#include <cassert>
#include <iostream>

void TestInferenceEngine() {
    std::cout << "Testing Inference Engine...\n";
    
    // Test engine creation
    hpie::InferenceConfig config;
    config.max_new_tokens = 10;
    config.temperature = 0.7f;
    
    hpie::InferenceEngine engine(config);
    assert(!engine.IsModelLoaded());
    
    std::cout << "✓ Engine creation test passed\n";
}

void TestTokenizer() {
    std::cout << "Testing Tokenizer...\n";
    
    hpie::Tokenizer tokenizer;
    assert(!tokenizer.IsLoaded());
    
    // Test basic tokenization
    auto tokens = tokenizer.Encode("Hello world");
    assert(!tokens.empty());
    
    std::string decoded = tokenizer.Decode(tokens);
    assert(!decoded.empty());
    
    std::cout << "✓ Tokenizer test passed\n";
}

void TestMemoryManager() {
    std::cout << "Testing Memory Manager...\n";
    
    hpie::MemoryManager manager(1024 * 1024); // 1MB
    
    void* ptr = manager.Allocate(1024);
    assert(ptr != nullptr);
    
    manager.Deallocate(ptr);
    
    auto stats = manager.GetStats();
    assert(stats.total_memory > 0);
    
    std::cout << "✓ Memory Manager test passed\n";
}

void TestQuantizer() {
    std::cout << "Testing Quantizer...\n";
    
    std::vector<float> test_data = {1.0f, -1.0f, 0.5f, -0.5f, 2.0f};
    
    hpie::Int8Quantizer quantizer;
    auto quantized = quantizer.QuantizeToInt8(test_data);
    
    assert(!quantized.GetData() == nullptr);
    assert(quantized.GetSize() == test_data.size());
    
    auto dequantized = quantizer.DequantizeFromInt8(quantized);
    assert(dequantized.size() == test_data.size());
    
    std::cout << "✓ Quantizer test passed\n";
}

void TestKVCache() {
    std::cout << "Testing KV Cache...\n";
    
    hpie::KVCache cache(1024);
    
    std::vector<uint32_t> key = {1, 2, 3};
    std::vector<float> value = {1.0f, 2.0f, 3.0f};
    
    bool stored = cache.Store(key, value);
    assert(stored);
    
    auto retrieved = cache.Retrieve(key);
    assert(!retrieved.empty());
    
    assert(cache.Contains(key));
    
    std::cout << "✓ KV Cache test passed\n";
}

int main() {
    std::cout << "Running High-Performance Inference Engine Tests\n";
    std::cout << "==============================================\n\n";
    
    hpie::Logger::Initialize("test.log", hpie::LogLevel::INFO);
    
    try {
        TestInferenceEngine();
        TestTokenizer();
        TestMemoryManager();
        TestQuantizer();
        TestKVCache();
        
        std::cout << "\n✓ All tests passed!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
    
    hpie::Logger::Shutdown();
}
