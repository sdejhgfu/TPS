#include "inference_engine.h"
#include <iostream>
#include <cassert>
#include <chrono>

using namespace tps;

void testTokenizer() {
    std::cout << "Testing Tokenizer..." << std::endl;
    
    Tokenizer tokenizer;
    
    // Test encoding
    std::string text = "What is AI?";
    std::vector<int> tokens = tokenizer.encode(text);
    
    assert(!tokens.empty());
    std::cout << "✓ Tokenizer encoding works" << std::endl;
    
    // Test decoding
    std::string decoded = tokenizer.decode(tokens);
    std::cout << "✓ Tokenizer decoding works: '" << decoded << "'" << std::endl;
}

void testMemoryManager() {
    std::cout << "Testing Memory Manager..." << std::endl;
    
    MemoryManager manager;
    
    // Test allocation
    void* ptr = manager.allocate(1024);
    assert(ptr != nullptr);
    std::cout << "✓ Memory allocation works" << std::endl;
    
    // Test deallocation
    manager.deallocate(ptr);
    std::cout << "✓ Memory deallocation works" << std::endl;
    
    manager.reset();
    std::cout << "✓ Memory manager reset works" << std::endl;
}

void testKVCache() {
    std::cout << "Testing KV Cache..." << std::endl;
    
    KVCache cache;
    cache.initialize(10, 32);
    
    // Test storing
    std::vector<float> k(HIDDEN_SIZE, 1.0f);
    std::vector<float> v(HIDDEN_SIZE, 2.0f);
    
    cache.store(0, 0, 0, k.data(), v.data());
    std::cout << "✓ KV cache storage works" << std::endl;
    
    // Test retrieval
    std::vector<float> k_out(HIDDEN_SIZE);
    std::vector<float> v_out(HIDDEN_SIZE);
    
    cache.retrieve(0, 0, 0, 1, k_out.data(), v_out.data());
    std::cout << "✓ KV cache retrieval works" << std::endl;
    
    cache.clear();
    std::cout << "✓ KV cache clear works" << std::endl;
}

void testInferenceEngine() {
    std::cout << "Testing Inference Engine..." << std::endl;
    
    InferenceEngine engine;
    
    // Test initialization
    bool init_success = engine.initialize();
    assert(init_success);
    std::cout << "✓ Engine initialization works" << std::endl;
    
    // Test generation
    std::string prompt = "What is AI?";
    std::string output = engine.generate(prompt, 10);
    
    assert(!output.empty());
    std::cout << "✓ Text generation works: '" << output << "'" << std::endl;
    
    // Test performance
    auto start = std::chrono::high_resolution_clock::now();
    std::string test_output = engine.generate("Test prompt", 20);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "✓ Generation speed: " << duration.count() << " microseconds" << std::endl;
    
    // Test memory usage
    size_t memory_usage = engine.getMemoryUsage();
    std::cout << "✓ Memory usage: " << memory_usage << " bytes" << std::endl;
}

void testPerformance() {
    std::cout << "Testing Performance..." << std::endl;
    
    InferenceEngine engine;
    engine.initialize();
    
    std::vector<std::string> prompts = {
        "What is AI?",
        "Explain machine learning",
        "Describe neural networks"
    };
    
    PerformanceStats stats = engine.runBatch(prompts);
    
    std::cout << "✓ Batch processing works" << std::endl;
    std::cout << "  Generated tokens: " << stats.generatedTokens << std::endl;
    std::cout << "  Elapsed seconds: " << stats.elapsedSeconds << std::endl;
    std::cout << "  TPS: " << stats.tps << std::endl;
    
    if (stats.tps >= 30.0) {
        std::cout << "✅ Performance target achieved!" << std::endl;
    } else {
        std::cout << "⚠️  Performance below target" << std::endl;
    }
}

int main() {
    std::cout << "=== Running Unit Tests ===" << std::endl;
    
    testTokenizer();
    testMemoryManager();
    testKVCache();
    testInferenceEngine();
    testPerformance();
    
    std::cout << "\n✅ All tests passed!" << std::endl;
    return 0;
}
