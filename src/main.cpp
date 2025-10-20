#include "inference_engine.h"
#include <iostream>
#include <iomanip>

using namespace tps;

int main() {
    InferenceEngine engine;
    
    // Initialize the engine
    if (!engine.initialize()) {
        std::cerr << "Failed to initialize inference engine" << std::endl;
        return 1;
    }
    
    // The 10 standardized prompts
    std::vector<std::string> prompts = Benchmark::getStandardPrompts();

    std::cout << "=== TPS Inference Engine Benchmark ===" << std::endl;
    std::cout << "Hardware: 2-socket, 8-core (16 total), 32GB DDR5" << std::endl;
    std::cout << "Model: Lightweight transformer optimized for 30+ TPS" << std::endl;
    std::cout << "Dependencies: Zero external dependencies" << std::endl;
    std::cout << "===========================================" << std::endl;

    // Run benchmark
    const auto stats = engine.runBatch(prompts);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Generated tokens: " << stats.generatedTokens << std::endl;
    std::cout << "Elapsed seconds: " << stats.elapsedSeconds << std::endl;
    std::cout << "Average TPS: " << stats.tps << std::endl;
    std::cout << "Memory used: " << stats.memoryUsed << " bytes" << std::endl;
    std::cout << "Cache hit ratio: " << stats.cacheHitRatio << std::endl;
    
    if (stats.tps >= 30.0) {
        std::cout << "✅ TARGET ACHIEVED: " << stats.tps << " TPS >= 30 TPS" << std::endl;
        std::cout << "🚀 PERFORMANCE: " << (stats.tps / 30.0) << "x faster than target!" << std::endl;
    } else {
        std::cout << "❌ TARGET MISSED: " << stats.tps << " TPS < 30 TPS" << std::endl;
    }
    
    std::cout << "===========================================" << std::endl;

    // Show sample outputs
    std::cout << "\nSample outputs:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "\n--- Prompt " << (i+1) << " ---" << std::endl;
        std::cout << "Input: " << prompts[i] << std::endl;
        std::cout << "Output: " << engine.generate(prompts[i], 30) << std::endl;
    }

    // Print detailed benchmark results
    Benchmark::printResults(stats);
    
    return 0;
}