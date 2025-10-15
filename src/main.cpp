#include "engine/inference_engine.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

using namespace hpie;

// Standardized test prompts from the challenge
const std::vector<std::string> TEST_PROMPTS = {
    "What is AI?",
    
    "Explain artificial intelligence, machine learning, and deep learning. Describe their differences and applications in detail.",
    
    "Write a comprehensive essay about the history of computing, covering major milestones from the abacus to modern quantum computers. Include key inventors, breakthrough technologies, and the impact on society. Discuss the evolution of programming languages, operating systems, and computer architectures.",
    
    "Summarize the differences between inference, training, and fine-tuning for large language models in exactly 6 bullet points. Each bullet ≤25 words.",
    
    "Write a robust Python script that:\n- reads NDJSON from stdin,\n- filters records where rule_level ≥ 10 or mitreid is not null,\n- outputs NDJSON,\n- includes argparse, logging, and unit tests.",
    
    "Produce a 900–1,200 word mini-report: 'History of Transformer Models (2017–2025)'. Use section headers, numbered references, and a concluding 'Key Takeaways' list of 10 items.",
    
    "Translate this to English and then explain it for a non-technical audience in 5 bullets:\n« Les compteurs intelligents doivent conserver au moins 13 mois de profils de charge et journaliser les événements de fraude. »",
    
    "Return ONLY valid JSON (no extra text) summarizing pros/cons of CPU-only vs GPU inference for a 20B parameter model. Fields: {\"hardware\": [..], \"throughput_tpm\": \"string\", \"latency_ms\": \"string\", \"energy_tradeoffs\": [..], \"when_to_choose\": {\"cpu\": [..], \"gpu\": [..]}}.",
    
    "Write a detailed design doc (~800 words) for a PostgreSQL + TimescaleDB pipeline that ingests 4B log rows/day, with partitioning, continuous aggregates, and a watermarked incremental job. Include sample DDL and 3 optimized queries.",
    
    "Generate a step-by-step tutorial that builds a retrieval-augmented generation (RAG) prototype with Ollama + FAISS. Include: environment setup, data ingestion, embedding choice rationale, retrieval API, evaluation checklist, and a final 'Gotchas' section."
};

void PrintUsage(const char* program_name) {
    std::cout << "High-Performance Inference Engine\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <path>        Path to model directory\n";
    std::cout << "  --prompt <text>       Single prompt to process\n";
    std::cout << "  --benchmark          Run full benchmark suite\n";
    std::cout << "  --max-tokens <n>      Maximum tokens to generate (default: 512)\n";
    std::cout << "  --temperature <f>     Sampling temperature (default: 0.7)\n";
    std::cout << "  --threads <n>         Number of threads (default: auto)\n";
    std::cout << "  --enable-cache       Enable KV caching\n";
    std::cout << "  --enable-quant       Enable quantization\n";
    std::cout << "  --profile            Enable performance profiling\n";
    std::cout << "  --verbose            Enable verbose logging\n";
    std::cout << "  --help               Show this help message\n\n";
}

void RunBenchmark(InferenceEngine& engine) {
    std::cout << "\n=== Running Benchmark Suite ===\n\n";
    
    Timer total_timer;
    size_t total_tokens_generated = 0;
    std::chrono::microseconds total_elapsed_time(0);
    
    std::vector<double> individual_tps;
    
    for (size_t i = 0; i < TEST_PROMPTS.size(); ++i) {
        const std::string& prompt = TEST_PROMPTS[i];
        
        std::cout << "Prompt " << (i + 1) << "/" << TEST_PROMPTS.size() << ": ";
        std::cout << prompt.substr(0, 50) << (prompt.length() > 50 ? "..." : "") << "\n";
        
        Timer prompt_timer;
        auto tokens = engine.Generate(prompt);
        auto elapsed = prompt_timer.Elapsed();
        
        total_tokens_generated += tokens.size();
        total_elapsed_time += elapsed;
        
        double tps = tokens.size() / (elapsed.count() / 1000000.0);
        individual_tps.push_back(tps);
        
        std::cout << "  Generated: " << tokens.size() << " tokens\n";
        std::cout << "  Time: " << (elapsed.count() / 1000.0) << " ms\n";
        std::cout << "  TPS: " << std::fixed << std::setprecision(2) << tps << "\n\n";
        
        // Print first few tokens as sample
        std::cout << "  Sample output: ";
        for (size_t j = 0; j < std::min(size_t(5), tokens.size()); ++j) {
            std::cout << tokens[j].text << " ";
        }
        std::cout << "\n\n";
    }
    
    auto total_benchmark_time = total_timer.Elapsed();
    double average_tps = total_tokens_generated / (total_elapsed_time.count() / 1000000.0);
    
    std::cout << "=== Benchmark Results ===\n";
    std::cout << "Total prompts processed: " << TEST_PROMPTS.size() << "\n";
    std::cout << "Total tokens generated: " << total_tokens_generated << "\n";
    std::cout << "Total processing time: " << (total_benchmark_time.count() / 1000.0) << " ms\n";
    std::cout << "Average TPS: " << std::fixed << std::setprecision(2) << average_tps << "\n";
    
    // Calculate statistics
    std::sort(individual_tps.begin(), individual_tps.end());
    double min_tps = individual_tps.front();
    double max_tps = individual_tps.back();
    double median_tps = individual_tps[individual_tps.size() / 2];
    
    std::cout << "TPS Statistics:\n";
    std::cout << "  Min: " << std::fixed << std::setprecision(2) << min_tps << "\n";
    std::cout << "  Max: " << std::fixed << std::setprecision(2) << max_tps << "\n";
    std::cout << "  Median: " << std::fixed << std::setprecision(2) << median_tps << "\n";
    
    // Performance comparison
    const double BASELINE_TPS = 30.0;
    double improvement = (average_tps / BASELINE_TPS - 1.0) * 100.0;
    
    std::cout << "\nPerformance vs Baseline (30 TPS):\n";
    std::cout << "  Improvement: " << std::fixed << std::setprecision(1) << improvement << "%\n";
    
    if (average_tps > BASELINE_TPS) {
        std::cout << "  ✓ EXCEEDS BASELINE REQUIREMENT\n";
    } else {
        std::cout << "  ✗ Below baseline requirement\n";
    }
    
    // Get engine statistics
    auto stats = engine.GetLastStats();
    std::cout << "\nEngine Statistics:\n";
    std::cout << "  Memory usage: " << (stats.memory_used / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  Cache hit rate: " << (stats.cache_hit_rate * 100.0) << "%\n";
    std::cout << "  Prefill time: " << (stats.prefill_time.count() / 1000.0) << " ms\n";
    std::cout << "  Decode time: " << (stats.decode_time.count() / 1000.0) << " ms\n";
}

int main(int argc, char* argv[]) {
    // Initialize logger
    Logger::Initialize("inference_engine.log", LogLevel::INFO);
    
    std::string model_path;
    std::string single_prompt;
    bool run_benchmark = false;
    bool enable_profile = false;
    bool verbose = false;
    
    InferenceConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            single_prompt = argv[++i];
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            config.max_new_tokens = std::stoul(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoul(argv[++i]);
        } else if (arg == "--enable-cache") {
            config.enable_caching = true;
        } else if (arg == "--enable-quant") {
            config.enable_quantization = true;
        } else if (arg == "--profile") {
            enable_profile = true;
        } else if (arg == "--verbose") {
            verbose = true;
            Logger::SetLevel(LogLevel::DEBUG);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }
    
    if (verbose) {
        Logger::Info("High-Performance Inference Engine starting...");
        Logger::Info("Configuration: max_tokens=%zu, temperature=%.2f, threads=%zu", 
                     config.max_new_tokens, config.temperature, config.num_threads);
    }
    
    try {
        // Create inference engine
        InferenceEngine engine(config);
        
        if (model_path.empty()) {
            std::cerr << "Error: Model path required. Use --model <path>\n";
            PrintUsage(argv[0]);
            return 1;
        }
        
        // Load model
        std::cout << "Loading model from: " << model_path << "\n";
        if (!engine.LoadModel(model_path)) {
            std::cerr << "Failed to load model from: " << model_path << "\n";
            return 1;
        }
        
        std::cout << "Model loaded successfully!\n\n";
        
        if (enable_profile) {
            PerformanceProfiler::Instance().StartTimer("total_inference");
        }
        
        if (run_benchmark) {
            RunBenchmark(engine);
        } else if (!single_prompt.empty()) {
            std::cout << "Processing single prompt:\n";
            std::cout << single_prompt << "\n\n";
            
            auto tokens = engine.Generate(single_prompt);
            
            std::cout << "Generated " << tokens.size() << " tokens:\n";
            for (const auto& token : tokens) {
                std::cout << token.text;
            }
            std::cout << "\n\n";
            
            auto stats = engine.GetLastStats();
            std::cout << "Statistics:\n";
            std::cout << "  TPS: " << std::fixed << std::setprecision(2) << stats.tokens_per_second << "\n";
            std::cout << "  Time: " << (stats.total_time.count() / 1000.0) << " ms\n";
            std::cout << "  Memory: " << (stats.memory_used / 1024.0 / 1024.0) << " MB\n";
        } else {
            // Interactive mode
            std::cout << "Interactive mode. Type prompts (or 'quit' to exit):\n\n";
            
            std::string prompt;
            while (std::getline(std::cin, prompt)) {
                if (prompt == "quit" || prompt == "exit") {
                    break;
                }
                
                if (prompt.empty()) {
                    continue;
                }
                
                std::cout << "\nGenerating...\n";
                auto tokens = engine.Generate(prompt);
                
                std::cout << "\nResponse:\n";
                for (const auto& token : tokens) {
                    std::cout << token.text;
                }
                std::cout << "\n\n";
                
                auto stats = engine.GetLastStats();
                std::cout << "(TPS: " << std::fixed << std::setprecision(1) << stats.tokens_per_second 
                          << ", " << (stats.total_time.count() / 1000.0) << "ms)\n\n";
            }
        }
        
        if (enable_profile) {
            PerformanceProfiler::Instance().EndTimer("total_inference");
            PerformanceProfiler::Instance().PrintReport();
        }
        
    } catch (const std::exception& e) {
        Logger::Error("Fatal error: %s", e.what());
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    
    Logger::Shutdown();
    return 0;
}
