#include "engine/inference_engine.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

using namespace hpie;

// Extended benchmark suite
class BenchmarkSuite {
public:
    struct BenchmarkResult {
        std::string prompt_name;
        std::string prompt;
        size_t tokens_generated;
        std::chrono::microseconds total_time;
        std::chrono::microseconds prefill_time;
        std::chrono::microseconds decode_time;
        double tokens_per_second;
        size_t memory_used;
        float cache_hit_rate;
        bool success;
        std::string error_message;
    };
    
    explicit BenchmarkSuite(InferenceEngine& engine) : engine_(engine) {}
    
    std::vector<BenchmarkResult> RunComprehensiveBenchmark() {
        std::vector<BenchmarkResult> results;
        
        // Standard test prompts
        std::vector<std::pair<std::string, std::string>> test_cases = {
            {"Simple Question", "What is AI?"},
            {"Technical Explanation", "Explain artificial intelligence, machine learning, and deep learning. Describe their differences and applications in detail."},
            {"Long Essay", "Write a comprehensive essay about the history of computing, covering major milestones from the abacus to modern quantum computers. Include key inventors, breakthrough technologies, and the impact on society."},
            {"Structured Output", "Summarize the differences between inference, training, and fine-tuning for large language models in exactly 6 bullet points. Each bullet ≤25 words."},
            {"Code Generation", "Write a robust Python script that reads NDJSON from stdin, filters records where rule_level ≥ 10 or mitreid is not null, outputs NDJSON, includes argparse, logging, and unit tests."},
            {"Long Report", "Produce a 900–1,200 word mini-report: 'History of Transformer Models (2017–2025)'. Use section headers, numbered references, and a concluding 'Key Takeaways' list of 10 items."},
            {"Translation Task", "Translate this to English and then explain it for a non-technical audience in 5 bullets: « Les compteurs intelligents doivent conserver au moins 13 mois de profils de charge et journaliser les événements de fraude. »"},
            {"JSON Output", "Return ONLY valid JSON (no extra text) summarizing pros/cons of CPU-only vs GPU inference for a 20B parameter model. Fields: {\"hardware\": [..], \"throughput_tpm\": \"string\", \"latency_ms\": \"string\", \"energy_tradeoffs\": [..], \"when_to_choose\": {\"cpu\": [..], \"gpu\": [..]}}."},
            {"Technical Design", "Write a detailed design doc (~800 words) for a PostgreSQL + TimescaleDB pipeline that ingests 4B log rows/day, with partitioning, continuous aggregates, and a watermarked incremental job. Include sample DDL and 3 optimized queries."},
            {"Tutorial Generation", "Generate a step-by-step tutorial that builds a retrieval-augmented generation (RAG) prototype with Ollama + FAISS. Include: environment setup, data ingestion, embedding choice rationale, retrieval API, evaluation checklist, and a final 'Gotchas' section."}
        };
        
        std::cout << "Running comprehensive benchmark suite...\n\n";
        
        for (const auto& test_case : test_cases) {
            BenchmarkResult result = RunSingleBenchmark(test_case.first, test_case.second);
            results.push_back(result);
            
            PrintBenchmarkResult(result);
            std::cout << "\n";
        }
        
        return results;
    }
    
    void GenerateReport(const std::vector<BenchmarkResult>& results) {
        std::ofstream report("benchmark_report.txt");
        
        if (!report.is_open()) {
            std::cerr << "Failed to open report file\n";
            return;
        }
        
        report << "High-Performance Inference Engine Benchmark Report\n";
        report << "=================================================\n\n";
        
        report << "Generated: " << GetCurrentTimestamp() << "\n\n";
        
        // Summary statistics
        size_t total_tokens = 0;
        std::chrono::microseconds total_time(0);
        size_t successful_tests = 0;
        
        for (const auto& result : results) {
            if (result.success) {
                total_tokens += result.tokens_generated;
                total_time += result.total_time;
                successful_tests++;
            }
        }
        
        double average_tps = total_tokens / (total_time.count() / 1000000.0);
        
        report << "SUMMARY\n";
        report << "-------\n";
        report << "Total test cases: " << results.size() << "\n";
        report << "Successful tests: " << successful_tests << "\n";
        report << "Total tokens generated: " << total_tokens << "\n";
        report << "Total processing time: " << (total_time.count() / 1000.0) << " ms\n";
        report << "Average TPS: " << std::fixed << std::setprecision(2) << average_tps << "\n\n";
        
        // Individual results
        report << "DETAILED RESULTS\n";
        report << "----------------\n\n";
        
        for (const auto& result : results) {
            report << "Test: " << result.prompt_name << "\n";
            report << "Prompt: " << result.prompt.substr(0, 100) << (result.prompt.length() > 100 ? "..." : "") << "\n";
            report << "Status: " << (result.success ? "SUCCESS" : "FAILED") << "\n";
            
            if (result.success) {
                report << "Tokens generated: " << result.tokens_generated << "\n";
                report << "Total time: " << (result.total_time.count() / 1000.0) << " ms\n";
                report << "Prefill time: " << (result.prefill_time.count() / 1000.0) << " ms\n";
                report << "Decode time: " << (result.decode_time.count() / 1000.0) << " ms\n";
                report << "TPS: " << std::fixed << std::setprecision(2) << result.tokens_per_second << "\n";
                report << "Memory used: " << (result.memory_used / 1024.0 / 1024.0) << " MB\n";
                report << "Cache hit rate: " << (result.cache_hit_rate * 100.0) << "%\n";
            } else {
                report << "Error: " << result.error_message << "\n";
            }
            report << "\n";
        }
        
        // Performance analysis
        report << "PERFORMANCE ANALYSIS\n";
        report << "--------------------\n";
        
        std::vector<double> tps_values;
        for (const auto& result : results) {
            if (result.success) {
                tps_values.push_back(result.tokens_per_second);
            }
        }
        
        if (!tps_values.empty()) {
            std::sort(tps_values.begin(), tps_values.end());
            
            report << "TPS Statistics:\n";
            report << "  Minimum: " << std::fixed << std::setprecision(2) << tps_values.front() << "\n";
            report << "  Maximum: " << std::fixed << std::setprecision(2) << tps_values.back() << "\n";
            report << "  Median: " << std::fixed << std::setprecision(2) << tps_values[tps_values.size() / 2] << "\n";
            
            double sum = 0.0;
            for (double tps : tps_values) {
                sum += tps;
            }
            double mean = sum / tps_values.size();
            report << "  Mean: " << std::fixed << std::setprecision(2) << mean << "\n";
            
            // Variance calculation
            double variance = 0.0;
            for (double tps : tps_values) {
                variance += (tps - mean) * (tps - mean);
            }
            variance /= tps_values.size();
            double std_dev = std::sqrt(variance);
            report << "  Standard Deviation: " << std::fixed << std::setprecision(2) << std_dev << "\n";
        }
        
        // Baseline comparison
        const double BASELINE_TPS = 30.0;
        report << "\nBASELINE COMPARISON\n";
        report << "-------------------\n";
        report << "Baseline TPS: " << BASELINE_TPS << "\n";
        report << "Achieved TPS: " << std::fixed << std::setprecision(2) << average_tps << "\n";
        
        if (average_tps > BASELINE_TPS) {
            double improvement = (average_tps / BASELINE_TPS - 1.0) * 100.0;
            report << "Improvement: +" << std::fixed << std::setprecision(1) << improvement << "%\n";
            report << "Status: EXCEEDS BASELINE REQUIREMENT ✓\n";
        } else {
            double deficit = (1.0 - average_tps / BASELINE_TPS) * 100.0;
            report << "Deficit: -" << std::fixed << std::setprecision(1) << deficit << "%\n";
            report << "Status: BELOW BASELINE REQUIREMENT ✗\n";
        }
        
        report.close();
        std::cout << "Benchmark report saved to benchmark_report.txt\n";
    }

private:
    InferenceEngine& engine_;
    
    BenchmarkResult RunSingleBenchmark(const std::string& name, const std::string& prompt) {
        BenchmarkResult result;
        result.prompt_name = name;
        result.prompt = prompt;
        result.success = false;
        
        try {
            Timer timer;
            auto tokens = engine_.Generate(prompt);
            auto elapsed = timer.Elapsed();
            
            result.tokens_generated = tokens.size();
            result.total_time = elapsed;
            result.success = true;
            
            // Get detailed statistics
            auto stats = engine_.GetLastStats();
            result.prefill_time = stats.prefill_time;
            result.decode_time = stats.decode_time;
            result.tokens_per_second = stats.tokens_per_second;
            result.memory_used = stats.memory_used;
            result.cache_hit_rate = stats.cache_hit_rate;
            
        } catch (const std::exception& e) {
            result.error_message = e.what();
        }
        
        return result;
    }
    
    void PrintBenchmarkResult(const BenchmarkResult& result) {
        std::cout << "[" << result.prompt_name << "]\n";
        
        if (result.success) {
            std::cout << "  Status: SUCCESS\n";
            std::cout << "  Tokens: " << result.tokens_generated << "\n";
            std::cout << "  Time: " << (result.total_time.count() / 1000.0) << " ms\n";
            std::cout << "  TPS: " << std::fixed << std::setprecision(2) << result.tokens_per_second << "\n";
            std::cout << "  Memory: " << (result.memory_used / 1024.0 / 1024.0) << " MB\n";
            std::cout << "  Cache hit rate: " << (result.cache_hit_rate * 100.0) << "%\n";
        } else {
            std::cout << "  Status: FAILED\n";
            std::cout << "  Error: " << result.error_message << "\n";
        }
    }
    
    std::string GetCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};

int main(int argc, char* argv[]) {
    Logger::Initialize("benchmark.log", LogLevel::INFO);
    
    std::string model_path;
    bool generate_report = true;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--no-report") {
            generate_report = false;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --model <path> [--no-report]\n";
            return 0;
        }
    }
    
    if (model_path.empty()) {
        std::cerr << "Error: Model path required. Use --model <path>\n";
        return 1;
    }
    
    try {
        // Configure engine for benchmarking
        InferenceConfig config;
        config.max_new_tokens = 512;
        config.enable_caching = true;
        config.enable_quantization = true;
        config.use_simd = true;
        config.enable_prefill_optimization = true;
        
        InferenceEngine engine(config);
        
        std::cout << "Loading model from: " << model_path << "\n";
        if (!engine.LoadModel(model_path)) {
            std::cerr << "Failed to load model\n";
            return 1;
        }
        
        std::cout << "Model loaded successfully!\n\n";
        
        // Run benchmark
        BenchmarkSuite benchmark(engine);
        auto results = benchmark.RunComprehensiveBenchmark();
        
        if (generate_report) {
            benchmark.GenerateReport(results);
        }
        
        // Print summary
        size_t total_tokens = 0;
        std::chrono::microseconds total_time(0);
        size_t successful = 0;
        
        for (const auto& result : results) {
            if (result.success) {
                total_tokens += result.tokens_generated;
                total_time += result.total_time;
                successful++;
            }
        }
        
        double average_tps = total_tokens / (total_time.count() / 1000000.0);
        
        std::cout << "\n=== BENCHMARK SUMMARY ===\n";
        std::cout << "Successful tests: " << successful << "/" << results.size() << "\n";
        std::cout << "Total tokens: " << total_tokens << "\n";
        std::cout << "Average TPS: " << std::fixed << std::setprecision(2) << average_tps << "\n";
        
        if (average_tps > 30.0) {
            std::cout << "✓ EXCEEDS BASELINE REQUIREMENT (30 TPS)\n";
        } else {
            std::cout << "✗ Below baseline requirement (30 TPS)\n";
        }
        
    } catch (const std::exception& e) {
        Logger::Error("Benchmark failed: %s", e.what());
        std::cerr << "Benchmark failed: " << e.what() << "\n";
        return 1;
    }
    
    Logger::Shutdown();
    return 0;
}
