# High-Performance Inference Engine

## Project Overview

This project implements a high-performance inference engine optimized for commodity hardware (Intel/AMD CPUs with DDR4/5/6 memory) to achieve >30 TPS on GPT-OSS-20B model weights.

## Performance Target

- **Baseline**: 30 TPS (Ollama benchmark)
- **Target**: >30 TPS on 2-socket, 8-core (16 total) 32GB DDR5 VM
- **Model**: GPT-OSS-20B standard weights

## Key Optimizations

1. **Memory Management**: Custom memory allocator with paging
2. **Quantization**: INT8/INT4 weight quantization strategies
3. **Caching**: Multi-level caching for prefill/encoding/decoding
4. **CPU Optimization**: SIMD instructions and thread pooling
5. **Zero Dependencies**: First-party implementation only

## Project Structure

```
├── src/                    # Core implementation
│   ├── engine/            # Inference engine core
│   ├── memory/            # Memory management
│   ├── quantization/      # Model quantization
│   ├── cache/             # Caching mechanisms
│   └── utils/             # Utilities and helpers
├── tests/                 # Test suites
├── benchmarks/            # Performance benchmarks
├── docs/                  # Documentation
└── scripts/               # Build and deployment scripts
```

## Getting Started

### Prerequisites

- Linux OS (tested on Ubuntu 20.04+)
- GCC 9.0+ with C++17 support
- CMake 3.16+
- 32GB+ RAM (recommended for optimal performance)
- Multi-core CPU with AVX2/AVX-512 support (recommended)

### Build Instructions

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Running the Inference Engine

### Basic Usage

```bash
cd build

# Run inference with a simple prompt
./inference_engine --model mock_model --prompt "Hello, world!"

# Run with custom parameters
./inference_engine \
    --model mock_model \
    --prompt "Write a story about AI" \
    --max-tokens 256 \
    --threads 16 \
    --seed 42
```

### Available Options

- `--model <path>` - Path to model weights (use "mock_model" for testing)
- `--prompt <text>` - Input prompt for generation
- `--max-tokens <n>` - Maximum number of tokens to generate (default: 128)
- `--threads <n>` - Number of CPU threads to use (default: auto-detect)
- `--memory <mb>` - Maximum memory in MB (default: auto-detect)
- `--seed <n>` - Random seed for reproducible results

## Testing Performance

### Quick Performance Test

```bash
cd build
./benchmark --model mock_model
```

### Full Performance Benchmark

```bash
# From project root
bash scripts/benchmark_full.sh
```

### Comprehensive Validation Suite

```bash
# From project root
bash scripts/validate_all.sh
```

### Check Results

```bash
cat build/benchmark_report.txt
```

## Running Tests

### Unit Tests

```bash
cd build
./test_runner
```

### Memory Leak Check

```bash
cd build
valgrind --leak-check=full ./test_runner
```

## Quick Start Guide

```bash
# Build the project
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# Run unit tests
./test_runner

# Run a quick benchmark
./benchmark --model mock_model

# Run comprehensive validation (from project root)
cd ..
bash scripts/validate_all.sh

# Check results
cat build/benchmark_report.txt
```

## Performance Metrics

- **TPS**: Tokens per second (wall-clock time)
- **Memory Usage**: Peak and average memory consumption
- **Cache Hit Rate**: Efficiency of caching mechanisms
- **Latency**: End-to-end inference latency

## Testing

The engine is tested against 10 standardized prompts covering various complexity levels and use cases.

## License

Proprietary - Competition Submission
