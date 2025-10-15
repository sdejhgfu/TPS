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

## Build Instructions

```bash
# Build the inference engine
make build

# Run benchmarks
make benchmark

# Run tests
make test
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
