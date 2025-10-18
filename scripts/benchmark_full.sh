#!/bin/bash

# Full performance benchmark script
# Uses 100% of system resources

# Get system specs
TOTAL_CORES=$(nproc)
TOTAL_MEM_MB=$(free -m | awk '/^Mem:/{print $2}')

echo "=========================================="
echo "Full Performance Benchmark"
echo "=========================================="
echo ""
echo "System Resources:"
echo "  Total Cores: $TOTAL_CORES"
echo "  Total Memory: $TOTAL_MEM_MB MB"
echo ""
echo "Using 100% of available resources"
echo ""
echo "=========================================="
echo ""

cd /home/ml/Documents/projects/TPS

echo "Running comprehensive benchmark..."
echo "This will generate up to 256 tokens per test"
echo ""

# Run benchmark with full resources
./build/benchmark \
    --model mock_model \
    --threads $TOTAL_CORES \
    --memory $TOTAL_MEM_MB \
    --max-tokens 256

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Benchmark completed successfully"
    echo ""
    echo "Check benchmark_report.txt for detailed results"
else
    echo "Benchmark failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

