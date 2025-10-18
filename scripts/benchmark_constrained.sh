#!/bin/bash

# Resource-constrained benchmark script
# Limits resources to 80% of system capacity

# Get system specs
TOTAL_CORES=$(nproc)
TOTAL_MEM_MB=$(free -m | awk '/^Mem:/{print $2}')

# Calculate 80% of resources
CONSTRAINED_CORES=$(echo "$TOTAL_CORES * 0.8" | bc | awk '{print int($1)}')
CONSTRAINED_MEM_MB=$(echo "$TOTAL_MEM_MB * 0.8" | bc | awk '{print int($1)}')

# Ensure at least 1 core
if [ "$CONSTRAINED_CORES" -lt 1 ]; then
    CONSTRAINED_CORES=1
fi

echo "=========================================="
echo "Resource-Constrained Benchmark"
echo "=========================================="
echo ""
echo "System Resources:"
echo "  Total Cores: $TOTAL_CORES"
echo "  Total Memory: $TOTAL_MEM_MB MB"
echo ""
echo "Constrained Resources (80%):"
echo "  Cores: $CONSTRAINED_CORES"
echo "  Memory: $CONSTRAINED_MEM_MB MB"
echo ""
echo "=========================================="
echo ""

# Create a lightweight test version of the benchmark
# We'll use taskset to limit CPU cores and ulimit for memory

cd /home/ml/Documents/projects/TPS

# Generate CPU affinity mask for first N cores
# For 3 cores: mask = 0-2
CORE_MASK="0-$((CONSTRAINED_CORES - 1))"

echo "Running benchmark with constraints..."
echo "CPU Affinity: $CORE_MASK"
echo "Memory Limit: ${CONSTRAINED_MEM_MB}M"
echo ""

# Run benchmark with resource limits
# Pass memory limit to benchmark application itself
echo "Starting benchmark..."
echo ""

# Use taskset to pin to specific cores and pass memory limit to app
taskset -c $CORE_MASK ./build/benchmark \
    --model mock_model \
    --threads $CONSTRAINED_CORES \
    --memory $CONSTRAINED_MEM_MB \
    --quick \
    --max-tokens 100

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Benchmark completed successfully"
else
    echo "Benchmark failed with exit code: $EXIT_CODE"
fi
echo "=========================================="


