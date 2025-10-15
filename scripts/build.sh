#!/bin/bash

# High-Performance Inference Engine Build Script

set -e

echo "Building High-Performance Inference Engine..."
echo "============================================="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found. Please run this script from the project root."
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto" \
         -DENABLE_SIMD=ON \
         -DENABLE_MULTITHREADING=ON

# Build the project
echo "Building..."
make -j$(nproc)

# Run tests
echo "Running tests..."
if [ -f "test_runner" ]; then
    ./test_runner
    echo "✓ Tests passed!"
else
    echo "Warning: Test runner not found"
fi

# Create release package
echo "Creating release package..."
cd ..
tar -czf inference_engine_release.tar.gz \
    build/inference_engine \
    build/benchmark \
    README.md \
    docs/

echo "Build completed successfully!"
echo "Executables:"
echo "  - build/inference_engine"
echo "  - build/benchmark"
echo "Package: inference_engine_release.tar.gz"
