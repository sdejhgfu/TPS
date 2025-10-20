#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_BUILD_TYPE=Release "$ROOT_DIR"
cmake --build . -j$(nproc)

echo "Running unit tests..."
"$BUILD_DIR/tps_tests"

echo "Running TPS demo..."
"$BUILD_DIR/tps"





