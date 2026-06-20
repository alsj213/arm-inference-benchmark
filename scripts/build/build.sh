#!/bin/bash
# Build script for benchmark-arm-inference

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

echo "Building benchmark-arm-inference..."
echo "Project root: $PROJECT_ROOT"

# Create build directory
mkdir -p $PROJECT_ROOT/build
cd $PROJECT_ROOT/build

# Configure cmake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBENCHMARK_NCNN=OFF \
    -DBENCHMARK_MNN=ON \
    -DBENCHMARK_TNN=OFF \
    -DBENCHMARK_TFLITE=OFF \
    -DBENCHMARK_QNN=OFF \
    -DBENCHMARK_ORT=ON \
    -DBENCHMARK_TVM=OFF

# Build
make -j$(nproc)

echo "Build completed. Output: $PROJECT_ROOT/build/benchmark_inference"
