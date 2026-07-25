#!/bin/bash
# Build host tools for model conversion
# This builds onnx2mnn for x86_64 host

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
BUILD_DIR=$PROJECT_ROOT/build_host_tools

echo "=== Building Host Model Conversion Tools ==="
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"
echo ""

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# echo ""

# 2. Build MNN tools (MNNConvert/onnx2mnn)
echo "=== Building MNN onnx2mnn ==="
cd $BUILD_DIR
mkdir -p MNN && cd MNN
cmake $PROJECT_ROOT/third_party/MNN \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_TOOLS=ON \
    -DMNN_BUILD_SHARED_LIBS=OFF \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_BUILD_BENCHMARK=OFF

make -j$(nproc)
echo "onnx2mnn built: $BUILD_DIR/MNN/onnx2mnn"
echo ""

# echo ""

# 4. Create symlinks for easy access
cd $PROJECT_ROOT
mkdir -p tools/bin

ln -sf $BUILD_DIR/MNN/MNNConvert tools/bin/MNNConvert

echo "=== Build Complete ==="
echo ""
echo "Tools available in $PROJECT_ROOT/tools/bin:"
ls -la $PROJECT_ROOT/tools/bin/
echo ""
echo "Available converters:"
echo "  - MNNConvert: ONNX to MNN format"
echo ""
echo ""
echo "Then run: ./scripts/convert_models.sh"
