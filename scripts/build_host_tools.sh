#!/bin/bash
# Build host tools for model conversion
# This builds onnx2ncnn, onnx2mnn, onnx2tnn for x86_64 host

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

# 1. Build ncnn tools (onnx2ncnn)
echo "=== Building ncnn onnx2ncnn ==="
cd $BUILD_DIR
mkdir -p ncnn && cd ncnn
cmake $PROJECT_ROOT/third_party/ncnn \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_BUILD_TOOLS=ON \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_OPENMP=ON \
    -DNCNN_SIMPLEOCV=OFF

make -j$(nproc)
echo "onnx2ncnn built: $BUILD_DIR/ncnn/tools/onnx/onnx2ncnn"
echo ""

# 2. Build MNN tools (onnx2mnn)
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

# 3. Build TNN tools (onnx2tnn)
echo "=== Building TNN onnx2tnn ==="
cd $BUILD_DIR
mkdir -p TNN && cd TNN
cmake $PROJECT_ROOT/third_party/TNN \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_BUILD_CONVERTER=ON \
    -DTNN_BUILD_SHARED=OFF \
    -DTNN_BUILD_BENCHMARK=OFF \
    -DTNN_BUILD_EXAMPLES=OFF

make -j$(nproc)
echo "onnx2tnn built: $BUILD_DIR/TNN/tools/onnx2tnn/onnx2tnn"
echo ""

# 4. Create symlinks for easy access
cd $PROJECT_ROOT
mkdir -p tools/bin

ln -sf $BUILD_DIR/ncnn/tools/onnx/onnx2ncnn tools/bin/onnx2ncnn
ln -sf $BUILD_DIR/MNN/MNNConvert tools/bin/MNNConvert
ln -sf $BUILD_DIR/TNN/tools/onnx2tnn/onnx2tnn tools/bin/onnx2tnn

echo "=== Build Complete ==="
echo ""
echo "Tools available in $PROJECT_ROOT/tools/bin:"
ls -la $PROJECT_ROOT/tools/bin/
echo ""
echo "Available converters:"
echo "  - onnx2ncnn: ONNX to ncnn format"
echo "  - MNNConvert: ONNX/TF/TFLite to MNN format"
echo "  - onnx2tnn: ONNX to TNN format (when built)"
echo ""
echo "For TFLite conversion, install tensorflow and onnx-tf:"
echo "  pip install tensorflow onnx onnx-tf"
echo ""
echo "Then run: ./scripts/convert_models.sh"
