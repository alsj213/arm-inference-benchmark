#!/bin/bash
# Convert ONNX models to each framework format
# This script must be run after download_pretrained.py

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
MODELS_ROOT=$PROJECT_ROOT/models

# Paths to conversion tools
NCNN_ONNX2NCNN=$PROJECT_ROOT/build/third_party/ncnn/tools/onnx/onnx2ncnn
MNN_ONNX2MNN=$PROJECT_ROOT/third_party/MNN/build/onnx2mnn

echo "=== Model Conversion Script ==="
echo "Project root: $PROJECT_ROOT"
echo "Models root: $MODELS_ROOT"
echo ""

# Check if ncnn onnx2ncnn is built
if [ ! -f "$NCNN_ONNX2NCNN" ]; then
    echo "WARNING: onnx2ncnn not found at $NCNN_ONNX2NCNN"
    echo "Please build the project first to build ncnn tools"
    echo ""
fi

# Check if MNN onnx2mnn is built
if [ ! -f "$MNN_ONNX2MNN" ]; then
    echo "WARNING: onnx2mnn not found at $MNN_ONNX2MNN"
    echo "Please build MNN tools first"
    echo ""
fi

# Function to convert mobilenetv2
convert_mobilenetv2() {
    echo "=== Converting MobileNetV2 ==="
    local ONNX=$MODELS_ROOT/classification/mobilenetv2/mobilenetv2.onnx
    local OUT_DIR=$MODELS_ROOT/classification/mobilenetv2

    mkdir -p $OUT_DIR

    # ncnn
    if [ -f "$NCNN_ONNX2NCNN" ]; then
        echo "Converting to ncnn..."
        $NCNN_ONNX2NCNN $ONNX $OUT_DIR/mobilenetv2_ncnn.model $OUT_DIR/mobilenetv2_ncnn.bin
        echo "ncnn output: $OUT_DIR/mobilenetv2_ncnn.model + .bin"
    fi

    # MNN
    if [ -f "$MNN_ONNX2MNN" ]; then
        echo "Converting to MNN..."
        $MNN_ONNX2MNN $ONNX $OUT_DIR/mobilenetv2_MNN.mnn
        echo "MNN output: $OUT_DIR/mobilenetv2_MNN.mnn"
    fi

    # TVM - export via relay, this needs to be done offline with TVM
    echo "For TVM: Please compile model separately using TVM Relay"
    echo ""
}

# Function to convert resnet50
convert_resnet50() {
    echo "=== Converting ResNet50 ==="
    local ONNX=$MODELS_ROOT/classification/resnet50/resnet50.onnx
    local OUT_DIR=$MODELS_ROOT/classification/resnet50

    mkdir -p $OUT_DIR

    # ncnn
    if [ -f "$NCNN_ONNX2NCNN" ]; then
        echo "Converting to ncnn..."
        $NCNN_ONNX2NCNN $ONNX $OUT_DIR/resnet50_ncnn.model $OUT_DIR/resnet50_ncnn.bin
        echo "ncnn output: $OUT_DIR/resnet50_ncnn.model + .bin"
    fi

    # MNN
    if [ -f "$MNN_ONNX2MNN" ]; then
        echo "Converting to MNN..."
        $MNN_ONNX2MNN $ONNX $OUT_DIR/resnet50_MNN.mnn
        echo "MNN output: $OUT_DIR/resnet50_MNN.mnn"
    fi

    echo ""
}

# Function to convert yolov8n
convert_yolov8n() {
    echo "=== Converting YOLOv8n ==="
    local ONNX=$MODELS_ROOT/detection/yolov8n/yolov8n.onnx
    local OUT_DIR=$MODELS_ROOT/detection/yolov8n

    mkdir -p $OUT_DIR

    # ncnn
    if [ -f "$NCNN_ONNX2NCNN" ]; then
        echo "Converting to ncnn..."
        $NCNN_ONNX2NCNN $ONNX $OUT_DIR/yolov8n_ncnn.model $OUT_DIR/yolov8n_ncnn.bin
        echo "ncnn output: $OUT_DIR/yolov8n_ncnn.model + .bin"
    fi

    # MNN
    if [ -f "$MNN_ONNX2MNN" ]; then
        echo "Converting to MNN..."
        $MNN_ONNX2MNN $ONNX $OUT_DIR/yolov8n_MNN.mnn
        echo "MNN output: $OUT_DIR/yolov8n_MNN.mnn"
    fi

    echo ""
}

# Function to convert bert
convert_bert() {
    echo "=== Converting BERT ==="
    local ONNX=$MODELS_ROOT/nlp/bert/bert.onnx
    local OUT_DIR=$MODELS_ROOT/nlp/bert

    mkdir -p $OUT_DIR

    # ncnn
    if [ -f "$NCNN_ONNX2NCNN" ]; then
        echo "Converting to ncnn..."
        $NCNN_ONNX2NCNN $ONNX $OUT_DIR/bert_ncnn.model $OUT_DIR/bert_ncnn.bin
        echo "ncnn output: $OUT_DIR/bert_ncnn.model + .bin"
    fi

    # MNN
    if [ -f "$MNN_ONNX2MNN" ]; then
        echo "Converting to MNN..."
        $MNN_ONNX2MNN $ONNX $OUT_DIR/bert_MNN.mnn
        echo "MNN output: $OUT_DIR/bert_MNN.mnn"
    fi

    echo ""
}

# Main conversion
echo "Checking for ONNX files..."
echo ""

if [ -f "$MODELS_ROOT/classification/mobilenetv2/mobilenetv2.onnx" ]; then
    convert_mobilenetv2
else
    echo "mobilenetv2.onnx not found, run download_pretrained.py first"
fi

if [ -f "$MODELS_ROOT/classification/resnet50/resnet50.onnx" ]; then
    convert_resnet50
else
    echo "resnet50.onnx not found, run download_pretrained.py first"
fi

if [ -f "$MODELS_ROOT/detection/yolov8n/yolov8n.onnx" ]; then
    convert_yolov8n
else
    echo "yolov8n.onnx not found, run download_pretrained.py first"
fi

if [ -f "$MODELS_ROOT/nlp/bert/bert.onnx" ]; then
    convert_bert
else
    echo "bert.onnx not found, run download_pretrained.py first"
fi

# Add TVM to run_benchmark.sh
echo "Updating run_benchmark.sh to include TVM..."
sed -i 's/BACKENDS=("ncnn" "mnn" "tnn" "tflite" "qnn")/BACKENDS=("ncnn" "mnn" "tnn" "tflite" "qnn" "tvm")/' $PROJECT_ROOT/scripts/run_benchmark.sh

echo ""
echo "=== Conversion Complete ==="
echo ""
echo "Notes:"
echo "  1. For TNN: Use onnx2tnn from TNN project"
echo "  2. For TFLite: Use TensorFlow lite converter"
echo "  3. For QNN: Use qnn-onnx-converter from Qualcomm QNN SDK"
echo "  4. For ONNX Runtime: Use original ONNX file directly"
echo "  5. For TVM: Compile model ahead-of-time via TVM Relay"
