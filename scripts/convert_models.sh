#!/bin/bash
# Convert ONNX models to each framework format
# This script must be run after download_pretrained.py
# Run ./scripts/build_host_tools.sh first to build conversion tools

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
MODELS_ROOT=$PROJECT_ROOT/models

# Paths to conversion tools (use host tools)
NCNN_ONNX2NCNN=$PROJECT_ROOT/tools/bin/onnx2ncnn
MNN_ONNX2MNN=$PROJECT_ROOT/tools/bin/MNNConvert
# TNN converter is pure Python in third_party (no compilation needed)
TNN_CONVERTER=$PROJECT_ROOT/third_party/TNN/tools/convert2tnn/converter.py

echo "=== Model Conversion Script ==="
echo "Project root: $PROJECT_ROOT"
echo "Models root: $MODELS_ROOT"
echo ""

# Check tools availability
echo "Checking conversion tools..."
if [ ! -f "$NCNN_ONNX2NCNN" ]; then
    echo "WARNING: onnx2ncnn not found"
    echo "  Run: ./scripts/build_host_tools.sh"
fi
if [ ! -f "$MNN_ONNX2MNN" ]; then
    echo "WARNING: onnx2mnn not found"
    echo "  Run: ./scripts/build_host_tools.sh"
fi
if [ ! -f "$TNN_ONNX2TNN" ]; then
    echo "WARNING: onnx2tnn not found"
    echo "  Run: ./scripts/build_host_tools.sh"
fi
echo ""

# Check for tensorflow (TFLite converter)
python3 -c "import tensorflow as tf" 2>/dev/null || {
    echo "WARNING: tensorflow not installed (for TFLite conversion)"
    echo "  Run: pip install tensorflow"
    echo ""
}

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
        $MNN_ONNX2MNN -f ONNX --modelFile $ONNX --MNNModel $OUT_DIR/mobilenetv2_MNN.mnn
        echo "MNN output: $OUT_DIR/mobilenetv2_MNN.mnn"
    fi

    # TNN
    if [ -f "$TNN_CONVERTER" ]; then
        echo "Converting to TNN..."
        MODEL_NAME=$(basename $ONNX .onnx)
        # Note: TNN converter requires isolated conda environment
        # Use: conda activate tnn_converter and ensure onnx2tnn C++ module built
        if python3 -c "import onnx2tnn" 2>/dev/null; then
            python3 $TNN_CONVERTER onnx2tnn $ONNX \
                -in input:1,3,224,224 \
                -optimize \
                -v v1.0 \
                -o $OUT_DIR 2>&1 | tail -3

            # Rename output files
            if [ -f "$OUT_DIR/input.tnnproto" ]; then
                mv "$OUT_DIR/input.tnnproto" "$OUT_DIR/${MODEL_NAME}_TNN.tnnproto"
                mv "$OUT_DIR/input.tnnmodel" "$OUT_DIR/${MODEL_NAME}_TNN.tnnmodel"
            fi
            echo "TNN output: $OUT_DIR/${MODEL_NAME}_TNN.tnnproto + .tnnmodel"
        else
            echo "⚠️  TNN converter not available (requires conda env + compiled C++ module)"
            echo "    Skipping TNN conversion (model file may already exist)"
        fi
    fi

    # TFLite (via tensorflow + onnx-tf)
    if python3 -c "import onnx; import tensorflow as tf; from onnx_tf.backend import prepare" 2>/dev/null; then
        echo "Converting to TFLite..."
        python3 $PROJECT_ROOT/scripts/convert_tflite.py $ONNX $OUT_DIR/mobilenetv2.tflite
    fi

    # ONNX Runtime - use original ONNX file directly
    echo "ONNX Runtime: using original ONNX file"

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
        $MNN_ONNX2MNN -f ONNX --modelFile $ONNX --MNNModel $OUT_DIR/resnet50_MNN.mnn
        echo "MNN output: $OUT_DIR/resnet50_MNN.mnn"
    fi

    # TNN
    if [ -f "$TNN_ONNX2TNN" ]; then
        echo "Converting to TNN..."
        cd $OUT_DIR
        $TNN_ONNX2TNN -onnx $ONNX -version v1.0
        mv model.tnnproto resnet50_TNN.tnnproto
        mv model.tnnmodel resnet50_TNN.tnnmodel
        cd - > /dev/null
        echo "TNN output: $OUT_DIR/resnet50_TNN.tnnproto + .tnnmodel"
    fi

    # TFLite
    if python3 -c "import onnx; import tensorflow as tf; from onnx_tf.backend import prepare" 2>/dev/null; then
        echo "Converting to TFLite..."
        python3 $PROJECT_ROOT/scripts/convert_tflite.py $ONNX $OUT_DIR/resnet50.tflite
    fi

    # ONNX Runtime - use original ONNX file directly
    echo "ONNX Runtime: using original ONNX file"

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
        $MNN_ONNX2MNN -f ONNX --modelFile $ONNX --MNNModel $OUT_DIR/yolov8n_MNN.mnn
        echo "MNN output: $OUT_DIR/yolov8n_MNN.mnn"
    fi

    # TNN
    if [ -f "$TNN_ONNX2TNN" ]; then
        echo "Converting to TNN..."
        cd $OUT_DIR
        $TNN_ONNX2TNN -onnx $ONNX -version v1.0
        mv model.tnnproto yolov8n_TNN.tnnproto
        mv model.tnnmodel yolov8n_TNN.tnnmodel
        cd - > /dev/null
        echo "TNN output: $OUT_DIR/yolov8n_TNN.tnnproto + .tnnmodel"
    fi

    # TFLite
    if python3 -c "import onnx; import tensorflow as tf; from onnx_tf.backend import prepare" 2>/dev/null; then
        echo "Converting to TFLite..."
        python3 $PROJECT_ROOT/scripts/convert_tflite.py $ONNX $OUT_DIR/yolov8n.tflite
    fi

    # ONNX Runtime - use original ONNX file directly
    echo "ONNX Runtime: using original ONNX file"

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
        $MNN_ONNX2MNN -f ONNX --modelFile $ONNX --MNNModel $OUT_DIR/bert_MNN.mnn
        echo "MNN output: $OUT_DIR/bert_MNN.mnn"
    fi

    # TNN
    if [ -f "$TNN_ONNX2TNN" ]; then
        echo "Converting to TNN..."
        cd $OUT_DIR
        $TNN_ONNX2TNN -onnx $ONNX -version v1.0
        mv model.tnnproto bert_TNN.tnnproto
        mv model.tnnmodel bert_TNN.tnnmodel
        cd - > /dev/null
        echo "TNN output: $OUT_DIR/bert_TNN.tnnproto + .tnnmodel"
    fi

    # TFLite
    if python3 -c "import onnx; import tensorflow as tf; from onnx_tf.backend import prepare" 2>/dev/null; then
        echo "Converting to TFLite..."
        python3 $PROJECT_ROOT/scripts/convert_tflite.py $ONNX $OUT_DIR/bert.tflite
    fi

    # ONNX Runtime - use original ONNX file directly
    echo "ONNX Runtime: using original ONNX file"

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

echo ""
echo "=== Conversion Complete ==="
echo ""
echo "Summary of default-enabled frameworks:"
echo "  ncnn:  .model + .bin  (via onnx2ncnn)"
echo "  MNN:   .mnn           (via onnx2mnn)"
echo "  TNN:   .tnnproto + .tnnmodel  (via onnx2tnn)"
echo "  TFLite: .tflite       (via tensorflow + onnx-tf)"
echo "  ONNX Runtime: .onnx   (use original ONNX file)"
echo ""
echo "For optional frameworks:"
echo "  QNN: Use qnn-onnx-converter from Qualcomm QNN SDK"
echo "  TVM: Compile model ahead-of-time via TVM Relay"
