#!/bin/bash
# Convert ONNX models to each framework format
# 当前仅支持 MNN 模型转换
# 如需恢复已停用的后端转换，见 backup/all-backends 分支
#
# This script must be run after download_pretrained.py
# Run ./scripts/build_host_tools.sh first to build conversion tools

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
MODELS_ROOT=$PROJECT_ROOT/models

# Paths to conversion tools (use host tools)
MNN_ONNX2MNN=$PROJECT_ROOT/tools/bin/MNNConvert

echo "=== Model Conversion Script ==="
echo "Project root: $PROJECT_ROOT"
echo "Models root: $MODELS_ROOT"
echo ""

# Check tools availability
echo "Checking conversion tools..."
if [ ! -f "$MNN_ONNX2MNN" ]; then
    echo "WARNING: onnx2mnn not found"
    echo "  Run: ./scripts/build_host_tools.sh"
fi
echo ""

    echo ""
}

# Function to convert mobilenetv2
convert_mobilenetv2() {
    echo "=== Converting MobileNetV2 ==="
    local ONNX=$MODELS_ROOT/classification/mobilenetv2/mobilenetv2.onnx
    local OUT_DIR=$MODELS_ROOT/classification/mobilenetv2

    mkdir -p $OUT_DIR

    fi

    # MNN
    if [ -f "$MNN_ONNX2MNN" ]; then
        echo "Converting to MNN..."
        $MNN_ONNX2MNN -f ONNX --modelFile $ONNX --MNNModel $OUT_DIR/mobilenetv2_MNN.mnn
        echo "MNN output: $OUT_DIR/mobilenetv2_MNN.mnn"
    fi

        MODEL_NAME=$(basename $ONNX .onnx)
                -in input:1,3,224,224 \
                -optimize \
                -v v1.0 \
                -o $OUT_DIR 2>&1 | tail -3

            # Rename output files
            fi
        else
        fi
    fi

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

    fi

    # MNN
    if [ -f "$MNN_ONNX2MNN" ]; then
        echo "Converting to MNN..."
        $MNN_ONNX2MNN -f ONNX --modelFile $ONNX --MNNModel $OUT_DIR/resnet50_MNN.mnn
        echo "MNN output: $OUT_DIR/resnet50_MNN.mnn"
    fi

        cd $OUT_DIR
        cd - > /dev/null
    fi

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

    fi

    # MNN
    if [ -f "$MNN_ONNX2MNN" ]; then
        echo "Converting to MNN..."
        $MNN_ONNX2MNN -f ONNX --modelFile $ONNX --MNNModel $OUT_DIR/yolov8n_MNN.mnn
        echo "MNN output: $OUT_DIR/yolov8n_MNN.mnn"
    fi

        cd $OUT_DIR
        cd - > /dev/null
    fi

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

    fi

    # MNN
    if [ -f "$MNN_ONNX2MNN" ]; then
        echo "Converting to MNN..."
        $MNN_ONNX2MNN -f ONNX --modelFile $ONNX --MNNModel $OUT_DIR/bert_MNN.mnn
        echo "MNN output: $OUT_DIR/bert_MNN.mnn"
    fi

        cd $OUT_DIR
        cd - > /dev/null
    fi

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
echo "  MNN:   .mnn           (via onnx2mnn)"
echo "  ONNX Runtime: .onnx   (use original ONNX file)"
echo ""
echo "For optional frameworks:"
echo "  TVM: Compile model ahead-of-time via TVM Relay"
