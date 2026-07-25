#!/bin/bash
#
# 单算子性能基准测试 - 完整版本
# 支持 MNN, ONNX Runtime, TVM
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGLE_OPS_DIR="$PROJECT_ROOT/models/single_ops"
ADB="/mnt/e/andorid/adb/adb.exe"
DEVICE_DIR="/data/local/tmp/single_ops"

WARMUP=${WARMUP:-10}
RUNS=${RUNS:-100}
THREADS=${THREADS:-4}

echo "========================================"
echo "  Single Operator Benchmark Suite"
echo "========================================"
echo "Warmup: $WARMUP, Runs: $RUNS, Threads: $THREADS"
echo

# 算子输入形状映射
declare -A OP_SHAPES=(
    ["AdaptiveAvgPool"]="1,128,7,7"
    ["Add"]="1,128,28,28"
    ["AvgPool2d_2x2_s2"]="1,64,56,56"
    ["BatchNorm2d"]="1,128,28,28"
    ["Conv2d_1x1_s1"]="1,64,56,56"
    ["Conv2d_3x3_s1"]="1,64,56,56"
    ["Conv2d_3x3_s2"]="1,64,56,56"
    ["ConvTranspose_2x2"]="1,64,28,28"
    ["Conv_BN_ReLU"]="1,64,56,56"
    ["DWConv_PWConv"]="1,64,56,56"
    ["DepthwiseConv_3x3_s1"]="1,64,56,56"
    ["Div"]="1,128,28,28"
    ["ElementWise_Pow"]="1,128,28,28"
    ["Exp"]="1,128,28,28"
    ["GELU"]="1,128,28,28"
    ["Hardswish"]="1,128,28,28"
    ["InstanceNorm2d"]="1,128,28,28"
    ["LayerNorm"]="1,512"
    ["LeakyReLU"]="1,128,28,28"
    ["Linear_1024_1024"]="1,1024"
    ["Linear_512_512"]="1,512"
    ["MaxPool2d_2x2_s2"]="1,64,56,56"
    ["Mul"]="1,128,28,28"
    ["PReLU"]="1,128,28,28"
    ["ReLU"]="1,128,28,28"
    ["ReLU6"]="1,128,28,28"
    ["ReduceMean_Spatial"]="1,128,28,28"
    ["ReduceSum_Spatial"]="1,128,28,28"
    ["ReflectionPad2d"]="1,64,28,28"
    ["Reshape_Flatten"]="1,128,7,7"
    ["Sigmoid"]="1,128,28,28"
    ["Softmax"]="1,1000"
    ["Sqrt"]="1,128,28,28"
    ["Tanh"]="1,128,28,28"
    ["Upsample_Bilinear"]="1,64,28,28"
    ["Upsample_Nearest"]="1,64,28,28"
    ["ZeroPad2d"]="1,64,28,28"
)

# 准备结果文件
mkdir -p "$PROJECT_ROOT/results"
RESULTS_FILE="$PROJECT_ROOT/results/single_ops_results_$(date +%Y%m%d_%H%M%S).csv"
echo "op,backend,mean_ms,min_ms,max_ms,std_ms" > "$RESULTS_FILE"

# 推送文件到设备
echo "Pushing binary and libraries..."
$ADB push "$PROJECT_ROOT/build_android/src/single_op/single_op_benchmark" /data/local/tmp/
$ADB push "$PROJECT_ROOT/build_android/third_party/MNN/libMNN.so" /data/local/tmp/
$ADB push "$PROJECT_ROOT/build_android/third_party/onnxruntime/libonnxruntime.so" /data/local/tmp/ 2>/dev/null || true

$ADB shell "chmod +x /data/local/tmp/single_op_benchmark"


echo "Pushing MNN models..."
$ADB push "$SINGLE_OPS_DIR/mnn" "$DEVICE_DIR/" 2>/dev/null

echo "Pushing ONNX models..."
$ADB push "$SINGLE_OPS_DIR"/*.onnx "$DEVICE_DIR/" 2>/dev/null

echo
echo "Starting benchmark..."
echo "========================================"

OPS=$(cd "$SINGLE_OPS_DIR" && ls *.onnx 2>/dev/null | sed 's/\.onnx$//' | sort)
TOTAL=$(echo "$OPS" | wc -l)
CURRENT=0

for op in $OPS; do
    CURRENT=$((CURRENT + 1))
    echo
    echo "[$CURRENT/$TOTAL] $op"
    echo "----------------------------------------"

    SHAPE="${OP_SHAPES[$op]}"
    if [ -z "$SHAPE" ]; then
        SHAPE="1,3,224,224"
    fi

    fi

    # Test MNN
    echo -n "  mnn:     "
    result=$($ADB shell "cd /data/local/tmp && export LD_LIBRARY_PATH=/data/local/tmp && ./single_op_benchmark --backend mnn --model $DEVICE_DIR/mnn/$op.mnn --input_shape $SHAPE --warmup $WARMUP --runs $RUNS --threads $THREADS 2>/dev/null" | grep "mean=")
    if [[ "$result" == *"mean="* ]]; then
        mean=$(echo "$result" | grep -oP 'mean=\K[0-9.]+')
        min=$(echo "$result" | grep -oP 'min=\K[0-9.]+')
        max=$(echo "$result" | grep -oP 'max=\K[0-9.]+')
        std=$(echo "$result" | grep -oP 'std=\K[0-9.]+')
        echo "mean=${mean}ms, min=${min}ms, max=${max}ms, std=${std}ms"
        echo "$op,mnn,$mean,$min,$max,$std" >> "$RESULTS_FILE"
    else
        echo "FAILED"
        echo "$op,mnn,0,0,0,0" >> "$RESULTS_FILE"
    fi

    # Test ONNX Runtime
    echo -n "  onnxrt:  "
    result=$($ADB shell "cd /data/local/tmp && export LD_LIBRARY_PATH=/data/local/tmp && ./single_op_benchmark --backend onnxrt --model $DEVICE_DIR/$op.onnx --input_shape $SHAPE --warmup $WARMUP --runs $RUNS --threads $THREADS 2>/dev/null" | grep "mean=")
    if [[ "$result" == *"mean="* ]]; then
        mean=$(echo "$result" | grep -oP 'mean=\K[0-9.]+')
        min=$(echo "$result" | grep -oP 'min=\K[0-9.]+')
        max=$(echo "$result" | grep -oP 'max=\K[0-9.]+')
        std=$(echo "$result" | grep -oP 'std=\K[0-9.]+')
        echo "mean=${mean}ms, min=${min}ms, max=${max}ms, std=${std}ms"
        echo "$op,onnxrt,$mean,$min,$max,$std" >> "$RESULTS_FILE"
    else
        echo "FAILED"
        echo "$op,onnxrt,0,0,0,0" >> "$RESULTS_FILE"
    fi
done

echo
echo "========================================"
echo "Benchmark complete!"
echo "Results: $RESULTS_FILE"
echo "========================================"

# Generate analysis
python3 "$PROJECT_ROOT/scripts/analyze_single_op_results.py" "$RESULTS_FILE"
