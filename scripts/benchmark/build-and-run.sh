#!/bin/bash
# Build and run benchmark with selection of backends
# Usage:

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

# Default options
BACKEND="all"
MODEL="all"
PRECISION="fp32"
THREADS="1"
WARMUP="10"
RUNS="100"
GPU="no"

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --gpu)
            GPU="yes"
            shift 1
            ;;
        --help)
            echo "Usage: ./scripts/build_and_run.sh [options]"
            echo "Options:"
            echo "  --backend <mnn|onnxrt|ort|all>   Backend to test (default: all)"
            echo "  --model <mobilenetv2|resnet50|yolov8n|bert|all>    Model to test (default: all)"
            echo "  --precision <fp32|fp16|int8>                    Precision (default: fp32)"
            echo "  --threads <num>                                Number of threads (default: 1)"
            echo "  --warmup <num>                                 Number of warmup runs (default: 10)"
            echo "  --runs <num>                                   Number of test runs (default: 100)"
            echo "  --gpu                                          Use GPU if available"
            echo "  --help                                         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert backend to cmake options
# 已停用的后端保留在此，重新启用后端后设为 ON 即可
cmake_options=()
if [ "$BACKEND" != "all" ]; then
    # Disable all except selected
    cmake_options+=("-DBENCHMARK_MNN=OFF")
    cmake_options+=("-DBENCHMARK_ORT=OFF")

    # Enable selected
    case "$BACKEND" in
        "mnn")
            cmake_options+=("-DBENCHMARK_MNN=ON")
            ;;
        "onnxrt"|"ort")
            cmake_options+=("-DBENCHMARK_ORT=ON")
            ;;
    esac
fi

echo "=== Building benchmark for Android arm64-v8a ==="
echo "Backend:  $BACKEND"
echo "Model:    $MODEL"
echo "Precision:$PRECISION"
echo "Threads:  $THREADS"
echo

# Build
mkdir -p $PROJECT_ROOT/build_android
cd $PROJECT_ROOT/build_android

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DCMAKE_BUILD_TYPE=Release \
    "${cmake_options[@]}"

make -j$(nproc)

echo
echo "=== Pushing and running on device ==="
echo

# Run via adb (read path from config file with fallback)
CONFIG_FILE="$PROJECT_ROOT/.benchmarkrc.yml"
if [ -f "$CONFIG_FILE" ]; then
    ADB_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c.get('device',{}).get('adb',''))" 2>/dev/null)
    if [ -n "$ADB_PATH" ] && [ -x "$ADB_PATH" ]; then
        adb() { "$ADB_PATH" "$@"; }
    fi
fi
if ! command -v adb &>/dev/null && [ -x "/mnt/e/andorid/adb/adb.exe" ]; then
    export PATH=$PATH:/mnt/e/andorid/adb/
    adb() { /mnt/e/andorid/adb/adb.exe "$@"; }
fi

# Create directory on device
adb shell mkdir -p /data/local/tmp/benchmark

# Push binary
adb push $PROJECT_ROOT/build_android/src/cnn/benchmark_inference /data/local/tmp/benchmark/
adb shell chmod +x /data/local/tmp/benchmark/benchmark_inference
adb push $PROJECT_ROOT/build_android/src/llm/llm_benchmark /data/local/tmp/benchmark/
adb shell chmod +x /data/local/tmp/benchmark/llm_benchmark

# Push models
echo "Pushing models..."
adb push $PROJECT_ROOT/models /data/local/tmp/benchmark/models

echo
echo "=== Running benchmark ==="
echo "=================================================="

if [ "$GPU" = "yes" ]; then
    adb shell "cd /data/local/tmp/benchmark && ./benchmark_inference --backend $BACKEND --model $MODEL --precision $PRECISION --threads $THREADS --warmup $WARMUP --runs $RUNS --gpu"
else
    adb shell "cd /data/local/tmp/benchmark && ./benchmark_inference --backend $BACKEND --model $MODEL --precision $PRECISION --threads $THREADS --warmup $WARMUP --runs $RUNS"
fi

echo "=================================================="
echo "Done!"
