#!/bin/bash
# Push and run benchmark on connected Android device

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

BINARY=$PROJECT_ROOT/build_android/src/benchmark_inference

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found: $BINARY"
    echo "Please run ./scripts/build_android.sh first"
    exit 1
fi

# Use adb from Windows (WSL2 case)
if [ -x "/mnt/e/andorid/adb/adb.exe" ]; then
    adb() {
        /mnt/e/andorid/adb/adb.exe "$@"
    }
fi

echo "=== Pushing to device ==="

# Create directory on device
adb shell mkdir -p /data/local/tmp/benchmark

# Push binary
adb push "$BINARY" /data/local/tmp/benchmark/
adb shell chmod +x /data/local/tmp/benchmark/benchmark_inference

# Push models
echo "Pushing models..."
adb push "$PROJECT_ROOT/models/classification" /data/local/tmp/benchmark/models/classification

# Push shared libraries (TFLite, ONNX Runtime)
echo "Pushing shared libraries..."
adb push "$PROJECT_ROOT/third_party/tflite_extracted/jni/arm64-v8a/libtensorflowlite_jni.so" /data/local/tmp/benchmark/ 2>/dev/null || true
adb push "$PROJECT_ROOT/third_party/onnxruntime/lib-android/aarch64/libonnxruntime.so" /data/local/tmp/benchmark/ 2>/dev/null || true
adb push "$PROJECT_ROOT/third_party/tvm/build-android/libtvm_runtime.so" /data/local/tmp/benchmark/ 2>/dev/null || true

echo "=== Starting benchmark ==="
echo "Running: ./benchmark_inference $*"
echo "=================================================="

adb shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=/data/local/tmp/benchmark ./benchmark_inference $@"

echo "=================================================="
echo "Done!"
