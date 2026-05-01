#!/bin/bash
# Push and run benchmark on connected Android device

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

# Default build type
BUILD_TYPE="Release"

# Use adb from Windows (WSL2 case)
if [ -x "/mnt/e/andorid/adb/adb.exe" ]; then
    adb() {
        /mnt/e/andorid/adb/adb.exe "$@"
    }
fi

# Parse arguments for hardware control flags and build type
HARDWARE_CONTROL=true
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-hardware-control)
            HARDWARE_CONTROL=false
            shift
            ;;
        --hardware-check)
            # Just check device status, don't run benchmark
            echo "=== 设备状态检查 ==="
            adb shell "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            adb shell "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
            adb shell "cat /sys/class/thermal/thermal_zone0/temp"
            exit 0
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options] [benchmark_args...]"
            echo ""
            echo "Options:"
            echo "  --build-type <type>    Build type: release or debug (default: release)"
            echo "  --no-hardware-control  Skip hardware control setup"
            echo "  --hardware-check       Check device status only"
            echo "  --help                 Show this help message"
            echo ""
            echo "Benchmark arguments are passed through to benchmark_inference"
            exit 0
            ;;
        *)
            # Keep other arguments for benchmark
            break
            ;;
    esac
done

# Select binary based on build type
if [ "$BUILD_TYPE" = "Debug" ]; then
    BINARY=$PROJECT_ROOT/build_android_debug/src/benchmark_inference
else
    BINARY=$PROJECT_ROOT/build_android/src/benchmark_inference
fi

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found: $BINARY"
    echo "Please run ./scripts/build_android.sh --$BUILD_TYPE first"
    exit 1
fi

# 设置测试环境（如果启用）
if [ "$HARDWARE_CONTROL" = true ]; then
    echo "=== 设置测试环境 ==="
    "$SCRIPT_DIR/setup_test_environment.sh"
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

# 恢复测试环境（如果启用）
if [ "$HARDWARE_CONTROL" = true ]; then
    echo "=== 恢复测试环境 ==="
    "$SCRIPT_DIR/restore_test_environment.sh"
fi

echo "Done!"
