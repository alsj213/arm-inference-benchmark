#!/bin/bash
# Push and run benchmark on connected Android device

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

# Default build type
BUILD_TYPE="Release"

# Read ADB path from config file (priority), fallback to WSL2 path
CONFIG_FILE="$PROJECT_ROOT/.benchmarkrc.yml"
if [ -f "$CONFIG_FILE" ]; then
    ADB_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c.get('device',{}).get('adb',''))" 2>/dev/null)
    if [ -n "$ADB_PATH" ] && [ -x "$ADB_PATH" ]; then
        adb() { "$ADB_PATH" "$@"; }
    fi
fi
if ! command -v adb &>/dev/null && [ -x "/mnt/e/andorid/adb/adb.exe" ]; then
    adb() { /mnt/e/andorid/adb/adb.exe "$@"; }
fi

# Parse arguments for hardware control flags and build type
HARDWARE_CONTROL=true
GENERATE_REPORT=false
RESULTS_DIR=""
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
        --generate-report)
            GENERATE_REPORT=true
            shift
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options] [benchmark_args...]"
            echo ""
            echo "Options:"
            echo "  --build-type <type>    Build type: release or debug (default: release)"
            echo "  --no-hardware-control  Skip hardware control setup"
            echo "  --hardware-check       Check device status only"
            echo "  --generate-report      Generate Markdown report after test"
            echo "  --results-dir <path>   Results directory (default: auto-generated)"
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

# Push shared libraries (仅当前启用的后端)
echo "Pushing shared libraries..."
# 已停用的后端：TFLite、TVM（见 backup/all-backends 分支恢复）
# adb push "$PROJECT_ROOT/third_party/tflite_extracted/jni/arm64-v8a/libtensorflowlite_jni.so" /data/local/tmp/benchmark/ 2>/dev/null || true
if [ -f "$PROJECT_ROOT/third_party/onnxruntime/build/Android/$BUILD_TYPE/libonnxruntime.so" ]; then
    adb push "$PROJECT_ROOT/third_party/onnxruntime/build/Android/$BUILD_TYPE/libonnxruntime.so" /data/local/tmp/benchmark/ 2>/dev/null || true
fi
# libMNN.so（共享库版）
if [ "$BUILD_TYPE" = "Debug" ]; then
    MNN_SO_DIR="$PROJECT_ROOT/build_android_debug"
else
    MNN_SO_DIR="$PROJECT_ROOT/build_android"
fi
MNN_SO="$MNN_SO_DIR/third_party/MNN/OFF/arm64-v8a/libMNN.so"
if [ -f "$MNN_SO" ]; then
    adb push "$MNN_SO" /data/local/tmp/benchmark/ 2>/dev/null || true
fi
# adb push "$PROJECT_ROOT/third_party/tvm/build-android/libtvm_runtime.so" /data/local/tmp/benchmark/ 2>/dev/null || true

echo "=== Starting benchmark ==="
echo "Running: ./benchmark_inference $*"
echo "=================================================="

# 创建结果目录（如果需要生成报告）
if [ "$GENERATE_REPORT" = true ]; then
    if [ -z "$RESULTS_DIR" ]; then
        RESULTS_DIR="$PROJECT_ROOT/results/single_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$RESULTS_DIR"
    echo "Results directory: $RESULTS_DIR"

    # 运行测试并保存输出到日志文件
    LOG_FILE="$RESULTS_DIR/benchmark_output.log"
    adb shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=/data/local/tmp/benchmark ./benchmark_inference $@" 2>&1 | tee "$LOG_FILE"
else
    # 不生成报告，直接输出到终端
    adb shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=/data/local/tmp/benchmark ./benchmark_inference $@"
fi

echo "=================================================="

# 恢复测试环境（如果启用）
if [ "$HARDWARE_CONTROL" = true ]; then
    echo "=== 恢复测试环境 ==="
    "$SCRIPT_DIR/restore_test_environment.sh"
fi

# 生成报告
if [ "$GENERATE_REPORT" = true ]; then
    echo "=== 生成测试报告 ==="
    "$SCRIPT_DIR/generate_report.py" "$RESULTS_DIR"
fi

echo "Done!"
