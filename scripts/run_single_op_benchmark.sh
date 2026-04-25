#!/bin/bash
#
# 单算子性能基准测试脚本
# 在 Android 设备上批量测试所有单算子模型的推理延迟
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGLE_OPS_DIR="$PROJECT_ROOT/models/single_ops"
DEVICE_DIR="/data/local/tmp/single_ops"
BACKENDS=${BACKENDS:-"ncnn,mnn,onnxruntime"}
WARMUP=${WARMUP:-10}
RUNS=${RUNS:-100}
PRECISION=${PRECISION:-"fp32"}

echo "=== Single Operator Benchmark ==="
echo "Backends: $BACKENDS"
echo "Warmup: $WARMUP, Runs: $RUNS"
echo "Precision: $PRECISION"
echo

# 推送到设备
echo "Pushing benchmark binary to device..."
adb push "$PROJECT_ROOT/build_android/src/benchmark_inference" /data/local/tmp/ 2>/dev/null
adb shell "chmod +x /data/local/tmp/benchmark_inference"

echo "Pushing models to device..."
adb shell "mkdir -p $DEVICE_DIR"

for backend in $(echo $BACKENDS | tr ',' ' '); do
    case $backend in
        ncnn)
            adb push "$SINGLE_OPS_DIR/ncnn" "$DEVICE_DIR/" 2>/dev/null
            ;;
        mnn)
            adb push "$SINGLE_OPS_DIR/mnn" "$DEVICE_DIR/" 2>/dev/null
            ;;
        onnxruntime)
            # ONNX Runtime 使用原始 ONNX 文件
            adb push "$SINGLE_OPS_DIR"/*.onnx "$DEVICE_DIR/" 2>/dev/null
            ;;
    esac
done

echo
echo "Running benchmarks..."
echo "======================"

RESULTS_FILE="$PROJECT_ROOT/results/single_ops_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p "$(dirname $RESULTS_FILE)"
echo "op_name,backend,precision,mean_ms,min_ms,max_ms,std_ms" > "$RESULTS_FILE"

# 获取所有算子名称
OPS=$(cd "$SINGLE_OPS_DIR" && ls *.onnx 2>/dev/null | sed 's/\.onnx$//')

for op in $OPS; do
    echo -n "Testing $op..."

    for backend in $(echo $BACKENDS | tr ',' ' '); do
        case $backend in
            ncnn)
                MODEL_PATH="$DEVICE_DIR/ncnn/$op"
                ;;
            mnn)
                MODEL_PATH="$DEVICE_DIR/mnn/$op.mnn"
                ;;
            onnxruntime)
                MODEL_PATH="$DEVICE_DIR/$op.onnx"
                ;;
        esac

        result=$(adb shell "cd /data/local/tmp && ./benchmark_inference --backend $backend --model $MODEL_PATH --precision $PRECISION --warmup $WARMUP --runs $RUNS 2>/dev/null" | tail -1)

        if [[ "$result" == *"mean"* ]]; then
            # 解析输出: "mean=0.12ms, min=0.10ms, max=0.15ms, std=0.01ms"
            mean=$(echo "$result" | grep -oP 'mean=\K[0-9.]+')
            min=$(echo "$result" | grep -oP 'min=\K[0-9.]+')
            max=$(echo "$result" | grep -oP 'max=\K[0-9.]+')
            std=$(echo "$result" | grep -oP 'std=\K[0-9.]+')
            echo "$op,$backend,$PRECISION,$mean,$min,$max,$std" >> "$RESULTS_FILE"
            echo -n " $backend:${mean}ms"
        else
            echo -n " $backend:FAIL"
        fi
    done
    echo
done

echo
echo "======================"
echo "Benchmark complete!"
echo "Results saved to: $RESULTS_FILE"
echo
echo "To analyze results, run:"
echo "  python3 scripts/analyze_single_op_results.py $RESULTS_FILE"
