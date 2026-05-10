#!/bin/bash
# 测试 MobileNetV2、ResNet50、YOLOv8n 的性能数据

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/profiling_utils.sh"

# 默认参数
BACKENDS=("mnn" "onnxrt")
MODELS=("mobilenetv2" "resnet50" "yolov8n")
THREADS=4
WARMUP=10
RUNS=100

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKENDS=("$2"); shift 2 ;;
        --model) MODELS=("$2"); shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --backend <name>  Backend to test (default: mnn onnxrt)"
            echo "  --model <name>    Model to test (default: mobilenetv2 resnet50 yolov8n)"
            echo "  --threads <num>   Number of threads (default: 4)"
            echo "  --warmup <num>    Warmup runs (default: 10)"
            echo "  --runs <num>      Benchmark runs (default: 100)"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# 主流程
main() {
    log_info "=== Performance Testing ==="
    log_info "Backends: ${BACKENDS[*]}"
    log_info "Models: ${MODELS[*]}"
    log_info "Threads: $THREADS, Warmup: $WARMUP, Runs: $RUNS"

    # Step 1: 初始化
    detect_adb || exit 1
    check_device || exit 1

    # Step 2: 创建结果目录
    create_result_dir "performance_test" "all"
    log_info "Results directory: $RESULT_DIR"

    # Step 3: 推送模型文件
    log_info "Pushing model files..."
    /mnt/e/andorid/adb/adb.exe push "$SCRIPT_DIR/models/classification/mobilenetv2/mobilenetv2.onnx" /data/local/tmp/benchmark/models/classification/mobilenetv2/ 2>/dev/null || true
    /mnt/e/andorid/adb/adb.exe push "$SCRIPT_DIR/models/classification/mobilenetv2/mobilenetv2_MNN.mnn" /data/local/tmp/benchmark/models/classification/mobilenetv2/ 2>/dev/null || true
    /mnt/e/andorid/adb/adb.exe push "$SCRIPT_DIR/models/classification/resnet50/resnet50.onnx" /data/local/tmp/benchmark/models/classification/resnet50/ 2>/dev/null || true
    /mnt/e/andorid/adb/adb.exe push "$SCRIPT_DIR/models/classification/resnet50/resnet50_MNN.mnn" /data/local/tmp/benchmark/models/classification/resnet50/ 2>/dev/null || true
    /mnt/e/andorid/adb/adb.exe push "$SCRIPT_DIR/models/detection/yolov8n/yolov8n.onnx" /data/local/tmp/benchmark/models/detection/yolov8n/ 2>/dev/null || true
    /mnt/e/andorid/adb/adb.exe push "$SCRIPT_DIR/models/detection/yolov8n/yolov8n_MNN.mnn" /data/local/tmp/benchmark/models/detection/yolov8n/ 2>/dev/null || true

    # Step 4: 运行测试
    for backend in "${BACKENDS[@]}"; do
        for model in "${MODELS[@]}"; do
            log_info "Testing $backend on $model..."
            run_test "$backend" "$model"
        done
    done

    # Step 5: 生成汇总报告
    generate_summary

    log_info "=== Performance Testing Complete ==="
    log_info "Results saved to: $RESULT_DIR"
}

run_test() {
    local backend="$1"
    local model="$2"

    local output_file="$RESULT_DIR/${backend}_${model}_result.txt"

    /mnt/e/andorid/adb/adb.exe shell "cd /data/local/tmp/benchmark && \
        LD_LIBRARY_PATH=. ./benchmark_inference \
        --backend $backend \
        --model $model \
        --threads $THREADS \
        --warmup $WARMUP \
        --runs $RUNS" 2>&1 | tee "$output_file"

    log_info "Result saved to: $output_file"
}

generate_summary() {
    local summary_file="$RESULT_DIR/summary.md"

    cat > "$summary_file" << EOF
# Performance Test Summary

## Test Configuration
- **Date**: $(date '+%Y-%m-%d %H:%M:%S')
- **Threads**: $THREADS
- **Warmup**: $WARMUP
- **Runs**: $RUNS

## Results

EOF

    for backend in "${BACKENDS[@]}"; do
        for model in "${MODELS[@]}"; do
            local result_file="$RESULT_DIR/${backend}_${model}_result.txt"
            if [ -f "$result_file" ]; then
                cat >> "$summary_file" << EOF
### $backend - $model
\`\`\`
$(grep -A 20 "Performance Results" "$result_file" 2>/dev/null | head -15)
\`\`\`

EOF
            fi
        done
    done

    log_info "Summary generated: $summary_file"
}

main "$@"
