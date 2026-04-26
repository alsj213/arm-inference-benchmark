#!/bin/bash
# framework_profiling.sh - 各推理框架内置 profiling
# 支持 MNN、ONNX Runtime、ncnn、TFLite 的逐算子耗时分析

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profiling_utils.sh"

# 默认参数
BACKEND="mnn"
MODEL="mobilenetv2"
THREADS=4
BENCHMARK_RUNS=100

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --runs) BENCHMARK_RUNS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --backend <name>      Backend to test (default: mnn)"
            echo "  --model <name>        Model to test (default: mobilenetv2)"
            echo "  --threads <num>       Number of threads (default: 4)"
            echo "  --runs <num>          Benchmark runs (default: 100)"
            echo ""
            echo "Supported backends:"
            echo "  mnn      - MNN profiling (MNN_OPENCL_PROFILE=1)"
            echo "  onnxrt   - ONNX Runtime profiling (EnableProfiling)"
            echo "  ncnn     - ncnn profiling (NCNN_PROFILE=1)"
            echo "  tflite   - TFLite profiling (--enable_op_profiling)"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# MNN profiling
run_mnn_profiling() {
    log_info "Running MNN profiling..."

    local device_dir="/data/local/tmp/benchmark"
    local output_file="profiling/mnn_profile.txt"

    # MNN 使用环境变量开启 profiling
    adb_cmd shell "cd $device_dir && \
        MNN_OPENCL_PROFILE=1 \
        LD_LIBRARY_PATH=. ./benchmark_inference \
        --backend mnn \
        --model $MODEL \
        --threads $THREADS \
        --warmup 5 \
        --runs 10 \
        2>&1 | tee $output_file"

    adb_cmd pull "$device_dir/$output_file" "$RESULT_DIR/framework/"

    # 提取关键信息
    log_info "MNN profile saved to: $RESULT_DIR/framework/mnn_profile.txt"
}

# ONNX Runtime profiling
run_ort_profiling() {
    log_info "Running ONNX Runtime profiling..."

    local device_dir="/data/local/tmp/benchmark"
    local output_file="profiling/ort_profile.json"

    # ORT profiling 需要在代码中启用，这里通过环境变量
    adb_cmd shell "cd $device_dir && \
        ORT_ENABLE_profiling=1 \
        ORT_profile_file=$output_file \
        LD_LIBRARY_PATH=. ./benchmark_inference \
        --backend onnxrt \
        --model $MODEL \
        --threads $THREADS \
        --warmup 5 \
        --runs 10"

    adb_cmd pull "$device_dir/$output_file" "$RESULT_DIR/framework/"

    log_info "ORT profile saved to: $RESULT_DIR/framework/ort_profile.json"
}

# ncnn profiling
run_ncnn_profiling() {
    log_info "Running ncnn profiling..."

    local device_dir="/data/local/tmp/benchmark"
    local output_file="profiling/ncnn_profile.txt"

    # ncnn 使用 NCNN_PROFILE 环境变量
    adb_cmd shell "cd $device_dir && \
        NCNN_PROFILE=1 \
        LD_LIBRARY_PATH=. ./benchmark_inference \
        --backend ncnn \
        --model $MODEL \
        --threads $THREADS \
        --warmup 5 \
        --runs 10 \
        2>&1 | tee $output_file"

    adb_cmd pull "$device_dir/$output_file" "$RESULT_DIR/framework/"

    log_info "ncnn profile saved to: $RESULT_DIR/framework/ncnn_profile.txt"
}

# TFLite profiling
run_tflite_profiling() {
    log_info "Running TFLite profiling..."

    local device_dir="/data/local/tmp/benchmark"
    local output_file="profiling/tflite_profile.csv"

    # TFLite 可以通过 benchmark_model 工具获取逐算子耗时
    # 这里使用我们的 benchmark 程序，需要在代码中支持
    adb_cmd shell "cd $device_dir && \
        TFLITE_PROFILING=1 \
        LD_LIBRARY_PATH=. ./benchmark_inference \
        --backend tflite \
        --model $MODEL \
        --threads $THREADS \
        --warmup 5 \
        --runs 10 \
        2>&1 | tee profiling/tflite_profile.txt"

    adb_cmd pull "$device_dir/profiling/tflite_profile.txt" "$RESULT_DIR/framework/"

    log_info "TFLite profile saved to: $RESULT_DIR/framework/tflite_profile.txt"
}

# 主流程
main() {
    log_info "=== Framework Profiling ==="
    log_info "Backend: $BACKEND, Model: $MODEL, Threads: $THREADS"

    # Step 1: 初始化
    detect_adb || exit 1
    check_device || exit 1

    # Step 2: 创建结果目录
    create_result_dir "$MODEL" "$BACKEND"

    # Step 3: 推送文件
    push_benchmark_files

    # Step 4: 保存设备信息
    get_device_info "$RESULT_DIR/device_info.txt"

    # Step 5: 根据 backend 运行对应的 profiling
    case "$BACKEND" in
        mnn)
            run_mnn_profiling
            ;;
        onnxrt|ort)
            run_ort_profiling
            ;;
        ncnn)
            run_ncnn_profiling
            ;;
        tflite)
            run_tflite_profiling
            ;;
        all)
            run_mnn_profiling
            run_ort_profiling
            run_ncnn_profiling
            run_tflite_profiling
            ;;
        *)
            log_error "Unsupported backend for framework profiling: $BACKEND"
            exit 1
            ;;
    esac

    # Step 6: 输出摘要
    print_summary

    log_info "=== Framework Profiling Complete ==="
    log_info "Results saved to: $RESULT_DIR"
}

print_summary() {
    echo ""
    echo "========================================="
    echo "  Framework Profiling Summary"
    echo "========================================="
    echo "Backend: $BACKEND"
    echo "Model: $MODEL"
    echo ""
    echo "Output files:"
    ls -la "$RESULT_DIR/framework/" 2>/dev/null
    echo ""
    echo "How to interpret:"
    echo "  - MNN: Look for layer-by-layer timing in output"
    echo "  - ORT: Open JSON in chrome://tracing"
    echo "  - ncnn: Look for layer timing in output"
    echo "  - TFLite: Check CSV for per-op timing"
    echo "========================================="
}

main "$@"
