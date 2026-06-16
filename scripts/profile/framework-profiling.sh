#!/bin/bash
# framework_profiling.sh - ORT 和 MNN 逐算子 profiling
# 支持 ONNX Runtime 和 MNN 的逐算子耗时分析

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profiling-utils.sh"

# 默认参数
BACKEND="mnn"
MODEL="mobilenetv2"
THREADS=4
BENCHMARK_RUNS=10
PROFILE_FORMAT="json"
BUILD_TYPE="release"  # 默认使用 Release 版本进行性能测试

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --runs) BENCHMARK_RUNS="$2"; shift 2 ;;
        --format) PROFILE_FORMAT="$2"; shift 2 ;;
        --build-type) BUILD_TYPE="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --backend <name>    Backend: mnn or onnxrt (default: mnn)"
            echo "  --model <name>      Model to test (default: mobilenetv2)"
            echo "  --threads <num>     Number of threads (default: 4)"
            echo "  --runs <num>        Benchmark runs (default: 10)"
            echo "  --format <type>     Output format: json or txt (default: json)"
            echo "  --build-type <type> Build type: debug or release (default: release)"
            echo ""
            echo "Supported backends:"
            echo "  mnn      - MNN profiling (逐层耗时)"
            echo "  onnxrt   - ONNX Runtime profiling (JSON 格式)"
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
        MNN_PROFILING=1 \
        MNN_PROFILING_FILE=$output_file \
        LD_LIBRARY_PATH=. ./benchmark_inference \
        --backend mnn \
        --model $MODEL \
        --threads $THREADS \
        --warmup 5 \
        --runs $BENCHMARK_RUNS \
        2>&1 | tee profiling/mnn_profile_console.txt"

    adb_cmd pull "$device_dir/$output_file" "$RESULT_DIR/framework/"
    adb_cmd pull "$device_dir/profiling/mnn_profile_console.txt" "$RESULT_DIR/framework/"

    log_info "MNN profile saved to: $RESULT_DIR/framework/mnn_profile.txt"
}

# ONNX Runtime profiling
run_ort_profiling() {
    log_info "Running ONNX Runtime profiling..."

    local device_dir="/data/local/tmp/benchmark"
    # ORT 会自动添加 _timestamp.json 后缀，所以这里只指定前缀（不带 .json）
    local output_prefix="profiling/ort_profile"

    # ORT 通过 --profiling 参数启用 profiling
    # 注意：profiling 文件会保存在设备上的当前工作目录
    # ORT 会生成文件如: profiling/ort_profile_20260501_123456.json
    adb_cmd shell "cd $device_dir && \
        LD_LIBRARY_PATH=. ./benchmark_inference \
        --backend onnxrt \
        --model $MODEL \
        --threads $THREADS \
        --warmup 5 \
        --runs $BENCHMARK_RUNS \
        --profiling $output_prefix"

    # 拉取 profiling 文件（ORT 会生成 ort_profile_20260501_123456.json 格式）
    # 获取最新的 profiling 文件
    local latest_profile=$(adb_cmd shell "ls -t $device_dir/profiling/ort_profile_*.json 2>/dev/null | head -n 1")
    latest_profile=$(echo "$latest_profile" | tr -d '\r' | tr -d '\n')
    if [ -n "$latest_profile" ]; then
        log_info "Found profiling file: $latest_profile"
        adb_cmd pull "$latest_profile" "$RESULT_DIR/framework/ort_profile.json" 2>/dev/null || true
        if [ -f "$RESULT_DIR/framework/ort_profile.json" ]; then
            log_info "ORT profile saved to: $RESULT_DIR/framework/ort_profile.json"
        else
            log_warning "Failed to pull ORT profiling file"
        fi
    else
        log_warning "No ORT profiling file found"
    fi

    log_info "View with: chrome://tracing -> Load $RESULT_DIR/framework/ort_profile.json"
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
    push_benchmark_files "" "" "$BUILD_TYPE"

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
        *)
            log_error "Unsupported backend for framework profiling: $BACKEND"
            log_error "Supported backends: mnn, onnxrt"
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
    echo "Threads: $THREADS"
    echo ""
    echo "Output files:"
    ls -la "$RESULT_DIR/framework/" 2>/dev/null
    echo ""
    echo "How to interpret:"
    if [ "$BACKEND" = "mnn" ]; then
        echo "  - MNN: Check mnn_profile.txt for layer-by-layer timing"
    elif [ "$BACKEND" = "onnxrt" ] || [ "$BACKEND" = "ort" ]; then
        echo "  - ORT: Open ort_profile.json in chrome://tracing"
    fi
    echo "========================================="
}

main "$@"
