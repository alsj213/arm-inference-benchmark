#!/bin/bash
# atrace_capture.sh - 使用 atrace 采集系统级 trace
# 输出可用 chrome://tracing 打开的 trace 文件

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profiling_utils.sh"

# 默认参数
BACKEND="mnn"
MODEL="mobilenetv2"
THREADS=4
DURATION=5
BUFFER_SIZE=32768
CATEGORIES="sched,freq,idle"
BENCHMARK_RUNS=1000

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --buffer-size) BUFFER_SIZE="$2"; shift 2 ;;
        --categories) CATEGORIES="$2"; shift 2 ;;
        --runs) BENCHMARK_RUNS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --backend <name>        Backend to test (default: mnn)"
            echo "  --model <name>          Model to test (default: mobilenetv2)"
            echo "  --threads <num>         Number of threads (default: 4)"
            echo "  --duration <sec>        Trace duration in seconds (default: 5)"
            echo "  --buffer-size <kb>      Buffer size in KB (default: 32768)"
            echo "  --categories <cats>     Trace categories (default: sched,freq,idle)"
            echo "  --runs <num>            Benchmark runs (default: 1000)"
            echo ""
            echo "Common categories:"
            echo "  sched      - CPU scheduling (thread migration, context switches)"
            echo "  freq       - CPU frequency changes"
            echo "  idle       - CPU idle states"
            echo "  binder_driver - Binder IPC"
            echo "  gfx        - Graphics"
            echo "  view       - View system"
            echo "  wm         - Window manager"
            echo "  am         - Activity manager"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# 主流程
main() {
    log_info "=== Atrace System Trace Capture ==="
    log_info "Backend: $BACKEND, Model: $MODEL, Threads: $THREADS"
    log_info "Duration: ${DURATION}s, Categories: $CATEGORIES"

    # Step 1: 初始化
    detect_adb || exit 1
    check_device || exit 1

    # Step 2: 创建结果目录
    create_result_dir "$MODEL" "$BACKEND"

    # Step 3: 推送文件
    push_benchmark_files

    # Step 4: 保存设备信息
    get_device_info "$RESULT_DIR/device_info.txt"

    # Step 5: 启动 benchmark（后台）
    log_info "Starting benchmark in background..."
    local device_dir="/data/local/tmp/benchmark"
    adb_cmd shell "cd $device_dir && \
        LD_LIBRARY_PATH=. nohup ./benchmark_inference \
        --backend $BACKEND \
        --model $MODEL \
        --threads $THREADS \
        --warmup 10 \
        --runs $BENCHMARK_RUNS \
        > profiling/benchmark_output.txt 2>&1 &"

    sleep 2

    # Step 6: atrace 采集
    log_info "Starting atrace capture (${DURATION}s)..."

    # 将逗号分隔的类别转换为空格分隔
    local atrace_categories=$(echo "$CATEGORIES" | tr ',' ' ')

    # 使用 -t 参数指定采集时长，并过滤掉开头的 "capturing trace..." 行
    adb_cmd shell "atrace -t $DURATION -b $BUFFER_SIZE $atrace_categories 2>/dev/null | grep -v '^capturing trace' > /data/local/tmp/benchmark/profiling/trace.txt"

    log_info "Atrace capture completed"

    # Step 7: 等待 benchmark 完成
    log_info "Waiting for benchmark to complete..."
    while adb_cmd shell "pidof benchmark_inference" &>/dev/null; do
        sleep 1
    done

    # Step 8: 拉取数据
    log_info "Pulling results..."
    adb_cmd pull "$device_dir/profiling/trace.txt" "$RESULT_DIR/atrace/"
    adb_cmd pull "$device_dir/profiling/benchmark_output.txt" "$RESULT_DIR/"

    # Step 9: 输出摘要
    print_summary

    log_info "=== Trace Capture Complete ==="
    log_info "Results saved to: $RESULT_DIR"
}

print_summary() {
    echo ""
    echo "========================================="
    echo "  Atrace Capture Summary"
    echo "========================================="
    echo "Backend: $BACKEND"
    echo "Model: $MODEL"
    echo "Duration: ${DURATION}s"
    echo "Categories: $CATEGORIES"
    echo ""
    echo "Output files:"
    echo "  - trace.txt: $RESULT_DIR/atrace/trace.txt"
    echo ""
    echo "How to view:"
    echo "  Option 1 (Recommended): Perfetto UI"
    echo "    - Open https://ui.perfetto.dev"
    echo "    - Click 'Open trace file' and select trace.txt"
    echo ""
    echo "  Option 2: Systrace (requires conversion)"
    echo "    - Run: python3 \$ANDROID_NDK/simpleperf/convert_trace.py trace.txt trace.html"
    echo "    - Open trace.html in Chrome"
    echo ""
    echo "  Note: Raw ftrace format may not work directly in chrome://tracing"
    echo "  For best compatibility, use perfetto_trace.sh instead"
    echo "========================================="
}

main "$@"
