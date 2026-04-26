#!/bin/bash
# profile_benchmark.sh - 集成入口：benchmark + profiling 一步到位
# 支持 simpleperf、atrace、perfetto、框架内置 profiling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profiling_utils.sh"

# 默认参数
BACKEND="mnn"
MODEL="mobilenetv2"
THREADS=4
BENCHMARK_RUNS=100
PROFILE_TOOLS="simpleperf"
SIMPLEPERF_DURATION=10
ATRACE_DURATION=5
PERFETTO_DURATION=5
PERFETTO_PRESET="cpu"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --runs) BENCHMARK_RUNS="$2"; shift 2 ;;
        --profile) PROFILE_TOOLS="$2"; shift 2 ;;
        --simpleperf-duration) SIMPLEPERF_DURATION="$2"; shift 2 ;;
        --atrace-duration) ATRACE_DURATION="$2"; shift 2 ;;
        --perfetto-duration) PERFETTO_DURATION="$2"; shift 2 ;;
        --perfetto-preset) PERFETTO_PRESET="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --backend <name>              Backend to test (default: mnn)"
            echo "  --model <name>                Model to test (default: mobilenetv2)"
            echo "  --threads <num>               Number of threads (default: 4)"
            echo "  --runs <num>                  Benchmark runs (default: 100)"
            echo "  --profile <tools>             Profiling tools, comma-separated (default: simpleperf)"
            echo "                                Options: simpleperf,atrace,perfetto,framework,all"
            echo "  --simpleperf-duration <sec>   Simpleperf sampling duration (default: 10)"
            echo "  --atrace-duration <sec>       Atrace capture duration (default: 5)"
            echo "  --perfetto-duration <sec>     Perfetto capture duration (default: 5)"
            echo "  --perfetto-preset <preset>    Perfetto preset: cpu/memory/full (default: cpu)"
            echo ""
            echo "Examples:"
            echo "  # Simpleperf only"
            echo "  $0 --backend mnn --model mobilenetv2 --profile simpleperf"
            echo ""
            echo "  # All profiling tools"
            echo "  $0 --backend mnn --model mobilenetv2 --profile all"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# 解析 profiling 工具列表
IFS=',' read -ra TOOLS <<< "$PROFILE_TOOLS"

# 检查是否启用特定工具
is_tool_enabled() {
    local tool="$1"
    for t in "${TOOLS[@]}"; do
        if [ "$t" = "$tool" ] || [ "$t" = "all" ]; then
            return 0
        fi
    done
    return 1
}

# 主流程
main() {
    log_info "=== Benchmark + Profiling ==="
    log_info "Backend: $BACKEND, Model: $MODEL, Threads: $THREADS, Runs: $BENCHMARK_RUNS"
    log_info "Profiling tools: $PROFILE_TOOLS"

    # Step 1: 初始化
    detect_adb || exit 1
    check_device || exit 1
    check_root

    # Step 2: 创建结果目录（只创建启用的工具目录）
    create_result_dir "$MODEL" "$BACKEND" "$PROFILE_TOOLS"

    # Step 3: 推送文件
    push_benchmark_files
    if is_tool_enabled "simpleperf"; then
        push_simpleperf
    fi

    # Step 4: 保存设备信息
    get_device_info "$RESULT_DIR/device_info.txt"

    # 采集测试前温度
    log_info "Collecting pre-test thermal data..."
    adb_cmd shell "cat /sys/class/thermal/thermal_zone*/temp" > "$RESULT_DIR/thermal_pre.txt" 2>/dev/null

    # Step 5: 创建包装脚本
    local device_dir="/data/local/tmp/benchmark"
    adb_cmd shell "mkdir -p $device_dir/profiling"

    create_wrapper_script

    # Step 6: 执行包装脚本
    log_info "Running benchmark with profiling..."
    adb_cmd shell "sh $device_dir/profiling/run_profiling.sh"

    # Step 7: 采集测试后温度
    log_info "Collecting post-test thermal data..."
    adb_cmd shell "cat /sys/class/thermal/thermal_zone*/temp" > "$RESULT_DIR/thermal_post.txt" 2>/dev/null

    # Step 8: 拉取所有数据
    pull_results

    # Step 9: 生成 simpleperf 报告
    if is_tool_enabled "simpleperf"; then
        generate_simpleperf_reports
    fi

    # Step 10: 生成汇总报告
    generate_summary

    # Step 11: 输出摘要
    print_summary

    log_info "=== Profiling Complete ==="
    log_info "Results saved to: $RESULT_DIR"
}

# 创建包装脚本
create_wrapper_script() {
    local device_dir="/data/local/tmp/benchmark"

    # 构建 profiling 命令
    local simpleperf_cmd=""
    local atrace_start_cmd=""
    local atrace_stop_cmd=""
    local perfetto_cmd=""

    if is_tool_enabled "simpleperf"; then
        simpleperf_cmd="
# 启动 simpleperf（后台）
/data/local/tmp/simpleperf record \\
    -p \$BENCH_PID \\
    -e cpu-cycles \\
    -f 4000 \\
    --call-graph fp \\
    --duration $SIMPLEPERF_DURATION \\
    -o profiling/perf.data &
SIMPLEPERF_PID=\$!"
    fi

    if is_tool_enabled "atrace"; then
        atrace_start_cmd="
# 启动 atrace
atrace --async_start -c -b 32768 sched freq idle"
        atrace_stop_cmd="
# 停止 atrace
atrace --async_stop -o profiling/trace.txt"
    fi

    if is_tool_enabled "perfetto"; then
        perfetto_cmd="
# 启动 perfetto
cat profiling/config.pbtx | perfetto --txt -c - -o /data/misc/perfetto-traces/benchmark_trace.perfetto-trace &
PERFETTO_PID=\$!"
    fi

    # 生成 perfetto 配置
    if is_tool_enabled "perfetto"; then
        local duration_ms=$((PERFETTO_DURATION * 1000))
        adb_cmd shell "cat > $device_dir/profiling/config.pbtx << 'CONFIG_EOF'
buffers: {
    size_kb: 32768
    fill_policy: RING_BUFFER
}
duration_ms: $duration_ms
data_sources: {
    config {
        name: \"linux.ftrace\"
        ftrace_config {
            ftrace_events: \"sched/sched_switch\"
            ftrace_events: \"sched/sched_wakeup\"
            ftrace_events: \"power/cpu_frequency\"
            ftrace_events: \"power/cpu_idle\"
            atrace_categories: \"sched\"
            atrace_categories: \"freq\"
            atrace_categories: \"idle\"
            atrace_apps: \"*\"
        }
    }
}
data_sources: {
    config {
        name: \"linux.process_stats\"
        process_stats_config {
            scan_all_processes_on_start: true
            proc_stats_poll_ms: 1000
        }
    }
}
CONFIG_EOF
chmod 644 $device_dir/profiling/config.pbtx"
    fi

    # 创建包装脚本
    adb_cmd shell "cat > $device_dir/profiling/run_profiling.sh << 'SCRIPT_EOF'
#!/bin/sh
cd /data/local/tmp/benchmark

# 启动 benchmark
LD_LIBRARY_PATH=. ./benchmark_inference \\
    --backend $BACKEND \\
    --model $MODEL \\
    --threads $THREADS \\
    --warmup 10 \\
    --runs $BENCHMARK_RUNS > profiling/benchmark_output.txt 2>&1 &
BENCH_PID=\$!
echo \$BENCH_PID > profiling/benchmark.pid

# 等待 warmup
sleep 3

$atrace_start_cmd
$simpleperf_cmd
$perfetto_cmd

# 等待 simpleperf 完成
if [ -n \"\$SIMPLEPERF_PID\" ]; then
    wait \$SIMPLEPERF_PID
fi

# 等待 perfetto 完成
if [ -n \"\$PERFETTO_PID\" ]; then
    wait \$PERFETTO_PID
    cp /data/misc/perfetto-traces/benchmark_trace.perfetto-trace profiling/trace.perfetto-trace
    chmod 644 profiling/trace.perfetto-trace
fi

$atrace_stop_cmd

# 等待 benchmark 完成
wait \$BENCH_PID
SCRIPT_EOF
chmod 755 $device_dir/profiling/run_profiling.sh"
}

# 拉取结果
pull_results() {
    log_info "Pulling results..."
    local device_dir="/data/local/tmp/benchmark"

    adb_cmd pull "$device_dir/profiling/benchmark_output.txt" "$RESULT_DIR/" 2>/dev/null

    if is_tool_enabled "simpleperf"; then
        adb_cmd pull "$device_dir/profiling/perf.data" "$RESULT_DIR/simpleperf/" 2>/dev/null
    fi

    if is_tool_enabled "atrace"; then
        adb_cmd pull "$device_dir/profiling/trace.txt" "$RESULT_DIR/atrace/" 2>/dev/null
    fi

    if is_tool_enabled "perfetto"; then
        adb_cmd pull "$device_dir/profiling/trace.perfetto-trace" "$RESULT_DIR/perfetto/" 2>/dev/null
        adb_cmd pull "$device_dir/profiling/config.pbtx" "$RESULT_DIR/perfetto/" 2>/dev/null
    fi
}

# 生成 simpleperf 报告
generate_simpleperf_reports() {
    local perf_data="$RESULT_DIR/simpleperf/perf.data"
    if [ ! -f "$perf_data" ]; then
        return
    fi

    log_info "Generating simpleperf reports..."

    local simpleperf_bin=""
    if [ -f "$ANDROID_NDK/simpleperf/bin/linux/x86_64/simpleperf" ]; then
        simpleperf_bin="$ANDROID_NDK/simpleperf/bin/linux/x86_64/simpleperf"
    elif command -v simpleperf &>/dev/null; then
        simpleperf_bin="simpleperf"
    else
        log_warn "simpleperf not found locally, reports not generated"
        return
    fi

    # 文本报告
    $simpleperf_bin report -i "$perf_data" --sort dso,symbol -n > "$RESULT_DIR/simpleperf/report_functions.txt" 2>/dev/null
    $simpleperf_bin report -i "$perf_data" --sort dso -n > "$RESULT_DIR/simpleperf/report_dso.txt" 2>/dev/null
    $simpleperf_bin report -i "$perf_data" --sort comm,dso,symbol --show-callchain > "$RESULT_DIR/simpleperf/report_callchain.txt" 2>/dev/null

    # 火焰图
    generate_flamegraph "$perf_data"

    log_info "simpleperf reports generated"
}

# 生成火焰图
generate_flamegraph() {
    local perf_data="$1"
    local folded_file="$RESULT_DIR/simpleperf/out.folded"
    local svg_file="$RESULT_DIR/simpleperf/flamegraph.svg"

    # 检查 simpleperf stackcollapse.py
    local stackcollapse_py="$ANDROID_NDK/simpleperf/stackcollapse.py"
    if [ ! -f "$stackcollapse_py" ]; then
        log_warn "stackcollapse.py not found, skipping flamegraph"
        return
    fi

    # 检查 FlameGraph 工具
    local flamegraph_pl="./tools/FlameGraph/flamegraph.pl"
    if [ ! -f "$flamegraph_pl" ]; then
        log_warn "FlameGraph not found. Run: git clone https://github.com/brendangregg/FlameGraph.git tools/FlameGraph"
        return
    fi

    # 生成
    log_info "Generating flamegraph..."
    python3 "$stackcollapse_py" -i "$perf_data" > "$folded_file" 2>/dev/null
    "$flamegraph_pl" "$folded_file" > "$svg_file" 2>/dev/null

    if [ -f "$svg_file" ]; then
        log_info "Flamegraph: $svg_file"
    fi
}

# 生成汇总报告
generate_summary() {
    local summary_file="$RESULT_DIR/summary.md"

    cat > "$summary_file" << EOF
# Profiling Summary

## Test Configuration
- **Backend**: $BACKEND
- **Model**: $MODEL
- **Threads**: $THREADS
- **Benchmark Runs**: $BENCHMARK_RUNS
- **Profiling Tools**: $PROFILE_TOOLS
- **Date**: $(date '+%Y-%m-%d %H:%M:%S')

## Device Info
$(cat "$RESULT_DIR/device_info.txt" 2>/dev/null | head -20)

## Benchmark Results
\`\`\`
$(cat "$RESULT_DIR/benchmark_output.txt" 2>/dev/null | tail -20)
\`\`\`

## Profiling Results

EOF

    if is_tool_enabled "simpleperf" && [ -f "$RESULT_DIR/simpleperf/report_functions.txt" ]; then
        cat >> "$summary_file" << EOF
### Simpleperf Top 10 Hot Functions
\`\`\`
$(head -20 "$RESULT_DIR/simpleperf/report_functions.txt" 2>/dev/null)
\`\`\`

EOF
    fi

    if is_tool_enabled "atrace"; then
        cat >> "$summary_file" << EOF
### Atrace
- View with: Chrome -> chrome://tracing -> Load trace.txt
- File: $RESULT_DIR/atrace/trace.txt

EOF
    fi

    if is_tool_enabled "perfetto"; then
        cat >> "$summary_file" << EOF
### Perfetto
- View with: https://ui.perfetto.dev -> Open trace.perfetto-trace
- File: $RESULT_DIR/perfetto/trace.perfetto-trace

EOF
    fi

    log_info "Summary generated: $summary_file"
}

# 打印摘要
print_summary() {
    echo ""
    echo "========================================="
    echo "  Profiling Complete"
    echo "========================================="
    echo "Backend: $BACKEND"
    echo "Model: $MODEL"
    echo "Profiling tools: $PROFILE_TOOLS"
    echo ""
    echo "Results directory: $RESULT_DIR"
    echo ""
    echo "Files:"
    find "$RESULT_DIR" -type f | sort | while read -r file; do
        echo "  - $(basename "$file")"
    done
    echo ""
    echo "How to view results:"
    if is_tool_enabled "simpleperf"; then
        echo "  - simpleperf: cat $RESULT_DIR/simpleperf/report_functions.txt"
    fi
    if is_tool_enabled "atrace"; then
        echo "  - atrace: Open chrome://tracing, load trace.txt"
    fi
    if is_tool_enabled "perfetto"; then
        echo "  - perfetto: Open https://ui.perfetto.dev, load trace.perfetto-trace"
    fi
    echo "========================================="
}

main "$@"
