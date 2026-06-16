#!/bin/bash
# perfetto_trace.sh - 使用 perfetto 采集综合 trace
# 输出可用 https://ui.perfetto.dev 打开的 trace 文件

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profiling-utils.sh"

# 默认参数
BACKEND="mnn"
MODEL="mobilenetv2"
THREADS=4
DURATION=5
BUFFER_SIZE=32768
PRESET="cpu"
BENCHMARK_RUNS=1000

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --buffer-size) BUFFER_SIZE="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --runs) BENCHMARK_RUNS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --backend <name>      Backend to test (default: mnn)"
            echo "  --model <name>        Model to test (default: mobilenetv2)"
            echo "  --threads <num>       Number of threads (default: 4)"
            echo "  --duration <sec>      Trace duration in seconds (default: 5)"
            echo "  --buffer-size <kb>    Buffer size in KB (default: 32768)"
            echo "  --preset <preset>     Preset config: cpu/memory/full (default: cpu)"
            echo "  --runs <num>          Benchmark runs (default: 1000)"
            echo ""
            echo "Presets:"
            echo "  cpu     - CPU scheduling, frequency, process info"
            echo "  memory  - Memory allocation, LMK, RSS"
            echo "  full    - CPU + memory + GPU + binder (high overhead)"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# 生成 perfetto 配置
generate_perfetto_config() {
    local config_file="$1"
    local duration_ms=$((DURATION * 1000))

    cat > "$config_file" << EOF
buffers: {
    size_kb: $BUFFER_SIZE
    fill_policy: RING_BUFFER
}

duration_ms: $duration_ms

EOF

    case "$PRESET" in
        cpu)
            cat >> "$config_file" << EOF
data_sources: {
    config {
        name: "linux.ftrace"
        ftrace_config {
            ftrace_events: "sched/sched_switch"
            ftrace_events: "sched/sched_wakeup"
            ftrace_events: "sched/sched_wakeup_new"
            ftrace_events: "sched/sched_process_exit"
            ftrace_events: "sched/sched_process_free"
            ftrace_events: "power/cpu_frequency"
            ftrace_events: "power/cpu_idle"
            ftrace_events: "power/suspend_resume"
            atrace_categories: "sched"
            atrace_categories: "freq"
            atrace_categories: "idle"
            atrace_apps: "*"
        }
    }
}

data_sources: {
    config {
        name: "linux.process_stats"
        process_stats_config {
            scan_all_processes_on_start: true
            proc_stats_poll_ms: 1000
        }
    }
}
EOF
            ;;
        memory)
            cat >> "$config_file" << EOF
data_sources: {
    config {
        name: "linux.ftrace"
        ftrace_config {
            ftrace_events: "kmem/rss_stat"
            ftrace_events: "kmem/ion_heap_grow"
            ftrace_events: "kmem/ion_heap_shrink"
            ftrace_events: "lowmemorykiller/lowmemory_kill"
            ftrace_events: "oom/oom_score_adj_update"
            atrace_categories: "view"
            atrace_categories: "am"
            atrace_apps: "*"
        }
    }
}

data_sources: {
    config {
        name: "linux.process_stats"
        process_stats_config {
            scan_all_processes_on_start: true
            proc_stats_poll_ms: 500
        }
    }
}
EOF
            ;;
        full)
            cat >> "$config_file" << EOF
data_sources: {
    config {
        name: "linux.ftrace"
        ftrace_config {
            ftrace_events: "sched/sched_switch"
            ftrace_events: "sched/sched_wakeup"
            ftrace_events: "power/cpu_frequency"
            ftrace_events: "power/cpu_idle"
            ftrace_events: "kmem/rss_stat"
            ftrace_events: "lowmemorykiller/lowmemory_kill"
            ftrace_events: "binder/binder_transaction"
            ftrace_events: "binder/binder_transaction_received"
            atrace_categories: "sched"
            atrace_categories: "freq"
            atrace_categories: "idle"
            atrace_categories: "gfx"
            atrace_categories: "view"
            atrace_categories: "wm"
            atrace_categories: "am"
            atrace_categories: "binder_driver"
            atrace_apps: "*"
        }
    }
}

data_sources: {
    config {
        name: "linux.process_stats"
        process_stats_config {
            scan_all_processes_on_start: true
            proc_stats_poll_ms: 500
        }
    }
}

data_sources: {
    config {
        name: "linux.sys_stats"
        sys_stats_config {
            cpufreq_period_ms: 500
            meminfo_period_ms: 1000
            vmstat_period_ms: 1000
        }
    }
}
EOF
            ;;
        *)
            log_error "Unknown preset: $PRESET"
            exit 1
            ;;
    esac
}

# 主流程
main() {
    log_info "=== Perfetto Trace Capture ==="
    log_info "Backend: $BACKEND, Model: $MODEL, Threads: $THREADS"
    log_info "Duration: ${DURATION}s, Preset: $PRESET"

    # Step 1: 初始化
    detect_adb || exit 1
    check_device || exit 1

    # Step 2: 创建结果目录
    create_result_dir "$MODEL" "$BACKEND"

    # Step 3: 推送文件
    push_benchmark_files

    # Step 4: 保存设备信息
    get_device_info "$RESULT_DIR/device_info.txt"

    # Step 5: 生成 perfetto 配置
    local config_file="$RESULT_DIR/perfetto/config.pbtx"
    generate_perfetto_config "$config_file"
    log_info "Perfetto config generated: $config_file"

    # 推送配置到设备
    adb_cmd push "$config_file" "/data/local/tmp/benchmark/profiling/config.pbtx"
    adb_cmd shell "chmod 644 /data/local/tmp/benchmark/profiling/config.pbtx"

    # Step 6: 启动 benchmark（使用包装脚本）
    log_info "Starting benchmark with perfetto profiling..."
    local device_dir="/data/local/tmp/benchmark"
    adb_cmd shell "mkdir -p $device_dir/profiling"

    adb_cmd shell "cat > $device_dir/profiling/run_with_perfetto.sh << 'SCRIPT_EOF'
#!/bin/sh
cd /data/local/tmp/benchmark
LD_LIBRARY_PATH=. ./benchmark_inference \\
    --backend $BACKEND \\
    --model $MODEL \\
    --threads $THREADS \\
    --warmup 10 \\
    --runs $BENCHMARK_RUNS > profiling/benchmark_output.txt 2>&1 &
BENCH_PID=\$!

# 等待 warmup
sleep 2

# 启动 perfetto（使用 stdin 传递配置，输出到 perfetto-traces 目录）
cat /data/local/tmp/benchmark/profiling/config.pbtx | perfetto --txt -c - -o /data/misc/perfetto-traces/benchmark_trace.perfetto-trace

# 复制结果到 benchmark 目录
cp /data/misc/perfetto-traces/benchmark_trace.perfetto-trace /data/local/tmp/benchmark/profiling/trace.perfetto-trace
chmod 644 /data/local/tmp/benchmark/profiling/trace.perfetto-trace

# 等待 benchmark 完成
wait \$BENCH_PID
SCRIPT_EOF
chmod 755 $device_dir/profiling/run_with_perfetto.sh"

    # Step 7: 执行
    log_info "Running perfetto capture (${DURATION}s)..."
    adb_cmd shell "sh $device_dir/profiling/run_with_perfetto.sh"

    # Step 8: 等待 benchmark 完成
    log_info "Waiting for benchmark to complete..."
    while adb_cmd shell "pidof benchmark_inference" &>/dev/null; do
        sleep 1
    done

    # Step 9: 拉取数据
    log_info "Pulling results..."
    adb_cmd pull "$device_dir/profiling/trace.perfetto-trace" "$RESULT_DIR/perfetto/"
    adb_cmd pull "$device_dir/profiling/benchmark_output.txt" "$RESULT_DIR/"

    # Step 10: 输出摘要
    print_summary

    log_info "=== Trace Capture Complete ==="
    log_info "Results saved to: $RESULT_DIR"
}

print_summary() {
    echo ""
    echo "========================================="
    echo "  Perfetto Trace Summary"
    echo "========================================="
    echo "Backend: $BACKEND"
    echo "Model: $MODEL"
    echo "Duration: ${DURATION}s"
    echo "Preset: $PRESET"
    echo ""
    echo "Output files:"
    echo "  - trace.perfetto-trace: $RESULT_DIR/perfetto/trace.perfetto-trace"
    echo "  - config.pbtx: $RESULT_DIR/perfetto/config.pbtx"
    echo ""
    echo "How to view:"
    echo "  1. Open https://ui.perfetto.dev"
    echo "  2. Click 'Open trace file'"
    echo "  3. Select trace.perfetto-trace"
    echo "========================================="
}

main "$@"
