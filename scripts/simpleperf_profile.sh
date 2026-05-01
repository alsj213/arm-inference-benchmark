#!/bin/bash
# simpleperf_profile.sh - 使用 simpleperf 进行 CPU 采样分析
# 生成火焰图和函数热点报告

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profiling_utils.sh"

# 默认参数
BACKEND="mnn"
MODEL="mobilenetv2"
THREADS=4
DURATION=10
FREQUENCY=4000
EVENT="cpu-cycles"
CALLGRAPH="dwarf"
BENCHMARK_RUNS=1000
BUILD_TYPE="debug"  # 火焰图分析默认使用 Debug 版本（包含调试符号）

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --frequency) FREQUENCY="$2"; shift 2 ;;
        --event) EVENT="$2"; shift 2 ;;
        --callgraph) CALLGRAPH="$2"; shift 2 ;;
        --runs) BENCHMARK_RUNS="$2"; shift 2 ;;
        --build-type) BUILD_TYPE="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --backend <name>      Backend to test (default: mnn)"
            echo "  --model <name>        Model to test (default: mobilenetv2)"
            echo "  --threads <num>       Number of threads (default: 4)"
            echo "  --duration <sec>      Sampling duration in seconds (default: 10)"
            echo "  --frequency <hz>      Sampling frequency in Hz (default: 4000)"
            echo "  --event <event>       Sampling event (default: cpu-cycles)"
            echo "  --callgraph <mode>    Call graph mode: fp or dwarf (default: dwarf)"
            echo "  --runs <num>          Benchmark runs (default: 1000)"
            echo "  --build-type <type>   Build type: debug or release (default: debug)"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# 主流程
main() {
    log_info "=== Simpleperf CPU Profiling ==="
    log_info "Backend: $BACKEND, Model: $MODEL, Threads: $THREADS"
    log_info "Duration: ${DURATION}s, Frequency: ${FREQUENCY}Hz, Event: $EVENT"

    # Step 1: 初始化
    detect_adb || exit 1
    check_device || exit 1
    check_root

    # Step 2: 创建结果目录
    create_result_dir "$MODEL" "$BACKEND"

    # Step 3: 推送文件
    push_benchmark_files "" "" "$BUILD_TYPE"
    push_simpleperf

    # Step 4: 保存设备信息
    get_device_info "$RESULT_DIR/device_info.txt"

    # Step 5: 在设备上创建包装脚本并执行
    local device_dir="/data/local/tmp/benchmark"
    adb_cmd shell "mkdir -p $device_dir/profiling"

    log_info "Creating profiling wrapper script on device..."
    adb_cmd shell "cat > $device_dir/profiling/run_with_profile.sh << 'SCRIPT_EOF'
#!/bin/sh
cd /data/local/tmp/benchmark
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

# 启动 simpleperf（使用 dwarf 模式展开调用栈，穿透汇编函数）
/data/local/tmp/simpleperf record \\
    -p \$BENCH_PID \\
    -e $EVENT \\
    -f $FREQUENCY \\
    --call-graph $CALLGRAPH \\
    --duration $DURATION \\
    -o profiling/perf.data

# 等待 benchmark 完成
wait \$BENCH_PID
SCRIPT_EOF
chmod 755 $device_dir/profiling/run_with_profile.sh"

    # 执行包装脚本
    log_info "Running benchmark with simpleperf profiling (${DURATION}s sampling)..."
    adb_cmd shell "sh $device_dir/profiling/run_with_profile.sh"

    # Step 6: 拉取数据
    log_info "Pulling results..."
    adb_cmd pull "$device_dir/profiling/benchmark_output.txt" "$RESULT_DIR/"
    adb_cmd pull "$device_dir/profiling/perf.data" "$RESULT_DIR/simpleperf/"

    # Step 7: 生成报告
    generate_reports

    # Step 8: 输出摘要
    print_summary

    log_info "=== Profiling Complete ==="
    log_info "Results saved to: $RESULT_DIR"
}

# 生成报告
generate_reports() {
    log_info "Generating reports..."
    local perf_data="$RESULT_DIR/simpleperf/perf.data"

    if [ ! -f "$perf_data" ]; then
        log_warn "perf.data not found, skipping report generation"
        return
    fi

    # 在设备上生成报告（使用设备上的 simpleperf）
    local device_dir="/data/local/tmp/benchmark"
    log_info "Generating reports on device..."

    # 函数热点报告
    adb_cmd shell "/data/local/tmp/simpleperf report \
        -i $device_dir/profiling/perf.data \
        --sort dso,symbol \
        -n" > "$RESULT_DIR/simpleperf/report_functions.txt" 2>/dev/null

    # DSO 级报告
    adb_cmd shell "/data/local/tmp/simpleperf report \
        -i $device_dir/profiling/perf.data \
        --sort dso \
        -n" > "$RESULT_DIR/simpleperf/report_dso.txt" 2>/dev/null

    # 调用链报告
    adb_cmd shell "/data/local/tmp/simpleperf report \
        -i $device_dir/profiling/perf.data \
        --sort comm,dso,symbol \
        --show-callchain" > "$RESULT_DIR/simpleperf/report_callchain.txt" 2>/dev/null

    # 生成火焰图
    generate_flamegraph "$perf_data"

    log_info "Reports generated"
}

# 生成火焰图
generate_flamegraph() {
    local perf_data="$1"
    local folded_file="$RESULT_DIR/simpleperf/out.folded"
    local svg_file="$RESULT_DIR/simpleperf/flamegraph.svg"

    # 检查 FlameGraph 工具
    local flamegraph_pl="./tools/FlameGraph/flamegraph.pl"
    if [ ! -f "$flamegraph_pl" ]; then
        log_warn "FlameGraph not found. Clone it with:"
        log_warn "  git clone https://github.com/brendangregg/FlameGraph.git tools/FlameGraph"
        return
    fi

    log_info "Generating flamegraph from report-sample..."
    local device_dir="/data/local/tmp/benchmark"

    # 使用 report-sample 生成完整的调用栈，然后转换为折叠格式
    # 注意：需要去掉 \r、处理函数名中的特殊字符、反转调用栈顺序
    adb_cmd shell "/data/local/tmp/simpleperf report-sample \
        -i $device_dir/profiling/perf.data \
        --show-callchain" | \
        tr -d '\r' | \
        awk '
        /^sample:/ {
            if (stack != "") {
                # 反转调用栈顺序（从根到叶）
                n = split(stack, arr, ";")
                reversed = arr[n]
                for (i = n-1; i >= 1; i--) {
                    reversed = reversed ";" arr[i]
                }
                print reversed " 1"
                stack = ""
            }
            next
        }
        /symbol:/ {
            # 提取符号名（去掉 "symbol: " 前缀）
            sym = substr($0, index($0, ":") + 2)
            # 将符号名中的分号替换为冒号，避免和调用栈分隔符冲突
            gsub(/;/, ":", sym)
            if (stack == "") {
                stack = sym
            } else {
                stack = stack ";" sym
            }
        }
        END {
            if (stack != "") {
                # 反转调用栈顺序（从根到叶）
                n = split(stack, arr, ";")
                reversed = arr[n]
                for (i = n-1; i >= 1; i--) {
                    reversed = reversed ";" arr[i]
                }
                print reversed " 1"
            }
        }' > "$folded_file" 2>/dev/null

    # 生成 SVG
    if [ -s "$folded_file" ]; then
        "$flamegraph_pl" "$folded_file" > "$svg_file" 2>/dev/null

        if [ -f "$svg_file" ]; then
            log_info "Flamegraph generated: $svg_file"
        else
            log_warn "Failed to generate flamegraph"
        fi
    else
        log_warn "No folded data generated, skipping flamegraph"
    fi
}

# 打印摘要
print_summary() {
    echo ""
    echo "========================================="
    echo "  Simpleperf Profiling Summary"
    echo "========================================="
    echo "Backend: $BACKEND"
    echo "Model: $MODEL"
    echo "Duration: ${DURATION}s"
    echo ""

    # Top 10 热点函数
    echo "--- Top 10 Hot Functions ---"
    if [ -f "$RESULT_DIR/simpleperf/report_functions.txt" ]; then
        grep -A 100 "Event:" "$RESULT_DIR/simpleperf/report_functions.txt" | \
            grep -B 100 "^\s*$" | \
            head -20
    else
        echo "  (report not generated)"
    fi

    echo ""
    echo "--- Top 5 DSO (Shared Libraries) ---"
    if [ -f "$RESULT_DIR/simpleperf/report_dso.txt" ]; then
        grep -A 100 "Event:" "$RESULT_DIR/simpleperf/report_dso.txt" | \
            grep -B 100 "^\s*$" | \
            head -10
    else
        echo "  (report not generated)"
    fi

    echo ""
    echo "Files:"
    echo "  - perf.data: $RESULT_DIR/simpleperf/perf.data"
    echo "  - report_functions.txt: $RESULT_DIR/simpleperf/report_functions.txt"
    echo "  - report_dso.txt: $RESULT_DIR/simpleperf/report_dso.txt"
    echo "========================================="
}

main "$@"
