#!/bin/bash
# ============================================================================
# profiler.sh — 统一性能分析工具
#
# 支持: CNN / LLM / SingleOp 三个赛道
# 工具: simpleperf (CPU) / MNN internal / ORT profiling / gperftools
#
# 用法:
#   ./profiler.sh cpu    --backend mnn --model mobilenetv2
#   ./profiler.sh llm    --backend mnn_llm --model qwen2-0.5b --precision fp16
#   ./profiler.sh cycles --backend mnn --model resnet50
#   ./profiler.sh compare --backends mnn,llamacpp --model qwen2-0.5b
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/profiling-utils.sh"

# ── 默认参数 ──
BACKEND="mnn"
MODEL="mobilenetv2"
TRACK="cnn"
THREADS=4
DURATION=10
FREQUENCY=4000
BENCHMARK_RUNS=1000
BUILD_TYPE="debug"
PRECISION="fp16"
N_PROMPT=128
N_GEN=128
N_REPEAT=5
OUTPUT_DIR=""

# ── 帮助 ──
usage() {
    cat << 'EOF'
profiler.sh — 统一性能分析工具

命令:
  cpu         simpleperf CPU 采样 + 火焰图 + 热点分析
  cycles      CPU IPC / cache-miss / branch-miss 微架构分析
  llm         LLM 专项: 逐层耗时 / attention / KV cache
  memory      RSS/VSS 峰值内存采样
  roofline    计算强度 vs 带宽 (Roofline Model)
  compare     两框架横向对比 profile

通用参数:
  --backend <name>      后端名称 (mnn/ort/llamacpp/mnn_llm)
  --model <name>        模型名称 (mobilenetv2/resnet50/qwen2-0.5b/...)
  --track <track>       赛道 (cnn/llm/single_op, 默认: cnn)
  --threads <n>         线程数 (默认: 4)
  --build-type <type>   debug 或 release (默认: debug)
  --output-dir <dir>    输出目录 (默认: results/profiling/<timestamp>)
  --device-dir <dir>    设备路径 (默认: /data/local/tmp/benchmark)

cpu/cycles 参数:
  --duration <sec>      采样时长 (默认: 10s)
  --frequency <hz>      采样频率 Hz (默认: 4000)
  --runs <n>            运行次数 (默认: 1000)

llm 参数:
  --precision <p>       精度 (fp16/fp32/q4, 默认: fp16)
  --n-prompt <n>        prompt token 数 (默认: 128)
  --n-gen <n>           generate token 数 (默认: 128)
  --n-repeat <n>        重复次数 (默认: 5)
EOF
    exit 0
}

# ── 解析参数 ──
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backend) BACKEND="$2"; shift 2 ;;
            --model) MODEL="$2"; shift 2 ;;
            --track) TRACK="$2"; shift 2 ;;
            --threads) THREADS="$2"; shift 2 ;;
            --duration) DURATION="$2"; shift 2 ;;
            --frequency) FREQUENCY="$2"; shift 2 ;;
            --runs) BENCHMARK_RUNS="$2"; shift 2 ;;
            --build-type) BUILD_TYPE="$2"; shift 2 ;;
            --precision) PRECISION="$2"; shift 2 ;;
            --n-prompt) N_PROMPT="$2"; shift 2 ;;
            --n-gen) N_GEN="$2"; shift 2 ;;
            --n-repeat) N_REPEAT="$2"; shift 2 ;;
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            --device-dir) DEVICE_DIR="$2"; shift 2 ;;
            --help|-h) usage ;;
            *) log_error "Unknown option: $1"; exit 1 ;;
        esac
    done

    DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/benchmark}"
}

# ── 选择正确的二进制 ──
select_binary() {
    case "$TRACK" in
        cnn|single_op)
            echo "benchmark_inference" ;;
        llm)
            echo "llm_benchmark" ;;
        *) log_error "Unknown track: $TRACK"; exit 1 ;;
    esac
}

# ── 构建 benchmark 命令行 ──
build_benchmark_cmd() {
    local binary="$1"
    local cmd="$DEVICE_DIR/$binary"

    case "$TRACK" in
        cnn)
            cmd="$cmd --backend $BACKEND --model $MODEL --threads $THREADS --warmup 10 --runs $BENCHMARK_RUNS"
            ;;
        llm)
            local llm_backend="$BACKEND"
            local model_path="$MODEL"
            case "$BACKEND" in
                llamacpp)
                    case "$PRECISION" in
                        fp16) model_path="qwen2-0_5b-instruct-fp16.gguf" ;;
                        q4|q4_k_m) model_path="qwen2-0_5b-instruct-q4_k_m.gguf" ;;
                    esac
                    ;;
                mnn_llm|mnn)
                    llm_backend="mnn_llm"
                    model_path="qwen2-0.5b-mnn/config.json"
                    ;;
            esac
            cmd="$cmd --backend $llm_backend --model $model_path --n-prompt $N_PROMPT --max-tokens $N_GEN --n-repeat $N_REPEAT --benchmark"
            ;;
    esac

    echo "$cmd > $DEVICE_DIR/profiling/benchmark_output.txt 2>&1"
}

# ═══════════════════════════════════════════════════════════════
#  命令实现
# ═══════════════════════════════════════════════════════════════

# ── simpleperf CPU 采样 ──
cmd_cpu() {
    local binary=$(select_binary)
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    OUTPUT_DIR="${OUTPUT_DIR:-results/profiling/${timestamp}_${BACKEND}_${MODEL}_cpu}"
    mkdir -p "$OUTPUT_DIR/simpleperf"

    detect_adb || exit 1
    check_device || exit 1
    check_root

    log_info "=== CPU Profiling: $BACKEND / $MODEL / $TRACK ==="
    log_info "Duration: ${DURATION}s, Frequency: ${FREQUENCY}Hz"

    # 推 simpleperf
    push_simpleperf

    # 保存设备信息
    get_device_info "$OUTPUT_DIR/device_info.txt"

    # 构建 benchmark 命令
    local bench_cmd=$(build_benchmark_cmd "$binary")

    # 在设备上执行: 启动 benchmark → simpleperf 采样
    adb_cmd shell "mkdir -p $DEVICE_DIR/profiling"
    local profile_script="$DEVICE_DIR/profiling/run_profile.sh"
    cat << SCRIPT_EOF | adb_cmd shell "cat > $profile_script"
#!/bin/sh
cd $DEVICE_DIR
LD_LIBRARY_PATH=. $bench_cmd &
BENCH_PID=\$!
echo \$BENCH_PID > profiling/benchmark.pid
sleep 2
$DEVICE_DIR/../simpleperf record \
    -p \$BENCH_PID \
    -e cpu-cycles \
    -f $FREQUENCY \
    --call-graph dwarf \
    -o profiling/perf.data \
    --duration $DURATION
wait \$BENCH_PID
SCRIPT_EOF
    adb_cmd shell "chmod 755 $profile_script"

    log_info "Running benchmark with simpleperf..."
    adb_cmd shell "sh $profile_script"

    # 拉取结果
    adb_cmd pull "$DEVICE_DIR/profiling/perf.data" "$OUTPUT_DIR/simpleperf/"
    adb_cmd pull "$DEVICE_DIR/profiling/benchmark_output.txt" "$OUTPUT_DIR/"

    # 生成报告
    generate_cpu_report

    log_info "CPU profiling complete → $OUTPUT_DIR"
}

# ── CPU 微架构分析 (IPC, cache, branch) ──
cmd_cycles() {
    local binary=$(select_binary)
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    OUTPUT_DIR="${OUTPUT_DIR:-results/profiling/${timestamp}_${BACKEND}_${MODEL}_cycles}"
    mkdir -p "$OUTPUT_DIR"

    detect_adb || exit 1
    check_device || exit 1
    check_root

    log_info "=== Microarchitecture Analysis: $BACKEND / $MODEL ==="

    push_simpleperf
    get_device_info "$OUTPUT_DIR/device_info.txt"

    local bench_cmd=$(build_benchmark_cmd "$binary")
    local events=("cpu-cycles" "instructions" "cache-misses" "branch-misses" "cpu-cycles:HG")

    for event in "${events[@]}"; do
        local safe_name=$(echo "$event" | tr ':-' '_')
        log_info "  Sampling event: $event"

        local profile_script="$DEVICE_DIR/profiling/run_${safe_name}.sh"
        cat << SCRIPT_EOF | adb_cmd shell "cat > $profile_script"
#!/bin/sh
cd $DEVICE_DIR
LD_LIBRARY_PATH=. $bench_cmd &
BENCH_PID=\$!
sleep 2
$DEVICE_DIR/../simpleperf record -p \$BENCH_PID -e $event -f $FREQUENCY -o profiling/perf_${safe_name}.data --duration $DURATION
wait \$BENCH_PID
SCRIPT_EOF
        adb_cmd shell "chmod 755 $profile_script"
        adb_cmd shell "sh $profile_script"
        adb_cmd pull "$DEVICE_DIR/profiling/perf_${safe_name}.data" "$OUTPUT_DIR/" 2>/dev/null || true
        sleep 2  # 冷却
    done

    # 计算 IPC
    generate_uarch_report "$OUTPUT_DIR"

    log_info "Microarchitecture analysis complete → $OUTPUT_DIR"
}

# LLM 专项分析（MNN 内置 profiler + simpleperf 叠加）
cmd_llm() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    OUTPUT_DIR="${OUTPUT_DIR:-results/profiling/${timestamp}_${BACKEND}_${MODEL}_llm}"
    mkdir -p "$OUTPUT_DIR"

    detect_adb || exit 1
    check_device || exit 1
    check_root

    log_info "=== LLM Profiling: $BACKEND / $MODEL / $PRECISION ==="

    # 1. 先跑一轮不带 profiler 的 benchmark 获取基线
    log_info "Step 1: Baseline benchmark..."
    local bench_cmd=$(build_benchmark_cmd "llm_benchmark")
    adb_cmd shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=. $bench_cmd" > "$OUTPUT_DIR/baseline.txt" 2>&1
    grep -E "prefill:|decode:|Peak|temp" "$OUTPUT_DIR/baseline.txt" | head -10

    # 2. CPU 采样（仅采样 decode 阶段）
    log_info "Step 2: CPU sampling during decode..."
    push_simpleperf

    local profile_script="$DEVICE_DIR/profiling/run_llm_profile.sh"
    cat << SCRIPT_EOF | adb_cmd shell "cat > $profile_script"
#!/bin/sh
cd $DEVICE_DIR
LD_LIBRARY_PATH=. $bench_cmd &
BENCH_PID=\$!
sleep 2
$DEVICE_DIR/../simpleperf record -p \$BENCH_PID -e cpu-cycles -f 4000 --call-graph dwarf -o profiling/perf_llm.data --duration 30
wait \$BENCH_PID
SCRIPT_EOF
    adb_cmd shell "chmod 755 $profile_script"
    adb_cmd shell "sh $profile_script"

    adb_cmd pull "$DEVICE_DIR/profiling/perf_llm.data" "$OUTPUT_DIR/" 2>/dev/null || true

    # 3. 生成火焰图
    if [ -f "$OUTPUT_DIR/perf_llm.data" ]; then
        generate_flamegraph "$OUTPUT_DIR/perf_llm.data" "$OUTPUT_DIR/flamegraph.svg"
    fi

    # 4. LLM 专项报告
    generate_llm_report

    log_info "LLM profiling complete → $OUTPUT_DIR"
}

# ── 两框架横向对比 profiler ──
cmd_compare() {
    IFS=',' read -ra backends <<< "$1"
    log_info "=== Cross-framework Profiling: ${backends[*]} / $MODEL ==="

    for b in "${backends[@]}"; do
        log_info "--- $b ---"
        BACKEND="$b"
        cmd_cpu
    done
}

# ═══════════════════════════════════════════════════════════════
#  报告生成
# ═══════════════════════════════════════════════════════════════

generate_cpu_report() {
    local perf_data="$OUTPUT_DIR/simpleperf/perf.data"
    [ ! -f "$perf_data" ] && { log_warn "No perf.data"; return; }

    # 热点函数 Top 20
    adb_cmd shell "/data/local/tmp/simpleperf report -i $DEVICE_DIR/profiling/perf.data --sort dso,symbol -n" \
        > "$OUTPUT_DIR/simpleperf/hotspots.txt" 2>/dev/null

    # DSO 分布
    adb_cmd shell "/data/local/tmp/simpleperf report -i $DEVICE_DIR/profiling/perf.data --sort dso -n" \
        > "$OUTPUT_DIR/simpleperf/dso.txt" 2>/dev/null

    # 火焰图
    generate_flamegraph "$perf_data" "$OUTPUT_DIR/simpleperf/flamegraph.svg"

    # Print summary
    echo ""
    echo "=== CPU Profile Summary ==="
    echo "Top 10 Hotspots:"
    head -25 "$OUTPUT_DIR/simpleperf/hotspots.txt" 2>/dev/null || echo "  (not available)"
}

generate_uarch_report() {
    local dir="$1"
    cat > "$dir/uarch_report.md" << EOF
# Microarchitecture Report: $BACKEND / $MODEL

| Metric | Value | Source |
|--------|-------|--------|
EOF

    # IPC = instructions / cycles
    if [ -f "$dir/perf_cpu_cycles.data" ] && [ -f "$dir/perf_instructions.data" ]; then
        # 从 simpleperf stat 提取
        echo "IPC 计算需要 simpleperf stat 支持" >> "$dir/uarch_report.md"
    fi

    log_info "uArch report: $dir/uarch_report.md"
}

generate_llm_report() {
    cat > "$OUTPUT_DIR/llm_report.md" << EOF
# LLM Profiling Report: $BACKEND / $MODEL

## Configuration
- **Precision**: $PRECISION
- **Threads**: $THREADS
- **Prompt**: $N_PROMPT tokens
- **Generation**: $N_GEN tokens
- **Repeats**: $N_REPEAT

## Baseline Performance
EOF
    grep -E "Mean|P50|P90|P99|Peak|temp" "$OUTPUT_DIR/baseline.txt" >> "$OUTPUT_DIR/llm_report.md" 2>/dev/null || true
    echo "" >> "$OUTPUT_DIR/llm_report.md"
    echo "## Flame Graph" >> "$OUTPUT_DIR/llm_report.md"
    echo "See: [flamegraph.svg](flamegraph.svg)" >> "$OUTPUT_DIR/llm_report.md"

    log_info "LLM report: $OUTPUT_DIR/llm_report.md"
}

generate_flamegraph() {
    local perf_data="$1"
    local svg_file="${2:-$OUTPUT_DIR/simpleperf/flamegraph.svg}"
    local folded_file="${svg_file%.svg}.folded"

    local flamegraph_pl="$PROJECT_ROOT/tools/FlameGraph/flamegraph.pl"
    local stackcollapse="$PROJECT_ROOT/tools/FlameGraph/stackcollapse-perf.pl"

    if [ ! -f "$flamegraph_pl" ]; then
        log_warn "FlameGraph tools not found at $flamegraph_pl"
        return
    fi

    # report-sample → folded → SVG
    adb_cmd shell "/data/local/tmp/simpleperf report-sample \
        -i $DEVICE_DIR/profiling/perf.data --show-callchain" 2>/dev/null | \
        tr -d '\r' | \
        awk '
        /^sample:/ { if (stack != "") { n=split(stack,arr,";"); r=arr[n]; for(i=n-1;i>=1;i--) r=r";"arr[i]; print r" 1"; stack="" } next }
        /symbol:/ { sym=substr($0,index($0,":")+2); gsub(/;/, ":", sym); stack=(stack==""?sym:stack";"sym) }
        END { if (stack != "") { n=split(stack,arr,";"); r=arr[n]; for(i=n-1;i>=1;i--) r=r";"arr[i]; print r" 1" } }' \
        > "$folded_file" 2>/dev/null

    if [ -s "$folded_file" ]; then
        "$flamegraph_pl" "$folded_file" > "$svg_file" 2>/dev/null
        log_info "Flamegraph: $svg_file"
    fi
}

# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

main() {
    [ $# -lt 1 ] && usage
    local cmd="$1"; shift
    parse_args "$@"

    case "$cmd" in
        cpu)      cmd_cpu ;;
        cycles)   cmd_cycles ;;
        llm)      cmd_llm ;;
        compare)  cmd_compare "$BACKEND" ;;
        *)        usage ;;
    esac
}

main "$@"
