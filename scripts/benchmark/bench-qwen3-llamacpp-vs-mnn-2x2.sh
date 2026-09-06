#!/bin/bash
# Qwen3 llama.cpp vs MNN LLM 性能对比 × 两个基准模型 (0.6B / 4B)
# 精度对齐:两框架均强制 Q4 (llama.cpp Q4_K_M ↔ MNN int4), --require-precision q4
# 统一参数: n_prompt=128 / max-tokens=128 / n_repeat=5 (与既有 0.6B 三方对比可比)
# 配置从 .benchmarkrc.yml 读取(项目约定:不硬编码设备路径);产物/模型运行前校验
set -eo pipefail

# ── 配置(从 .benchmarkrc.yml,项目单点约定) ──
CFG_GET() { python3 -c "import yaml,sys; print(yaml.safe_load(open('.benchmarkrc.yml'))$1)"; }
ADB=$(CFG_GET "['device']['adb']")
DEV_ID=$(CFG_GET "['device']['id']")
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)   # scripts/benchmark/ → 仓根
DEV="/data/local/tmp/benchmark"                    # 设备侧固定测试目录
TS=$(date +%Y%m%d_%H%M%S)
LOG="$REPO_ROOT/results/qwen3_llamacpp_vs_mnn_2x2_${TS}.log"

echo "=== llama.cpp vs MNN LLM × Qwen3 0.6B/4B (Q4 对齐) ===" | tee "$LOG"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "设备: $($ADB shell getprop ro.product.model)" | tee -a "$LOG"
echo "平台: $($ADB shell getprop ro.board.platform)" | tee -a "$LOG"
echo "Governor: $($ADB shell 'su -c "cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"')" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── 温度门禁(M4 修复:while 循环冷却到阈值,非单次休眠;超时中止) ──
check_temp() {
    local t temp tries=0
    t=$($ADB shell 'su -c "cat /sys/class/thermal/thermal_zone0/temp"' | tr -d '\r')
    temp=$((t / 1000))
    echo "[温度] ${temp}°C" | tee -a "$LOG"
    while [ "$temp" -gt 45 ] && [ "$tries" -lt 10 ]; do
        echo "[警告] 温度 ${temp}°C > 45°C,冷却 60s (第 $((tries + 1))/10 次)..." | tee -a "$LOG"
        sleep 60
        t=$($ADB shell 'su -c "cat /sys/class/thermal/thermal_zone0/temp"' | tr -d '\r')
        temp=$((t / 1000))
        tries=$((tries + 1))
    done
    if [ "$temp" -gt 45 ]; then
        echo "[ERROR] 冷却 10 次仍 ${temp}°C > 45°C,中止(降频风险)" | tee -a "$LOG"
        exit 1
    fi
}

# ── 前置校验(M8 修复:产物/模型存在性,缺则报错退出而非静默) ──
ensure_prereq() {
    local model=$1
    "$ADB" shell "test -x $DEV/llm_benchmark" || { echo "[ERROR] 设备缺 $DEV/llm_benchmark(先 build+push)" | tee -a "$LOG"; exit 1; }
    "$ADB" shell "test -e $DEV/$model"          || { echo "[ERROR] 设备缺模型 $DEV/$model(先 push)" | tee -a "$LOG"; exit 1; }
}

# ── 单后端跑:pipefail + 显式 rc,失败即退出(数据真实性红线) ──
run_bench() {
    local backend=$1 label=$2 model=$3 rc=0
    ensure_prereq "$model"
    check_temp                       # 首轮前与每轮前门禁
    echo "" | tee -a "$LOG"
    echo "########## $label ##########" | tee -a "$LOG"
    "$ADB" shell "cd $DEV && LD_LIBRARY_PATH=. ./llm_benchmark --backend $backend --model $model --benchmark --n-prompt 128 --max-tokens 128 --n-repeat 5 --require-precision q4 --json" 2>&1 | tee -a "$LOG" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[ERROR] $label 运行失败 (exit=$rc)" | tee -a "$LOG"
        exit 1
    fi
}

# 顺序：0.6B 先 llama.cpp 后 MNN；4B 同序（对照图同口径）
run_bench llamacpp "llama.cpp  0.6B (Q4_K_M)"  "qwen3_models/Qwen3-0.6B-Q4_K_M.gguf"
run_bench mnn_llm   "MNN LLM    0.6B (int4)"    "qwen3_models/qwen3-0.6b-mnn/config.json"
run_bench llamacpp "llama.cpp  4B   (Q4_K_M)"  "qwen3_models/Qwen3-4B-Q4_K_M.gguf"
run_bench mnn_llm   "MNN LLM    4B   (int4)"    "qwen3_models/qwen3-4b-mnn/config.json"

echo "" | tee -a "$LOG"
echo "=== 2x2 对比完成 ===" | tee -a "$LOG"
echo "日志: $LOG"
