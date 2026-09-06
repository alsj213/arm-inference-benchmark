#!/bin/bash
# Qwen3-0.6B-Q4_K_M 三方对比：MobileLLM vs llama.cpp vs MNN LLM
# 统一 llm_benchmark 二进制 + 统一参数 (n_prompt=128 / n_gen=128 / repeat=5)
set -eo pipefail

ADB="/mnt/e/andorid/adb/adb.exe"
DEV="/data/local/tmp/benchmark"
TS=$(date +%Y%m%d_%H%M%S)
LOG="/home/liu/project/newwork/benchmark/results/qwen3_06b_3way_${TS}.log"
GGUF="qwen3_models/Qwen3-0.6B-Q4_K_M.gguf"
MNN="qwen3_models/qwen3-0.6b-mnn/config.json"

echo "=== Qwen3-0.6B-Q4_K_M 三方性能对比 ===" | tee "$LOG"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "设备: $($ADB shell getprop ro.product.model)" | tee -a "$LOG"
echo "平台: $($ADB shell getprop ro.board.platform)" | tee -a "$LOG"
echo "Governor: $($ADB shell 'su -c "cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"')" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 温度检查（若超 45°C 等待冷却）
check_temp() {
    local t
    t=$($ADB shell 'su -c "cat /sys/class/thermal/thermal_zone0/temp"' | tr -d '\r')
    local temp=$((t / 1000))
    echo "[温度] $temp°C" | tee -a "$LOG"
    if [ "$temp" -gt 45 ]; then
        echo "[警告] 温度 $temp°C > 45°C，等待冷却 60s..." | tee -a "$LOG"
        sleep 60
    fi
}

run_bench() {
    local backend=$1 label=$2 model=$3
    echo "" | tee -a "$LOG"
    echo "########## $label ##########" | tee -a "$LOG"
    # --require-precision q4：三方必须真实为 Q4，任一不匹配（如 MNN 实际 int8）则失败
    # pipefail + 显式 rc:设备端非零退出(精度不匹配/崩溃/路径缺失)不被 tee 吞掉
    local rc=0
    "$ADB" shell "cd $DEV && LD_LIBRARY_PATH=. ./llm_benchmark --backend $backend --model $model --benchmark --n-prompt 128 --max-tokens 128 --n-repeat 5 --require-precision q4 --json" 2>&1 | tee -a "$LOG" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[ERROR] $label 运行失败 (exit=$rc)" | tee -a "$LOG"
        exit 1
    fi
    check_temp
}

run_bench mobilellm "MobileLLM (Q4_K_M)" "$GGUF"
run_bench llamacpp  "llama.cpp  (Q4_K_M)" "$GGUF"
run_bench mnn_llm   "MNN LLM   (Q4)"       "$MNN"

echo "" | tee -a "$LOG"
echo "=== 三方对比完成 ===" | tee -a "$LOG"
echo "日志: $LOG"
