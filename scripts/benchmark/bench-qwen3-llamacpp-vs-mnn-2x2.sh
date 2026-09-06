#!/bin/bash
# Qwen3 llama.cpp vs MNN LLM 性能对比 × 两个基准模型 (0.6B / 4B)
# 精度对齐:两框架均强制 Q4 (llama.cpp Q4_K_M ↔ MNN int4), --require-precision q4
# 统一参数: n_prompt=128 / max-tokens=128 / n_repeat=5 (与既有 0.6B 三方对比可比)
set -e

ADB="/mnt/e/andorid/adb/adb.exe"
DEV="/data/local/tmp/benchmark"
TS=$(date +%Y%m%d_%H%M%S)
LOG="/home/liu/project/newwork/benchmark/results/qwen3_llamacpp_vs_mnn_2x2_${TS}.log"

echo "=== llama.cpp vs MNN LLM × Qwen3 0.6B/4B (Q4 对齐) ===" | tee "$LOG"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "设备: $($ADB shell getprop ro.product.model)" | tee -a "$LOG"
echo "平台: $($ADB shell getprop ro.board.platform)" | tee -a "$LOG"
echo "Governor: $($ADB shell 'su -c "cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"')" | tee -a "$LOG"
echo "温度(起): $($ADB shell 'su -c "cat /sys/class/thermal/thermal_zone0/temp"' | tr -d '\r')" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 温度检查（若超 45°C 等待冷却）
check_temp() {
    local t temp
    t=$($ADB shell 'su -c "cat /sys/class/thermal/thermal_zone0/temp"' | tr -d '\r')
    temp=$((t / 1000))
    echo "[温度] ${temp}°C" | tee -a "$LOG"
    if [ "$temp" -gt 45 ]; then
        echo "[警告] 温度 ${temp}°C > 45°C，等待冷却 60s..." | tee -a "$LOG"
        sleep 60
    fi
}

# 每轮后同样检查，避免串场升温
run_bench() {
    local backend=$1 label=$2 model=$3
    echo "" | tee -a "$LOG"
    echo "########## $label ##########" | tee -a "$LOG"
    "$ADB" shell "cd $DEV && LD_LIBRARY_PATH=. ./llm_benchmark --backend $backend --model $model --benchmark --n-prompt 128 --max-tokens 128 --n-repeat 5 --require-precision q4 --json" 2>&1 | tee -a "$LOG"
    check_temp
}

# 顺序：0.6B 先 llama.cpp 后 MNN；4B 同序（对照图同口径）
run_bench llamacpp "llama.cpp  0.6B (Q4_K_M)"  "qwen3_models/Qwen3-0.6B-Q4_K_M.gguf"
run_bench mnn_llm   "MNN LLM    0.6B (int4)"    "qwen3_models/qwen3-0.6b-mnn/config.json"
run_bench llamacpp "llama.cpp  4B   (Q4_K_M)"  "qwen3_models/Qwen3-4B-Q4_K_M.gguf"
run_bench mnn_llm   "MNN LLM    4B   (int4)"    "qwen3_models/qwen3-4b-mnn/config.json"

echo "" | tee -a "$LOG"
echo "=== 2x2 对比完成 ===" | tee -a "$LOG"
echo "日志: $LOG"
