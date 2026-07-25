#!/bin/bash
# Qwen3-4B Benchmark: MNN LLM vs llama.cpp
# 用法: ./scripts/benchmark/bench-qwen3-4b.sh
set -e

ADB="${ADB:-adb}"
DEVICE_DIR="/data/local/tmp/benchmark"
MODEL_DIR="$DEVICE_DIR/models/qwen3-4b"
RESULTS_DIR="/home/liu/project/newwork/benchmark/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/qwen3-4b_benchmark_${TIMESTAMP}.log"

echo "=== Qwen3-4B Benchmark: MNN LLM vs llama.cpp ==="
echo "Timestamp: $TIMESTAMP"
echo ""

# Step 1: 设备握手
echo "--- Step 1: Device Check ---"
$ADB devices | grep "device$" || { echo "❌ Device not found"; exit 1; }
echo "Device: $($ADB shell getprop ro.product.model)"
echo "Temp: $($ADB shell cat /sys/class/thermal/thermal_zone0/temp | head -1)°C (raw)"
echo ""

# Step 2: Push GGUF models for llama.cpp
echo "--- Step 2: Push Models ---"
$ADB shell "mkdir -p $MODEL_DIR"

echo "Pushing Q4_K_M GGUF (2.5GB)..."
$ADB push models/llm/Qwen3-4B-Q4_K_M.gguf $MODEL_DIR/Qwen3-4B-Q4_K_M.gguf 2>&1 | tail -1

echo "Pushing Q8_0 GGUF (4.3GB)..."
$ADB push models/llm/Qwen3-4B-Q8_0.gguf $MODEL_DIR/Qwen3-4B-Q8_0.gguf 2>&1 | tail -1

# Step 3: Push MNN LLM model directories (exported by llmexport.py)
if [ -d "models/llm/Qwen3-4B-MNN-Q4" ]; then
    echo "Pushing MNN Q4 model dir..."
    $ADB shell "mkdir -p $MODEL_DIR/mnn_q4"
    $ADB push models/llm/Qwen3-4B-MNN-Q4/ $MODEL_DIR/mnn_q4/ 2>&1 | tail -1
fi
if [ -d "models/llm/Qwen3-4B-MNN-Q8" ]; then
    echo "Pushing MNN Q8 model dir..."
    $ADB shell "mkdir -p $MODEL_DIR/mnn_q8"
    $ADB push models/llm/Qwen3-4B-MNN-Q8/ $MODEL_DIR/mnn_q8/ 2>&1 | tail -1
fi

echo ""

# Step 4: 锁频 + 清缓存
echo "--- Step 3: Setup Test Environment ---"
./scripts/setup/setup-test-env.sh 2>&1 | tail -5
echo ""

# Step 5: 运行 llama.cpp benchmarks
echo "=== llama.cpp Qwen3-4B Benchmarks ===" | tee -a "$RESULT_FILE"

run_llama_bench() {
    local model="$1"
    local label="$2"
    echo "" | tee -a "$RESULT_FILE"
    echo "--- llama.cpp $label ---" | tee -a "$RESULT_FILE"
    $ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=. taskset f0 ./llama-bench \
        -m $MODEL_DIR/$model \
        -p 512 -n 128 \
        -t 4 \
        -r 5 \
        -o md" 2>&1 | tee -a "$RESULT_FILE"
}

run_llama_bench "Qwen3-4B-Q4_K_M.gguf" "Q4_K_M"
run_llama_bench "Qwen3-4B-Q8_0.gguf" "Q8_0"

# Step 6: 运行 MNN LLM benchmarks
echo "" | tee -a "$RESULT_FILE"
echo "=== MNN LLM Qwen3-4B Benchmarks ===" | tee -a "$RESULT_FILE"

run_mnn_bench() {
    local model="$1"
    local label="$2"
    echo "" | tee -a "$RESULT_FILE"
    echo "--- MNN LLM $label ---" | tee -a "$RESULT_FILE"
    $ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=. taskset f0 ./llm_benchmark \
        --backend mnn_llm \
        --model $MODEL_DIR/$model \
        --n-prompt 512 \
        --max-tokens 128 \
        --n-repeat 5 \
        --benchmark" 2>&1 | tee -a "$RESULT_FILE"
}

if [ -d "models/llm/Qwen3-4B-MNN-Q4" ]; then
    run_mnn_bench "mnn_q4/Qwen3-4B-MNN-Q4/config.json" "Q4"
fi
if [ -d "models/llm/Qwen3-4B-MNN-Q8" ]; then
    run_mnn_bench "mnn_q8/Qwen3-4B-MNN-Q8/config.json" "Q8"
fi

# Step 7: 恢复环境
echo "" | tee -a "$RESULT_FILE"
echo "--- Restore Environment ---"
./scripts/setup/restore-test-env.sh 2>&1 | tail -3

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $RESULT_FILE"
