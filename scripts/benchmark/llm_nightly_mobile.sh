#!/bin/bash
set -e
# MNN LLM Mobile Nightly Benchmark
# Adapted from transformers/llm/benchmark/llm_nightly.sh for Android SD865

ADB="/mnt/e/andorid/adb/adb.exe"
DEVICE_DIR="/data/local/tmp/benchmark"
MODEL_DIR="${DEVICE_DIR}/models/qwen3-0.6b-mnn"
ANDROID_NDK="${ANDROID_NDK:-/home/liu/android-ndk}"
PROJECT_ROOT="$(cd $(dirname $0)/../..; pwd)"
MNN_ROOT="${PROJECT_ROOT}/third_party/MNN"
THREAD_NUM=4
MODEL_NAME="Qwen3-0.6B"

echo "=== MNN LLM Mobile Nightly: ${MODEL_NAME} ==="

# 1. Export Model with HQQ
echo ">>> Exporting ${MODEL_NAME} with HQQ quantization..."
cd ${MNN_ROOT}/transformers/llm/export
source /home/liu/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

python llmexport.py \
    --path /tmp/qwen3-0.6b-hf \
    --export mnn \
    --hqq \
    --dst_path /tmp/qwen3-0.6b-mnn-hqq 2>&1 | tail -3

# 2. Build Android llm_bench
echo ">>> Building Android llm_bench..."
mkdir -p ${MNN_ROOT}/build_android_mobile
cd ${MNN_ROOT}/build_android_mobile

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON \
    -DMNN_ARM82=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_BUILD_TEST=OFF -DMNN_BUILD_BENCHMARK=OFF -DMNN_BUILD_TOOLS=OFF \
    -DMNN_BUILD_SHARED_LIBS=ON

make -j$(nproc) llm_bench

# 3. Push to device
echo ">>> Pushing to device..."
${ADB} devices | grep "device$" || { echo "ERROR: No device!"; exit 1; }

# Model
${ADB} shell "su -c 'mkdir -p ${MODEL_DIR}'"
for f in llm.mnn llm.mnn.weight llm_config.json tokenizer.mtok config.json; do
    if [ -f "/tmp/qwen3-0.6b-mnn-hqq/$f" ]; then
        ${ADB} push "/tmp/qwen3-0.6b-mnn-hqq/$f" /data/local/tmp/
        ${ADB} shell "su -c 'cp /data/local/tmp/$f ${MODEL_DIR}/'"
    fi
done

# Fix thread_num in config
${ADB} shell "su -c 'sed -i \"s/\\\"thread_num\\\": 4/\\\"thread_num\\\": ${THREAD_NUM}/\" ${MODEL_DIR}/config.json'"

# Libraries & binary
${ADB} push ${MNN_ROOT}/build_android_mobile/OFF/arm64-v8a/libMNN.so /data/local/tmp/
${ADB} push ${MNN_ROOT}/build_android_mobile/OFF/arm64-v8a/libllm.so /data/local/tmp/
${ADB} push ${MNN_ROOT}/build_android_mobile/llm_bench /data/local/tmp/
${ADB} shell "su -c 'cp /data/local/tmp/libMNN.so ${DEVICE_DIR}/ && cp /data/local/tmp/libllm.so ${DEVICE_DIR}/ && cp /data/local/tmp/llm_bench ${DEVICE_DIR}/ && chmod +x ${DEVICE_DIR}/llm_bench'"

# 4. Performance Test (cold start, clear cache)
echo ">>> Running Performance Benchmark..."
${ADB} shell "su -c 'echo 3 > /proc/sys/vm/drop_caches'"
${ADB} shell "su -c 'cd ${DEVICE_DIR}; LD_LIBRARY_PATH=${DEVICE_DIR} ./llm_bench -m ${MODEL_DIR}/config.json -p 512 -n 128 -t ${THREAD_NUM} -j 2>&1'" | tee /tmp/llm_mobile_bench.log

# 5. Extract results
echo ""
echo "=== Results ==="
python3 -c "
import json, subprocess, sys
try:
    with open('/tmp/qwen3-0.6b-mnn-hqq/llm_bench.json') as f:
        data = json.load(f)
    for r in data.get('results', []):
        print(f\"{r['type']}: {r['tps']:.2f} ± {r.get('std',0):.2f} t/s\")
except Exception as e:
    print(f'Could not parse results: {e}')
    # Fallback: grep from log
    pass
" 2>/dev/null || grep -E "tg128|pp512|prefill|decode" /tmp/llm_mobile_bench.log

echo ""
echo "=== Device Info ==="
${ADB} shell cat /proc/cpuinfo | grep -E "Hardware|processor" | head -5
${ADB} shell getprop ro.product.model

echo "=== Done ==="
