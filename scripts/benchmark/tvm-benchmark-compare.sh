#!/bin/bash
# TVM vs MNN vs ORT 三方对比测试
# 前提: Docker TVM 已完成编译 (tvm-codegen image), benchmark_inference 已编译
set -euo pipefail

export ADB=/mnt/e/andorid/adb/adb.exe
export ANDROID_NDK=/home/liu/android-ndk
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/tools/tvm/compiled_models"
RESULTS_DIR="$PROJECT_DIR/results/phase_tvm"
DEVICE_DIR="/data/local/tmp/tvm_models"
mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

MODEL="${1:-mobilenetv2}"

echo "============================================"
echo "  TVM vs MNN vs ORT — $MODEL"
echo "============================================"

# ── Step 1: 设备握手 ──
echo "[1/6] Device check..."
$ADB devices | grep "device$" || { echo "ERROR: device not connected"; exit 1; }
$ADB shell getprop ro.product.model
echo "  OK"

# ── Step 2: Docker TVM 编译 ONNX → .so ──
echo "[2/6] TVM compilation (Docker)..."
SO_PATH="$OUTPUT_DIR/${MODEL}_tvm.so"
if [ -f "$SO_PATH" ]; then
    echo "  .so already exists: $SO_PATH ($(ls -lh "$SO_PATH" | awk '{print $5}'))"
else
    docker run --rm \
        -v "$PROJECT_DIR:/workspace" \
        -v "$OUTPUT_DIR:/output" \
        tvm-codegen \
        --model "$MODEL" \
        --output-dir "/output" \
        --opt-level 3 2>&1 | tee "$RESULTS_DIR/${MODEL}_tvm_compile.log"
    echo "  Compiled: $(ls -lh "$SO_PATH" | awk '{print $5}')"
fi

# ── Step 3: 推送 TVM .so 到设备 ──
echo "[3/6] Pushing TVM .so to device..."
$ADB shell mkdir -p "$DEVICE_DIR"
$ADB push "$SO_PATH" "$DEVICE_DIR/"
echo "  OK"

# ── Step 4: 编译并推送 benchmark 二进制 ──
echo "[4/6] Building benchmark binary..."
cd "$PROJECT_DIR"
./scripts/build/build-android.sh 2>&1 | tail -5
$ADB push build_android/src/benchmark_inference /data/local/tmp/
echo "  OK"

# ── Step 5: 运行三方对比 ──
echo "[5/6] Running benchmark..."

for backend in mnn ort tvm; do
    echo ""
    echo "  === $backend ==="
    $ADB shell "/data/local/tmp/benchmark_inference \
        --model $MODEL --backend $backend \
        --iterations 100 --warmup 50 --threads 4" \
        2>&1 | tee "$RESULTS_DIR/${MODEL}_${backend}.log"
done

# ── Step 6: 提取对比摘要 ──
echo "[6/6] Extracting summary..."

python3 -c "
import re, sys, os

def extract_latency(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        text = f.read()
    m = re.search(r'Mean:\s+([\d.]+)\s*ms', text)
    return float(m.group(1)) if m else None

print()
print('=== TVM vs MNN vs ORT — $MODEL ===')
print('| 后端   | 延迟 (ms) |')
print('|--------|----------|')
for b in ['mnn', 'ort', 'tvm']:
    lat = extract_latency(f'$RESULTS_DIR/${MODEL}_{b}.log')
    if lat:
        print(f'| {b:6s} | {lat:8.2f} |')
    else:
        print(f'| {b:6s} |    FAILED |')
" 2>&1 | tee "$RESULTS_DIR/${MODEL}_summary.md"

echo ""
echo "=== Done ==="
echo "Results: $RESULTS_DIR/"
