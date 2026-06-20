#!/bin/bash
# 将 GEMM ONNX 模型批量转换为 MNN 格式
# 用法: ./scripts/convert_gemm_mnn.sh

set -euo pipefail

MNN_CONVERT="/home/liu/project/newwork/benchmark/build_host_tools/MNN/MNNConvert"
ONNX_DIR="/home/liu/project/newwork/benchmark/models/single_ops_gemm"
MNN_DIR="/home/liu/project/newwork/benchmark/models/single_ops_gemm_mnn"

if [ ! -f "$MNN_CONVERT" ]; then
    echo "ERROR: MNNConvert not found at $MNN_CONVERT"
    echo "  Build with: ./scripts/build_host_tools.sh"
    exit 1
fi

mkdir -p "$MNN_DIR"

MANIFEST="/home/liu/project/newwork/benchmark/models/single_ops_gemm/manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest.json not found, run generate_gemm_operators.py first"
    exit 1
fi

echo "=== 转换 GEMM ONNX → MNN ==="
echo "ONNX: $ONNX_DIR"
echo "MNN:  $MNN_DIR"
echo ""

ok=0; skip=0; fail=0

for onnx_path in "$ONNX_DIR"/*.onnx; do
    base=$(basename "$onnx_path" .onnx)
    mnn_path="$MNN_DIR/${base}.mnn"

    if [ -f "$mnn_path" ]; then
        echo "[SKIP] $base (已存在)"
        ((skip++)) || true
        continue
    fi

    if "$MNN_CONVERT" -f ONNX --modelFile "$onnx_path" --MNNModel "$mnn_path" 2>&1 | \
        grep -v "^$" | head -2; then
        kb=$(du -k "$mnn_path" | cut -f1)
        echo "[OK]   $base → ${kb}KB"
        ((ok++)) || true
    else
        echo "[FAIL] $base"
        ((fail++)) || true
    fi
done

echo ""
echo "=== 完成: OK=$ok SKIP=$skip FAIL=$fail ==="
ls -lh "$MNN_DIR"/*.mnn | awk '{print $5, $NF}' | sed "s|$MNN_DIR/||"
