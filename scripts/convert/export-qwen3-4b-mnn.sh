#!/bin/bash
# MNN LLM 导出脚本 — Qwen3-4B
# 用法: ./scripts/export_qwen3_4b_mnn.sh
set -e

VENV=/home/liu/.openharness-venv/bin/python3
MNN_EXPORT=/home/liu/project/newwork/benchmark/third_party/MNN/transformers/llm/export/llmexport.py
HF_MODEL=/home/liu/project/newwork/benchmark/models/llm/Qwen3-4B-HF
DST_DIR=/home/liu/project/newwork/benchmark/models/llm

echo "=== MNN LLM Export: Qwen3-4B ==="

# Check HF model exists
if [ ! -f "$HF_MODEL/config.json" ]; then
    echo "❌ HF model not found at $HF_MODEL"
    echo "   Run the ModelScope download first."
    exit 1
fi

echo "HF model found: $(du -sh $HF_MODEL | cut -f1)"

# Export Q4 (HQQ, ARM optimized)
echo ""
echo "--- Exporting Q4 (HQQ) ---"
$VENV $MNN_EXPORT \
    --path "$HF_MODEL" \
    --type qwen3 \
    --dst_path "$DST_DIR/Qwen3-4B-MNN-Q4" \
    --quant_bit 4 \
    --hqq \
    --export mnn \
    --onnx_slim

echo "Q4 export done: $(ls -lh $DST_DIR/Qwen3-4B-MNN-Q4/ 2>/dev/null | head -5)"

# Export Q8
echo ""
echo "--- Exporting Q8 ---"
$VENV $MNN_EXPORT \
    --path "$HF_MODEL" \
    --type qwen3 \
    --dst_path "$DST_DIR/Qwen3-4B-MNN-Q8" \
    --quant_bit 8 \
    --hqq \
    --export mnn \
    --onnx_slim

echo "Q8 export done: $(ls -lh $DST_DIR/Qwen3-4B-MNN-Q8/ 2>/dev/null | head -5)"
echo ""
echo "=== Export Complete ==="
