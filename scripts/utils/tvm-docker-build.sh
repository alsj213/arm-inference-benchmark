#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# DEPRECATED: 本项目已不再使用 Docker 方案编译 TVM 模型。
# 当前使用源码编译: python tools/tvm/compile_model_relax.py
# Docker 镜像 tvm-codegen 不存在，此脚本仅保留作为参考。
# ═══════════════════════════════════════════════════════════════
# TVM Codegen (Docker) → .so → 推理
#
# 用法:
#   ./scripts/tvm_docker_build.sh              # 构建 Docker 镜像 (一次)
#   ./scripts/tvm_docker_build.sh --compile mobilenetv2  # 编译模型
#   ./scripts/tvm_docker_build.sh --all                   # 编译全部 6 个模型
#
# 架构:
#   Docker (TVM + LLVM) → ONNX → .so → 本项目 tvm_backend.cpp dlopen 加载

echo "⚠️  DEPRECATED: 本项目已改为源码直接编译 (tools/tvm/compile_model_relax.py)"
echo "    此 Docker 方案不再使用，仅保留脚本作为参考。"
exit 0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_DIR/tools/tvm/docker"
OUTPUT_DIR="$PROJECT_DIR/tools/tvm/compiled_models"

mkdir -p "$OUTPUT_DIR"

cmd="${1:-}"

# ── 构建 Docker 镜像 ──
if [ "$cmd" = "--build" ] || [ "$cmd" = "" ]; then
    echo "=== Building TVM Docker image ==="
    docker build -t tvm-codegen "$DOCKER_DIR"
    echo "=== Image built: tvm-codegen ==="
fi

# ── 编译单个模型 ──
if [ "$cmd" = "--compile" ]; then
    MODEL="${2:-mobilenetv2}"
    echo "=== Compiling $MODEL with TVM (Docker) ==="
    docker run --rm \
        -v "$PROJECT_DIR:/workspace" \
        -v "$OUTPUT_DIR:/output" \
        tvm-codegen \
        --model "$MODEL" \
        --output-dir "/output" \
        --opt-level 3

    echo "=== Compiled artifacts ==="
    ls -lh "$OUTPUT_DIR/${MODEL}_tvm.so" 2>/dev/null || echo "  (check Docker output above)"
    ls -lh "$OUTPUT_DIR/${MODEL}_tvm_meta.json" 2>/dev/null
    echo ""
    echo "Next: push to device + run benchmark"
    echo "  adb push $OUTPUT_DIR/${MODEL}_tvm.so /data/local/tmp/tvm_models/"
    echo "  adb shell /data/local/tmp/benchmark_inference --model $MODEL --backend tvm"
fi

# ── 批量编译 ──
if [ "$cmd" = "--all" ]; then
    for model in mobilenetv2 resnet50 bert yolov8n qwen2_05b mobilevit_s; do
        echo ""
        echo "============================================"
        echo "  Compiling: $model"
        echo "============================================"
        docker run --rm \
            -v "$PROJECT_DIR:/workspace" \
            -v "$OUTPUT_DIR:/output" \
            tvm-codegen \
            --model "$model" \
            --output-dir "/output"
    done
    echo ""
    echo "=== All models compiled ==="
    ls -lh "$OUTPUT_DIR/"
fi

# ── 帮助 ──
if [ "$cmd" = "--help" ] || [ "$cmd" = "-h" ]; then
    echo "TVM Docker Codegen Pipeline"
    echo ""
    echo "Usage:"
    echo "  $0              Build Docker image"
    echo "  $0 --compile <model>   Compile single model (mobilenetv2/resnet50/bert/yolov8n/qwen2_05b/mobilevit_s)"
    echo "  $0 --all               Compile all 6 models"
    echo ""
    echo "After compilation:"
    echo "  $0 --push <model>      Push .so to device + run benchmark"
fi
