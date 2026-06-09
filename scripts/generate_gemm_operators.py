#!/usr/bin/env python3
"""
生成 GEMM (MatMul) 单算子 ONNX 模型，覆盖 4 个维度扫描：

轴1 方阵扫描: M=N=K, 16→1024 (7个) — Pack/Unpack 占比主曲线
轴2 内维K扫描: M=N=256, K=16→1024 (7个) — A Pack 占比变化
轴3 外维MN扫描: K=256, M=N=16→1024 (7个) — B Pack 占比变化
轴4 非方阵: BERT/LLM 典型 shapes (4个)

用途:
  Task A: MNN CPU 阶段计时 (pack_B / pack_A / compute / unpack_C)
  Task B: MNN CPU vs GPU 交叉点测试

输出目录: models/single_ops_gemm/
"""

import os, sys, json

import numpy as np
import onnx
from onnx import helper, TensorProto

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "single_ops_gemm")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════
# 算子设计
# ══════════════════════════════════════════

OPERATORS = []

# ── 轴1: 方阵扫描 (M=N=K) ──
SQUARE_DIMS = [16, 32, 64, 128, 256, 512, 1024]
for d in SQUARE_DIMS:
    OPERATORS.append({
        "name": f"GEMM_Square_{d}x{d}",
        "M": d, "K": d, "N": d,
        "axis": "square",
        "flops": 2 * d * d * d,
        "memo": f"{d}×{d}×{d} FLOPs={2*d**3:,}"
    })

# ── 轴2: 内维 K 扫描 (M=N=256 固定) ──
K_DIMS = [16, 32, 64, 128, 512, 1024]  # 256 已由方阵覆盖
for k in K_DIMS:
    m = n = 256
    OPERATORS.append({
        "name": f"GEMM_KScan_K{k}_M256_N256",
        "M": m, "K": k, "N": n,
        "axis": "kscan",
        "flops": 2 * m * n * k,
        "memo": f"M=256,K={k},N=256 FLOPs={2*m*n*k:,}"
    })

# ── 轴3: 外维 MN 扫描 (K=256 固定) ──
MN_DIMS = [16, 32, 64, 128, 512, 1024]  # 256 已由方阵覆盖
for d in MN_DIMS:
    OPERATORS.append({
        "name": f"GEMM_MNScan_M{d}_K256_N{d}",
        "M": d, "K": 256, "N": d,
        "axis": "mnscan",
        "flops": 2 * d * 256 * d,
        "memo": f"M={d},K=256,N={d} FLOPs={2*d*256*d:,}"
    })

# ── 轴4: BERT/LLM 典型非方阵 ──
NONSQUARE = [
    # BERT FFN 第一层: [seq, 768] × [768, 3072] → [seq, 3072]
    {"name": "GEMM_BERT_FFN1_S128_H768_I3072", "M": 128, "K": 768, "N": 3072,
     "axis": "nonsquare", "memo": "BERT FFN1: M=128(seq),K=768(hidden),N=3072(inter)"},
    # BERT projection: [seq, 3072] × [3072, 768] → [seq, 768]
    {"name": "GEMM_BERT_Proj_S128_H3072_O768", "M": 128, "K": 3072, "N": 768,
     "axis": "nonsquare", "memo": "BERT Proj: M=128,K=3072,N=768"},
    # Attention score: [seq, 768] × [768, seq] (Q×K^T)
    {"name": "GEMM_Attention_128x768x128", "M": 128, "K": 768, "N": 128,
     "axis": "nonsquare", "memo": "Attention Q×K^T: M=128,K=768,N=128"},
    # LLM large FFN: [1, 4096] × [4096, 14336]
    {"name": "GEMM_LLM_FFN_1x4096x14336", "M": 1, "K": 4096, "N": 14336,
     "axis": "nonsquare", "memo": "LLM FFN large: M=1(token),K=4096(hidden),N=14336(inter)"},
    # LLM attention project: [1, 4096] × [4096, 4096]
    {"name": "GEMM_LLM_AttnProj_1x4096x4096", "M": 1, "K": 4096, "N": 4096,
     "axis": "nonsquare", "memo": "LLM Attn Proj: M=1,K=4096,N=4096"},
]
for op in NONSQUARE:
    op["flops"] = 2 * op["M"] * op["K"] * op["N"]
    OPERATORS.append(op)


def make_gemm_onnx(name, M, K, N):
    """
    创建 GEMM ONNX: C = A × B
    A: [M, K]  (输入)
    B: [K, N]  (权重, 初始器)
    C: [M, N]  (输出)
    """
    A = helper.make_tensor_value_info("input", TensorProto.FLOAT, [M, K])
    C = helper.make_tensor_value_info("output", TensorProto.FLOAT, [M, N])

    # Weight as initializer (固定, 随机值)
    weight_data = np.random.RandomState(42).randn(K, N).astype(np.float32)
    weight_init = helper.make_tensor(
        "weight", TensorProto.FLOAT, [K, N],
        weight_data.tobytes(), raw=True
    )

    # MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "weight"],
        outputs=["output"],
        name=f"MatMul_{name}"
    )

    graph = helper.make_graph(
        nodes=[matmul_node],
        name=name,
        inputs=[A],
        outputs=[C],
        initializer=[weight_init],
    )

    opset_imports = [helper.make_opsetid("", 18)]
    model = helper.make_model(graph, opset_imports=opset_imports, ir_version=10)
    model = onnx.shape_inference.infer_shapes(model)
    return model


def main():
    print(f"生成 {len(OPERATORS)} 个 GEMM ONNX 模型 → {OUT_DIR}/")
    print()

    manifest = []

    for i, op in enumerate(OPERATORS):
        name = op["name"]
        onnx_path = os.path.join(OUT_DIR, f"{name}.onnx")

        if os.path.exists(onnx_path):
            print(f"[{i+1:2d}/{len(OPERATORS)}] SKIP {name} (已存在)")
        else:
            model = make_gemm_onnx(name, op["M"], op["K"], op["N"])
            onnx.save(model, onnx_path)
            size_kb = os.path.getsize(onnx_path) / 1024
            print(f"[{i+1:2d}/{len(OPERATORS)}] OK   {name:45s} [{op['M']},{op['K']}]×[{op['K']},{op['N']}] {size_kb:.0f}KB  {op['memo']}")

        manifest.append({
            "name": name,
            "onnx": f"models/single_ops_gemm/{name}.onnx",
            "M": op["M"], "K": op["K"], "N": op["N"],
            "flops": op["flops"],
            "axis": op["axis"],
            "input_name": "input",
            "input_shape": [op["M"], op["K"]],
        })

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # 按 axis 统计
    from collections import Counter
    axis_counts = Counter(m["axis"] for m in manifest)
    print(f"\n→ 生成完成: {len(OPERATORS)} 个 ONNX, manifest → {manifest_path}")
    print(f"   分类: {dict(axis_counts)}")
    for ax in ["square", "kscan", "mnscan", "nonsquare"]:
        shapes = [(m["M"], m["K"], m["N"]) for m in manifest if m["axis"] == ax]
        print(f"   {ax}: {len(shapes)} ops, shapes={shapes}")


if __name__ == "__main__":
    main()
