#!/usr/bin/env python3
"""TVM Relay 编译 7 个方阵 GEMM ONNX → .so"""
import sys, os, time

TVM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "tvm")
sys.path.insert(0, os.path.join(TVM_ROOT, "python"))
sys.path.insert(0, os.path.join(TVM_ROOT, "topi", "python"))

import tvm
from tvm import relay
from tvm.relay.frontend.onnx import from_onnx
import onnx

TARGET = "llvm -mtriple=aarch64-linux-android -mattr=+neon -mcpu=cortex-a77"
NDK = "/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ONNX_DIR = os.path.join(PROJ_ROOT, "models", "single_ops_gemm")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compiled_models")
os.makedirs(OUT_DIR, exist_ok=True)

SQUARE_MODELS = [
    "GEMM_Square_16x16",
    "GEMM_Square_32x32",
    "GEMM_Square_64x64",
    "GEMM_Square_128x128",
    "GEMM_Square_256x256",
    "GEMM_Square_512x512",
    "GEMM_Square_1024x1024",
]

ok = skip = fail = 0
t0 = time.time()

for name in SQUARE_MODELS:
    so_path = f"{OUT_DIR}/{name}_tvm.so"
    if os.path.exists(so_path):
        print(f"[SKIP] {name} (已存在)")
        skip += 1
        continue

    onnx_path = f"{ONNX_DIR}/{name}.onnx"
    if not os.path.exists(onnx_path):
        print(f"[FAIL] {name} — ONNX 不存在: {onnx_path}")
        fail += 1
        continue

    t1 = time.time()
    try:
        m = onnx.load(onnx_path)
        # 获取输入 shape
        init_n = {x.name for x in m.graph.initializer}
        sd = {}
        for vi in m.graph.input:
            if vi.name not in init_n:
                sd[vi.name] = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
                break

        mod, params = from_onnx(m, shape=sd, freeze_params=True)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=TARGET, params=params)
        lib.export_library(so_path, cc=NDK)
        ok += 1
        kb = os.path.getsize(so_path) / 1024
        print(f"[OK]   {name:30s} {kb:6.0f}KB ({time.time()-t1:.1f}s)")
    except Exception as e:
        fail += 1
        print(f"[FAIL] {name:30s} {str(e)[:100]}")

print(f"\nOK={ok} SKIP={skip} FAIL={fail} in {(time.time()-t0)/60:.1f}min")
