#!/usr/bin/env python3
"""编译 single_ops_extracted 中未编译的 TVM .so"""
import sys, os, time, json
TVM_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "tvm")
sys.path.insert(0, os.path.join(TVM_ROOT, "python"))
sys.path.insert(0, os.path.join(TVM_ROOT, "topi", "python"))

import tvm
from tvm import relay
from tvm.relay.frontend.onnx import from_onnx
import onnx

TARGET = "llvm -mtriple=aarch64-linux-android -mattr=+neon -mcpu=cortex-a77"
NDK = "/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"
EXTRACTED = os.path.join(os.path.dirname(__file__), "..", "..", "models", "single_ops_extracted")
OUT = os.path.join(os.path.dirname(__file__), "compiled_models")

with open("/tmp/remaining_ops.json") as f:
    ops = json.load(f)

ok = skip = fail = 0
t0 = time.time()
for i, name in enumerate(ops):
    so = f"{OUT}/{name}_tvm.so"
    if os.path.exists(so):
        skip += 1
        if i % 20 == 0: print(f"[{i+1:3d}/{len(ops)}] skip {name}")
        continue

    onnx_path = f"{EXTRACTED}/{name}.onnx"
    if not os.path.exists(onnx_path):
        fail += 1; continue

    t1 = time.time()
    try:
        m = onnx.load(onnx_path)
        init_n = {x.name for x in m.graph.initializer}
        sd = {}
        for vi in m.graph.input:
            if vi.name not in init_n:
                sd[vi.name] = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
                break
        mod, params = from_onnx(m, shape=sd, freeze_params=True)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=TARGET, params=params)
        os.makedirs(OUT, exist_ok=True)
        lib.export_library(so, cc=NDK)
        ok += 1
        kb = os.path.getsize(so) / 1024
        print(f"[{i+1:3d}/{len(ops)}] OK  {name:50s} {kb:6.0f}KB ({time.time()-t1:.1f}s)")
    except Exception as e:
        fail += 1
        print(f"[{i+1:3d}/{len(ops)}] FAIL {name:50s} {str(e)[:80]}")

elapsed = time.time() - t0
print(f"\nOK={ok} SKIP={skip} FAIL={fail} in {elapsed/60:.1f}min")
