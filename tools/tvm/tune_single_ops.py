#!/usr/bin/env python3
"""
TVM 单算子多优化等级编译 — opt=3 vs opt=4 对比
================================================
AutoTVM 调优需要 RPC 连接设备实测，CPP RPC server 不支持 standalone 直接连接，
RPC tracker 有 tornado/cloudpickle 依赖冲突。

替代方案: 编译同一算子用 opt_level=3 (基线) 和 opt_level=4 (激进优化)，
在设备上 benchmark 对比，评估编译器优化是否能改善单算子性能。

用法:
  python3 tools/tvm/tune_single_ops.py --all
  python3 tools/tvm/tune_single_ops.py Conv1x1_K16_C64_M784
"""

import os, sys, time, argparse, json
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TVM_ROOT = PROJECT_ROOT / "third_party" / "tvm"
OUTPUT_DIR = SCRIPT_DIR / "compiled_models"
SINGLE_OPS_DIR = PROJECT_ROOT / "models" / "single_ops"

sys.path.insert(0, str(TVM_ROOT / "python"))
sys.path.insert(0, str(TVM_ROOT / "topi" / "python"))

import tvm
from tvm import relay
from tvm.relay.frontend.onnx import from_onnx

NDK_CXX = "/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"
TARGET_ARM = "llvm -mtriple=aarch64-linux-android -mattr=+neon -mcpu=cortex-a77"

OPS = {
    "Conv1x1_K16_C64_M784":       (1, 64, 28, 28),
    "Conv1x1_K1024_C256_M784":    (1, 256, 28, 28),
    "Conv1x1_M49_C32_K64":        (1, 32, 7, 7),
    "Conv1x1_M49_C256_K512":      (1, 256, 7, 7),
    "Conv1x1_M784_C32_K64":       (1, 32, 28, 28),
    "Conv1x1_M3136_C64_K128":     (1, 64, 56, 56),
    "Conv1x1_Misaligned_C31_K64": (1, 31, 56, 56),
    "Conv1x1_Misaligned_C33_K64": (1, 33, 56, 56),
    "DWConv_C16_3x3":             (1, 16, 112, 112),
    "DWConv_C960_3x3":            (1, 960, 7, 7),
    "MatMul_512x512x512":         (1, 512),
    "MatMul_768x768x768":         (1, 768),
    "MatMul_3072x768":            (1, 3072),
    "MatMul_768x3072":            (1, 768),
    "LayerNorm_BERT":             (1, 128, 768),
    "Softmax_BERT":               (1, 128, 128),
    "GELU_BERT":                  (1, 128, 3072),
}


def compile_op(op_name: str, shape: tuple, opt_level: int) -> Optional[Path]:
    onnx_path = SINGLE_OPS_DIR / f"{op_name}.onnx"
    if not onnx_path.exists():
        return None

    suffix = "_opt4_tvm" if opt_level == 4 else "_tvm"
    so_path = OUTPUT_DIR / f"{op_name}{suffix}.so"
    if so_path.exists():
        return so_path

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    input_name = onnx_model.graph.input[0].name
    mod, params = from_onnx(onnx_model, shape={input_name: list(shape)},
                            freeze_params=True)

    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target=TARGET_ARM, params=params)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lib.export_library(str(so_path), cc=NDK_CXX)
    return so_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("op", nargs="?", default=None)
    args = parser.parse_args()

    if args.op:
        ops = {args.op: OPS[args.op]}
    elif args.all:
        ops = OPS
    else:
        parser.print_help()
        return

    results = {}
    t0 = time.time()

    for name, shape in ops.items():
        print(f"\n{'='*50}")
        print(f"  {name} shape={shape}")
        print(f"{'='*50}")
        try:
            t1 = time.time()
            u = compile_op(name, shape, 3)
            ukb = u.stat().st_size / 1024 if u else 0
            print(f"  opt=3: {ukb:.0f}KB ({time.time()-t1:.1f}s)")

            t1 = time.time()
            t = compile_op(name, shape, 4)
            tkb = t.stat().st_size / 1024 if t else 0
            print(f"  opt=4: {tkb:.0f}KB ({time.time()-t1:.1f}s)")

            results[name] = {"opt3": u, "opt4": t}
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"汇总 ({(time.time()-t0)/60:.1f} min)")
    print(f"{'='*50}")
    for name, r in results.items():
        u = r["opt3"]; t = r["opt4"]
        print(f"  {'✅' if u and t else '❌'} {name:35s} opt3={u.stat().st_size/1024:.0f}KB opt4={t.stat().st_size/1024:.0f}KB")


if __name__ == "__main__":
    main()
