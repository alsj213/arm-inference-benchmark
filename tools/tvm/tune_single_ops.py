#!/usr/bin/env python3
"""MetaSchedule 单算子 RPC 调优 — 小算子低内存，适合 WSL2"""
import os, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TVM_ROOT = PROJECT_ROOT / "third_party" / "tvm"
OUTPUT_DIR = SCRIPT_DIR / "compiled_models"
SINGLE_OPS_DIR = PROJECT_ROOT / "models" / "single_ops"

sys.path.insert(0, str(TVM_ROOT / "python"))
sys.path.insert(0, str(TVM_ROOT / "3rdparty" / "tvm-ffi" / "python"))

os.environ.setdefault("TVM_TRACKER_HOST", "127.0.0.1")
os.environ.setdefault("TVM_TRACKER_PORT", "9190")
os.environ.setdefault("TVM_TRACKER_KEY", "snapdragon865")

import tvm
from tvm import relax
from tvm.relax.frontend.onnx.onnx_frontend import from_onnx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.builder import LocalBuilder
import onnx

TARGET_ARM = tvm.target.Target({
    "kind": "llvm", "mtriple": "aarch64-linux-android",
    "mattr": ["+neon"], "mcpu": "cortex-a77", "num-cores": 8,
})

NDK_CXX = "/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"

# 关键 CNN 算子
OPS_TO_TUNE = {
    "Conv1x1_K16_C64_M784":   (1, 64, 28, 28),    # MobileNetV2 小 conv
    "Conv1x1_K1024_C256_M784":(1, 256, 28, 28),   # ResNet bottleneck
    "DWConv_C960_3x3":        (1, 960, 7, 7),     # MobileNetV2 depthwise
    "MatMul_512x512x512":     (1, 512),            # BERT intermediate
}


def tune_single_op(op_name: str, shape: tuple, trials: int = 100):
    onnx_path = SINGLE_OPS_DIR / f"{op_name}.onnx"
    if not onnx_path.exists():
        print(f"  ⚠️ {op_name}: ONNX 不存在, 跳过")
        return None

    work_dir = str(OUTPUT_DIR / "tuning_logs" / op_name)
    os.makedirs(work_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  {op_name}  shape={shape}  trials={trials}")
    print(f"{'='*50}")

    # 1. 加载
    onnx_model = onnx.load(str(onnx_path))
    input_name = onnx_model.graph.input[0].name
    mod = from_onnx(onnx_model, shape_dict={input_name: list(shape)})
    print(f"  函数: {list(mod.functions.keys())}")

    # 2. 调优
    t0 = time.time()
    try:
        database = ms.relax_integration.tune_relax(
            mod=mod, target=TARGET_ARM, params={},
            work_dir=work_dir,
            max_trials_global=trials,
            num_trials_per_iter=2,
            strategy="evolutionary",
            cost_model="xgb",
            space="post-order-apply",
            runner="rpc",
            builder=LocalBuilder(timeout_sec=600, max_workers=1),
        )
        t1 = time.time()
        print(f"  调优: {(t1-t0)/60:.1f} min")

        mod = database.apply(mod)
    except Exception as e:
        print(f"  ❌ 调优失败: {e}")
        return None

    # 3. 编译
    print(f"  编译 Relax → .so ...")
    t2 = time.time()
    pipeline = relax.get_pipeline()
    with TARGET_ARM:
        built_mod = pipeline(mod)
    executable = tvm.compile(built_mod, target=TARGET_ARM)

    so_path = OUTPUT_DIR / f"{op_name}_tvm.so"
    executable.export_library(str(so_path),
        workspace_dir=str(OUTPUT_DIR / "workspace"), cc=NDK_CXX)
    t3 = time.time()

    kb = so_path.stat().st_size / 1024
    print(f"  ✅ {so_path.name} ({kb:.0f} KB)  编译:{(t3-t2)/60:.1f}min  总计:{(t3-t0)/60:.1f}min")
    return so_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("op", nargs="?", default=None)
    args = parser.parse_args()

    if args.op:
        ops = {args.op: OPS_TO_TUNE[args.op]}
    else:
        ops = OPS_TO_TUNE

    results = {}
    t_total = time.time()
    for name, shape in ops.items():
        results[name] = tune_single_op(name, shape, args.trials)

    print(f"\n{'='*50}")
    print(f"汇总 ({len(ops)} 算子, 总耗时 {(time.time()-t_total)/60:.1f} min):")
    for name, path in results.items():
        status = f"{path.stat().st_size/1024:.0f} KB" if path else "失败"
        print(f"  {'✅' if path else '❌'} {name}: {status}")
