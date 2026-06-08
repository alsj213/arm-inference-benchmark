#!/usr/bin/env python3
"""
TVM Relay 全量模型编译脚本 — v0.15.0 stable
=============================================
ONNX → Relay IR → relay.build() → .so (GraphExecutor + NDK 交叉编译)

用法:
  export TVM_ROOT=third_party/tvm
  export PYTHONPATH=$TVM_ROOT/python:$TVM_ROOT/topi/python
  export LD_LIBRARY_PATH=$TVM_ROOT/build:$LD_LIBRARY_PATH

  python3 tools/tvm/compile_all_models.py --all
  python3 tools/tvm/compile_all_models.py mobilenetv2
  python3 tools/tvm/compile_all_models.py --single-ops
"""

import os, sys, time, argparse
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TVM_ROOT = PROJECT_ROOT / "third_party" / "tvm"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = SCRIPT_DIR / "compiled_models"

sys.path.insert(0, str(TVM_ROOT / "python"))
sys.path.insert(0, str(TVM_ROOT / "topi" / "python"))

import tvm
from tvm import relay
from tvm.relay.frontend.onnx import from_onnx

NDK_CXX = "/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"
TARGET_ARM = "llvm -mtriple=aarch64-linux-android -mattr=+neon -mcpu=cortex-a77"

ONNX_MODELS = {
    "mobilenetv2": {
        "onnx_path": "models/classification/mobilenetv2/mobilenetv2.onnx",
        "input_shape": [1, 3, 224, 224],
    },
    "resnet50": {
        "onnx_path": "models/classification/resnet50/resnet50.onnx",
        "input_shape": [1, 3, 224, 224],
    },
    "yolov8n": {
        "onnx_path": "models/detection/yolov8n/yolov8n.onnx",
        "input_shape": [1, 3, 640, 640],
        "input_name": "images",
    },
    "bert": {
        "onnx_path": "models/nlp/bert/bert.onnx",
        "input_shape": [1, 128],
        "input_name": "input_ids",
    },
    "mobilevit_s": {
        "onnx_path": "models/classification/mobilevit_s/mobilevit_s.onnx",
        "input_shape": [1, 3, 256, 256],
        "input_name": "input",
    },
}

SINGLE_OP_MODELS = {
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
    "LayerNorm_512":              (1, 512),
    "Softmax_128":                (1, 128),
    "GELU_512x3072":              (1, 512, 3072),
}

SINGLE_OPS_DIR = MODELS_DIR / "single_ops"


def compile_onnx_model(model_name: str, cfg: dict) -> Optional[Path]:
    """ONNX → Relay → .so (GraphExecutorFactory, 包含 graph + kernel + params)"""
    onnx_path = PROJECT_ROOT / cfg["onnx_path"]
    if not onnx_path.exists():
        print(f"  ONNX 不存在: {onnx_path}")
        return None

    so_path = OUTPUT_DIR / f"{model_name}_tvm.so"
    if so_path.exists():
        size_mb = so_path.stat().st_size / (1024 * 1024)
        print(f"  {model_name}: 已存在 ({size_mb:.1f} MB), 跳过")
        return so_path

    print(f"\n{'='*60}")
    print(f"[Relay] 编译 {model_name} → .so")
    print(f"  ONNX:   {onnx_path}")
    print(f"  Target: {TARGET_ARM}")
    print(f"{'='*60}")

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 获取第一个输入名并固定形状
    input_name = cfg.get("input_name", None)
    if input_name is None:
        input_name = onnx_model.graph.input[0].name
    shape_dict = {input_name: cfg["input_shape"]}

    t0 = time.time()
    mod, params = from_onnx(onnx_model, shape=shape_dict, freeze_params=True)
    print(f"  Load: {time.time()-t0:.1f}s, functions={list(mod.functions.keys())}")

    t1 = time.time()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=TARGET_ARM, params=params)
    print(f"  Build: {time.time()-t1:.1f}s")

    t2 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lib.export_library(str(so_path), cc=NDK_CXX)
    size_mb = so_path.stat().st_size / (1024 * 1024)
    print(f"  {so_path.name} ({size_mb:.1f} MB) total={time.time()-t0:.1f}s")
    return so_path


def compile_single_op(op_name: str, input_shape: tuple) -> Optional[Path]:
    """编译单个算子"""
    onnx_path = SINGLE_OPS_DIR / f"{op_name}.onnx"
    if not onnx_path.exists():
        print(f"  {op_name}: ONNX 不存在")
        return None

    so_path = OUTPUT_DIR / f"{op_name}_tvm.so"
    if so_path.exists():
        print(f"  {op_name}: 已存在 ({so_path.stat().st_size/1024:.0f} KB), 跳过")
        return so_path

    print(f"  [Relay] {op_name} shape={input_shape} ...", end=" ", flush=True)
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        input_name = onnx_model.graph.input[0].name
        mod, params = from_onnx(onnx_model, shape={input_name: list(input_shape)}, freeze_params=True)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=TARGET_ARM, params=params)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        lib.export_library(str(so_path), cc=NDK_CXX)
        print(f"OK ({so_path.stat().st_size/1024:.0f} KB)")
        return so_path
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def print_summary(results: dict):
    print(f"\n{'='*60}")
    print("编译汇总")
    print(f"{'='*60}")
    success = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]
    for name in success:
        so_path = results[name]
        if so_path and so_path.exists():
            size_mb = so_path.stat().st_size / (1024 * 1024)
            print(f"  {'✅' if so_path else '❌'} {name:40s} {size_mb:7.1f} MB")
    for name in failed:
        print(f"  ❌ {name:40s} (编译失败)")
    for name in skipped:
        print(f"  ⏭️  {name:40s} (文件缺失)")
    print(f"\n  成功: {len(success)}, 失败: {len(failed)}, 跳过: {len(skipped)}")


def main():
    parser = argparse.ArgumentParser(description="TVM Relay 模型编译 (v0.15.0)")
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--single-ops", action="store_true")
    args = parser.parse_args()

    results = {}

    if args.all or args.model in ONNX_MODELS:
        for name, cfg in ONNX_MODELS.items():
            if args.model and args.model != name:
                continue
            try:
                result = compile_onnx_model(name, cfg)
                results[name] = result
            except Exception as e:
                print(f"  {name} 编译失败: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

    if args.single_ops:
        print(f"\n{'='*60}")
        print(f"编译单算子 ({len(SINGLE_OP_MODELS)} 个)")
        print(f"{'='*60}")
        for op_name, shape in SINGLE_OP_MODELS.items():
            result = compile_single_op(op_name, shape)
            results[op_name] = result

    if not args.all and not args.single_ops and not args.model:
        parser.print_help()
        return

    print_summary(results)


if __name__ == "__main__":
    main()
