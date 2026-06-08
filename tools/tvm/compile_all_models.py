#!/usr/bin/env python3
"""
TVM Relax 全量模型编译脚本 — Phase B
=====================================
支持 2 条编译路径:
  路径 A: torch.export → Relax (resnet50, mobilenetv2)
  路径 B: ONNX → Relax (yolov8n, bert, 单算子)

用法:
  export TVM_ROOT=third_party/tvm
  export PYTHONPATH=$TVM_ROOT/python:$TVM_ROOT/3rdparty/tvm-ffi/python
  export LD_LIBRARY_PATH=$TVM_ROOT/build:$TVM_ROOT/build/lib:$LD_LIBRARY_PATH

  # 全量编译
  python3 tools/tvm/compile_all_models.py --all

  # 单模型
  python3 tools/tvm/compile_all_models.py resnet50
  python3 tools/tvm/compile_all_models.py yolov8n

  # 单算子
  python3 tools/tvm/compile_all_models.py --single-ops

  # 自动调优
  python3 tools/tvm/compile_all_models.py --all --tune
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Callable

# ── 路径设置 ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TVM_ROOT = PROJECT_ROOT / "third_party" / "tvm"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = SCRIPT_DIR / "compiled_models"

sys.path.insert(0, str(TVM_ROOT / "python"))
sys.path.insert(0, str(TVM_ROOT / "3rdparty" / "tvm-ffi" / "python"))

import tvm
from tvm import relax
from tvm.relax.frontend.onnx.onnx_frontend import from_onnx

# ── NDK 交叉编译器 ──
NDK_CXX = "/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"

# ARM Android target (骁龙 865: ARMv8.2-A + NEON)
TARGET_ARM = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "aarch64-linux-android",
    "mattr": ["+neon"],
    "mcpu": "cortex-a77",
})

# ── 整模型配置 ──
# 路径 A: torchvision 模型
TORCHVISION_MODELS = {
    "resnet50": {
        "type": "torchvision",
        "class": "resnet50",
        "input_shape": (1, 3, 224, 224),
    },
    "mobilenetv2": {
        "type": "torchvision",
        "class": "mobilenet_v2",
        "input_shape": (1, 3, 224, 224),
    },
}

# 路径 B: ONNX 模型 (单输入)
ONNX_MODELS = {
    "yolov8n": {
        "type": "onnx",
        "onnx_path": "models/detection/yolov8n/yolov8n.onnx",
        "input_shape": (1, 3, 640, 640),
        "input_name": "images",
    },
    "bert": {
        "type": "onnx",
        "onnx_path": "models/nlp/bert/bert_patched.onnx",
        "input_shape": (1, 128),
        "input_name": "input_ids",
        # BERT has 2 inputs, but we use input_ids only for now
        # attention_mask is set to all ones internally
    },
    "mobilevit_s": {
        "type": "onnx",
        "onnx_path": "models/classification/mobilevit_s/mobilevit_s.onnx",
        "input_shape": (1, 3, 256, 256),
        "input_name": "input",
    },
}

# ── 单算子配置 ──
SINGLE_OP_MODELS = {
    # conv1x1
    "Conv1x1_K16_C64_M784":       (1, 64, 28, 28),
    "Conv1x1_K1024_C256_M784":    (1, 256, 28, 28),
    "Conv1x1_M49_C32_K64":        (1, 32, 7, 7),
    "Conv1x1_M49_C256_K512":      (1, 256, 7, 7),
    "Conv1x1_M784_C32_K64":       (1, 32, 28, 28),
    "Conv1x1_M3136_C64_K128":     (1, 64, 56, 56),
    # conv1x1 misaligned
    "Conv1x1_Misaligned_C31_K64": (1, 31, 56, 56),
    "Conv1x1_Misaligned_C33_K64": (1, 33, 56, 56),
    # dwconv
    "DWConv_C16_3x3":             (1, 16, 112, 112),
    "DWConv_C960_3x3":            (1, 960, 7, 7),
    # matmul
    "MatMul_512x512x512":         (1, 512),
    "MatMul_768x768x768":         (1, 768),
    "MatMul_3072x768":            (1, 3072),
    "MatMul_768x3072":            (1, 768),
}

SINGLE_OPS_DIR = MODELS_DIR / "single_ops"


def compile_torchvision_model(model_name: str, cfg: dict, tune: bool = False) -> Path:
    """路径 A: torch.export → Relax → .so"""
    import torch
    import torchvision.models as tv_models

    print(f"\n{'='*60}")
    print(f"[TorchVision] 编译 {model_name} → TVM Relax (.so)")
    print(f"  Target: {TARGET_ARM}")
    print(f"  Input:  {cfg['input_shape']}")
    print(f"{'='*60}")

    # 1. 加载 PyTorch 模型
    print("[1/5] 加载 PyTorch 模型...")
    if cfg["class"] == "mobilenet_v2":
        model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif cfg["class"] == "resnet50":
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model class: {cfg['class']}")
    model = model.eval()

    # 2. torch.export 导出
    print("[2/5] torch.export.export() ...")
    example_input = (torch.randn(*cfg["input_shape"], dtype=torch.float32),)
    with torch.no_grad():
        exported = torch.export.export(model, example_input)

    # 3. 转换为 Relax IR (参数嵌入)
    print("[3/5] from_exported_program() → Relax IRModule ...")
    from tvm.relax.frontend.torch import from_exported_program
    mod = from_exported_program(exported, keep_params_as_input=False)
    func_names = list(mod.functions.keys())
    print(f"  IRModule functions: {func_names}")

    # 4. Relax 编译流水线
    print("[4/5] relax.get_pipeline() + tvm.compile() ...")
    if tune:
        mod = apply_autotune(mod, model_name, cfg["input_shape"])

    pipeline = relax.get_pipeline()
    with TARGET_ARM:
        built_mod = pipeline(mod)
    executable = tvm.compile(built_mod, target=TARGET_ARM)
    print("  编译成功！")

    # 5. 导出 .so
    return _export_so(executable, model_name)


def compile_onnx_model(model_name: str, cfg: dict, tune: bool = False) -> Optional[Path]:
    """路径 B: ONNX → Relax → .so"""
    onnx_path = PROJECT_ROOT / cfg["onnx_path"]
    if not onnx_path.exists():
        print(f"\n⚠️  跳过 {model_name}: ONNX 文件不存在 ({onnx_path})")
        return None

    print(f"\n{'='*60}")
    print(f"[ONNX] 编译 {model_name} → TVM Relax (.so)")
    print(f"  ONNX:   {onnx_path}")
    print(f"  Target: {TARGET_ARM}")
    print(f"  Input:  {cfg['input_shape']}")
    print(f"{'='*60}")

    # 1. 加载 ONNX 模型
    print("[1/4] 加载 ONNX 模型...")
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  输入数: {len(onnx_model.graph.input)}")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")

    # 2. ONNX → Relax IRModule
    print("[2/4] from_onnx() → Relax IRModule ...")
    input_name = cfg.get("input_name", None)
    shape_dict = {input_name: list(cfg["input_shape"])} if input_name else None

    mod = from_onnx(onnx_model, shape_dict=shape_dict)
    func_names = list(mod.functions.keys())
    print(f"  IRModule functions: {func_names}")

    # 3. Relax 编译流水线
    print("[3/4] relax.get_pipeline() + tvm.compile() ...")
    if tune:
        mod = apply_autotune(mod, model_name, cfg["input_shape"])

    pipeline = relax.get_pipeline()
    with TARGET_ARM:
        built_mod = pipeline(mod)
    executable = tvm.compile(built_mod, target=TARGET_ARM)
    print("  编译成功！")

    # 4. 导出 .so
    return _export_so(executable, model_name)


def compile_single_op(op_name: str, input_shape: tuple, tune: bool = False) -> Optional[Path]:
    """编译单个算子的 ONNX 模型"""
    onnx_path = SINGLE_OPS_DIR / f"{op_name}.onnx"
    if not onnx_path.exists():
        print(f"  ⚠️  跳过 {op_name}: ONNX 文件不存在 ({onnx_path})")
        return None

    so_path = OUTPUT_DIR / f"{op_name}_tvm.so"
    if so_path.exists():
        size_mb = so_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {op_name}: 已存在 ({size_mb:.2f} MB), 跳过")
        return so_path

    print(f"  [ONNX→TVM] {op_name} shape={input_shape} ...", end=" ", flush=True)

    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))

        # 获取第一个输入名
        input_name = onnx_model.graph.input[0].name
        shape_dict = {input_name: list(input_shape)}

        mod = from_onnx(onnx_model, shape_dict=shape_dict)

        pipeline = relax.get_pipeline()
        with TARGET_ARM:
            built_mod = pipeline(mod)
        executable = tvm.compile(built_mod, target=TARGET_ARM)

        so_path = _export_so(executable, op_name)
        print(f"OK ({so_path.stat().st_size/1024:.1f} KB)")
        return so_path
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def _export_so(executable, name: str) -> Path:
    """导出 .so 文件 (NDK 交叉编译)"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    so_path = OUTPUT_DIR / f"{name}_tvm.so"

    print(f"[导出] export_library() → {so_path} ...")
    executable.export_library(
        str(so_path),
        workspace_dir=str(OUTPUT_DIR / "workspace"),
        cc=NDK_CXX,
    )

    size_mb = so_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ {so_path.name} ({size_mb:.1f} MB)")
    return so_path


def apply_autotune(mod, model_name: str, input_shape: tuple):
    """TVM MetaSchedule / AutoTVM 自动调优（基础实现）"""
    print(f"  [Tune] Auto-tuning {model_name} ...")
    # MetaSchedule tuning — 对 arm CPU 进行基础调优
    try:
        from tvm import meta_schedule as ms

        # 使用默认的 CPU 调优规则（可能较慢但有效）
        database = ms.tune_relax(
            mod=mod,
            target=TARGET_ARM,
            params={},
            # 限制 trials 数量以控制编译时间
            max_trials_global=32,
            num_trials_per_iter=8,
        )

        # 应用调优结果
        mod = database.apply(mod)
        print(f"  [Tune] Auto-tuning 完成")
        return mod
    except Exception as e:
        print(f"  [Tune] Warning: auto-tuning skipped ({e})")
        return mod


def verify_compiled(so_path: Path, model_name: str, input_shape: tuple):
    """用 TVM runtime 验证编译产物（主机侧）"""
    try:
        lib = tvm.runtime.load_module(str(so_path))
        dev = tvm.cpu(0)
        vm = relax.VirtualMachine(lib, dev)

        data = np.random.randn(*input_shape).astype("float32")
        vm_input = tvm.runtime.tensor(data, dev)
        output = vm["main"](vm_input)

        if isinstance(output, tvm.ir.Array):
            result = output[0]
        else:
            result = output

        print(f"  ✅ 验证通过: output shape={result.shape}, dtype={result.dtype}")
        return True
    except Exception as e:
        print(f"  ⚠️  验证失败 (可能因 NDK 交叉编译无法在主机运行): {e}")
        return False


def print_summary(results: dict):
    """打印编译汇总"""
    print(f"\n{'='*60}")
    print("编译汇总")
    print(f"{'='*60}")
    success = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]

    total_size = 0
    for name in success:
        so_path = results[name]
        if so_path and so_path.exists():
            size_mb = so_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  ✅ {name:40s} {size_mb:7.2f} MB")
    for name in skipped:
        print(f"  ⏭️  {name:40s} (文件缺失, 跳过)")
    for name in failed:
        print(f"  ❌ {name:40s} (编译失败)")

    print(f"\n  总计: {len(success)} 成功, {len(skipped)} 跳过, {len(failed)} 失败")
    print(f"  产物总大小: {total_size:.1f} MB")
    print(f"  产物目录: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="TVM Relax 全量模型编译")
    parser.add_argument("model", nargs="?", default=None,
                        help="模型名 (resnet50/mobilenetv2/yolov8n/bert/mobilevit_s)")
    parser.add_argument("--all", action="store_true", help="编译所有整模型")
    parser.add_argument("--single-ops", action="store_true", help="编译所有单算子")
    parser.add_argument("--tune", action="store_true", help="启用 auto-tuning")
    parser.add_argument("--verify", action="store_true", help="编译后验证")
    args = parser.parse_args()

    results = {}

    # ── 编译整模型 ──
    if args.all or args.model in TORCHVISION_MODELS:
        for name, cfg in TORCHVISION_MODELS.items():
            if args.model and args.model != name:
                continue
            so_path = OUTPUT_DIR / f"{name}_tvm.so"
            try:
                result = compile_torchvision_model(name, cfg, args.tune)
                results[name] = result
                if args.verify and result:
                    verify_compiled(result, name, cfg["input_shape"])
            except Exception as e:
                print(f"  ❌ {name} 编译失败: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

    if args.all or args.model in ONNX_MODELS:
        for name, cfg in ONNX_MODELS.items():
            if args.model and args.model != name:
                continue
            so_path = OUTPUT_DIR / f"{name}_tvm.so"
            try:
                result = compile_onnx_model(name, cfg, args.tune)
                results[name] = result
                if args.verify and result:
                    verify_compiled(result, name, cfg["input_shape"])
            except Exception as e:
                print(f"  ❌ {name} 编译失败: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

    # ── 编译单算子 ──
    if args.single_ops or args.all:
        print(f"\n{'='*60}")
        print(f"编译单算子 ({len(SINGLE_OP_MODELS)} 个)")
        print(f"{'='*60}")
        for op_name, shape in SINGLE_OP_MODELS.items():
            result = compile_single_op(op_name, shape, args.tune)
            results[op_name] = result

    # ── 无参数默认行为 ──
    if not args.all and not args.single_ops and not args.model:
        parser.print_help()
        return

    print_summary(results)


if __name__ == "__main__":
    main()
