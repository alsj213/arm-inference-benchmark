#!/usr/bin/env python3
"""MetaSchedule RPC 调优 + 编译 → .so 一键脚本"""
import os, sys, time, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TVM_ROOT = PROJECT_ROOT / "third_party" / "tvm"
OUTPUT_DIR = SCRIPT_DIR / "compiled_models"

sys.path.insert(0, str(TVM_ROOT / "python"))
sys.path.insert(0, str(TVM_ROOT / "3rdparty" / "tvm-ffi" / "python"))

# RPC 环境变量
os.environ.setdefault("TVM_TRACKER_HOST", "127.0.0.1")
os.environ.setdefault("TVM_TRACKER_PORT", "9190")
os.environ.setdefault("TVM_TRACKER_KEY", "snapdragon865")

import tvm
from tvm import relax
from tvm.relax.frontend.onnx.onnx_frontend import from_onnx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.builder import LocalBuilder
import onnx
import numpy as np

TARGET_ARM = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "aarch64-linux-android",
    "mattr": ["+neon"],
    "mcpu": "cortex-a77",
    "num-cores": 8,
})

NDK_CXX = "/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"

MODELS = {
    "mobilenetv2": {
        "onnx": "models/classification/mobilenetv2/mobilenetv2.onnx",
        "input_shape": [1, 3, 224, 224],
    },
    "resnet50": {
        "onnx": "models/classification/resnet50/resnet50.onnx",
        "input_shape": [1, 3, 224, 224],
    },
    "yolov8n": {
        "onnx": "models/detection/yolov8n/yolov8n.onnx",
        "input_shape": [1, 3, 640, 640],
        "input_name": "images",
    },
    "mobilevit_s": {
        "onnx": "models/classification/mobilevit_s/mobilevit_s.onnx",
        "input_shape": [1, 3, 256, 256],
        "input_name": "input",
    },
}


def tune_and_compile(model_name: str, max_trials: int = 800):
    cfg = MODELS[model_name]
    onnx_path = PROJECT_ROOT / cfg["onnx"]
    if not onnx_path.exists():
        print(f"❌ ONNX 不存在: {onnx_path}")
        return None

    work_dir = str(OUTPUT_DIR / "tuning_logs" / model_name)
    os.makedirs(work_dir, exist_ok=True)

    print("=" * 60)
    print(f"  MetaSchedule RPC 调优: {model_name}")
    print(f"  ONNX:     {onnx_path}")
    print(f"  Target:   aarch64-linux-android (Cortex-A77, 8核)")
    print(f"  Trials:   {max_trials}")
    print(f"  Start:    {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    # ── 1. 加载 ONNX ──
    print("\n[1/4] 加载 ONNX → Relax IRModule ...")
    onnx_model = onnx.load(str(onnx_path))
    input_name = cfg.get("input_name")
    if input_name is None:
        input_name = onnx_model.graph.input[0].name
    shape_dict = {input_name: cfg["input_shape"]}
    mod = from_onnx(onnx_model, shape_dict=shape_dict)

    func_count = len(mod.functions)
    print(f"  函数数: {func_count}")

    # ── 2. MetaSchedule 调优 ──
    print(f"\n[2/4] MetaSchedule 调优 (max_trials={max_trials}) ...")
    t0 = time.time()

    try:
        database = ms.relax_integration.tune_relax(
            mod=mod,
            target=TARGET_ARM,
            params={},
            work_dir=work_dir,
            max_trials_global=max_trials,
            num_trials_per_iter=4,   # WSL2 内存有限，降低并行度
            strategy="evolutionary",
            cost_model="xgb",
            space="post-order-apply",
            runner="rpc",
            builder=LocalBuilder(
                timeout_sec=600,    # 大算子编译需 5+ 分钟
                max_workers=2,      # 限制并行编译进程，避免 OOM
            ),
        )
        t1 = time.time()
        print(f"\n  调优耗时: {(t1-t0)/60:.1f} 分钟")

        # 应用调优结果
        print("[3/4] 应用调优结果 ...")
        from tvm.relax.transform import MetaScheduleApplyDatabase
        with TARGET_ARM:
            mod = MetaScheduleApplyDatabase(work_dir)(mod)
        print("  ✅ 调优结果已应用")

    except Exception as e:
        print(f"\n  ❌ 调优失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ── 3. 编译 + 导出 .so ──
    print("[4/4] 编译 Relax pipeline → .so ...")
    t2 = time.time()

    pipeline = relax.get_pipeline()
    with TARGET_ARM:
        built_mod = pipeline(mod)
    executable = tvm.compile(built_mod, target=TARGET_ARM)

    so_path = OUTPUT_DIR / f"{model_name}_tvm.so"
    executable.export_library(
        str(so_path),
        workspace_dir=str(OUTPUT_DIR / "workspace"),
        cc=NDK_CXX,
    )

    size_mb = so_path.stat().st_size / (1024 * 1024)
    t3 = time.time()

    print(f"  ✅ 导出: {so_path} ({size_mb:.1f} MB)")
    print(f"  编译耗时: {(t3-t2)/60:.1f} 分钟")

    print("\n" + "=" * 60)
    print(f"  ✅ {model_name} 调优+编译完成!")
    print(f"  总耗时: {(t3-t0)/60:.1f} 分钟")
    print("=" * 60)

    return so_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="模型名: mobilenetv2, resnet50, yolov8n, mobilevit_s")
    parser.add_argument("--trials", type=int, default=800, help="调优 trials 数 (default: 800)")
    args = parser.parse_args()

    if args.model not in MODELS:
        print(f"未知模型: {args.model}, 可用: {list(MODELS.keys())}")
        sys.exit(1)

    result = tune_and_compile(args.model, args.trials)
    if result:
        print(f"\n产物: {result}")
    else:
        sys.exit(1)
