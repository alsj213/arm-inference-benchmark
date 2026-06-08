#!/usr/bin/env python3
"""选择 mobilenetv2 中最关键的 1 个子图做 MetaSchedule RPC 调优 — 低内存安全"""
import os, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TVM_ROOT = PROJECT_ROOT / "third_party" / "tvm"
OUTPUT_DIR = SCRIPT_DIR / "compiled_models"

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

ONNX_PATH = PROJECT_ROOT / "models" / "classification" / "mobilenetv2" / "mobilenetv2.onnx"

# mobilenetv2 中最关键的几个子图（按 FLOP 排序前 3）
# 从之前的调优日志中提取的任务名
KEY_TASKS = {
    "conv1x1_small": "minimum10",    # 47K FLOP, weight=6 — 高频小卷积
    "conv1x1_large": "minimum1",     # 1.2M FLOP, weight=1 — 大瓶颈层
    "dwconv_3x3":   "maximum4",      # 112K FLOP, weight=1 — depthwise
}


def tune_and_compile(op_label: str, op_name: str, trials: int = 32):
    work_dir = str(OUTPUT_DIR / "tuning_logs" / f"mbv2_{op_label}")
    os.makedirs(work_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  {op_label} ({op_name})  trials={trials}")
    print(f"{'='*50}")

    # 1. 加载整模型 ONNX
    print("  加载 ONNX ...")
    onnx_model = onnx.load(str(ONNX_PATH))
    input_name = onnx_model.graph.input[0].name
    mod = from_onnx(onnx_model, shape_dict={input_name: [1, 3, 224, 224]})

    # 2. 只调优指定子图
    t0 = time.time()
    try:
        database = ms.relax_integration.tune_relax(
            mod=mod, target=TARGET_ARM, params={},
            work_dir=work_dir,
            max_trials_global=trials,
            num_trials_per_iter=2,
            op_names=[op_name],           # ← 只调这 1 个子图!
            strategy="evolutionary",
            cost_model="xgb",
            space="post-order-apply",
            runner="rpc",
            builder=LocalBuilder(timeout_sec=600, max_workers=1),
        )
        t1 = time.time()
        print(f"  调优: {(t1-t0)/60:.1f} min ({trials} trials)")

        # v0.24: 用 MetaScheduleApplyDatabase transform 应用调优结果
        from tvm.relax.transform import MetaScheduleApplyDatabase
        with TARGET_ARM:
            mod = MetaScheduleApplyDatabase(work_dir)(mod)
    except Exception as e:
        print(f"  ❌ 调优失败: {e}")
        import traceback; traceback.print_exc()
        return None

    # 3. 编译全模型（带调优结果）
    print("  编译全模型 Relax → .so ...")
    t2 = time.time()
    pipeline = relax.get_pipeline()
    with TARGET_ARM:
        built_mod = pipeline(mod)
    executable = tvm.compile(built_mod, target=TARGET_ARM)

    so_path = OUTPUT_DIR / f"mobilenetv2_tuned_{op_label}_tvm.so"
    ARM_LIB = str(TVM_ROOT / "build_android_rpc")
    executable.export_library(str(so_path),
        workspace_dir=str(OUTPUT_DIR / "workspace"), cc=NDK_CXX,
        options=[
            '-L' + ARM_LIB, '-L' + ARM_LIB + '/lib',
            '-ltvm_ffi', '-ltvm_runtime',
            '-Wl,-rpath,/data/local/tmp/benchmark',
        ])
    t3 = time.time()

    mb = so_path.stat().st_size / (1024*1024)
    print(f"  ✅ {mb:.1f} MB  编译:{(t3-t2)/60:.1f}min  总计:{(t3-t0)/60:.1f}min")
    return so_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("op", nargs="?", default=None,
                       choices=list(KEY_TASKS.keys()),
                       help="conv1x1_small | conv1x1_large | dwconv_3x3")
    args = parser.parse_args()

    if args.op:
        tasks = {args.op: KEY_TASKS[args.op]}
    elif args.all:
        tasks = KEY_TASKS
    else:
        # 默认只调最轻量的
        tasks = {"conv1x1_small": KEY_TASKS["conv1x1_small"]}

    results = {}
    t_total = time.time()
    for label, op_name in tasks.items():
        results[label] = tune_and_compile(label, op_name, args.trials)

    print(f"\n{'='*50}")
    print(f"汇总 ({len(tasks)} 算子, 耗时 {(time.time()-t_total)/60:.1f} min):")
    for label, path in results.items():
        status = f"{path.stat().st_size/1024/1024:.1f} MB" if path else "失败"
        print(f"  {'✅' if path else '❌'} {label}: {status}")
