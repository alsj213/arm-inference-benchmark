#!/usr/bin/env python3
"""
AutoTVM + AutoScheduler RPC 设备实测调优
========================================
RPC 链路: Host tracker(9190) ← ADB reverse ← Device RPC server → ADB forward → Host tuner
支持:
  - AutoTVM: 模板化调优 (ARM conv2d / dense 模板成熟)
  - AutoScheduler (Ansor): 无模板搜索 (更广搜索空间，但耗时更长)

用法:
  export TVM_ROOT=third_party/tvm
  export PYTHONPATH=$TVM_ROOT/python:$TVM_ROOT/topi/python
  export LD_LIBRARY_PATH=$TVM_ROOT/build:$LD_LIBRARY_PATH

  # AutoTVM 调优单算子
  python3 tools/tvm/tune_autotvm.py --autotvm --model mobilenetv2

  # AutoScheduler 调优
  python3 tools/tvm/tune_autotvm.py --autoscheduler --model mobilenetv2

  # 单算子
  python3 tools/tvm/tune_autotvm.py --autotvm --op Conv1x1_K1024_C256_M784
"""

import os, sys, time, argparse, json, tempfile
from pathlib import Path
from typing import Optional, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TVM_ROOT = PROJECT_ROOT / "third_party" / "tvm"
OUTPUT_DIR = SCRIPT_DIR / "compiled_models"
SINGLE_OPS_DIR = PROJECT_ROOT / "models" / "single_ops"

sys.path.insert(0, str(TVM_ROOT / "python"))
sys.path.insert(0, str(TVM_ROOT / "topi" / "python"))

import tvm
from tvm import relay, autotvm, auto_scheduler
from tvm.relay.frontend.onnx import from_onnx
import numpy as np

NDK_CXX = "/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"
TARGET_ARM = "llvm -mtriple=aarch64-linux-android -mattr=+neon -mcpu=cortex-a77"

# RPC 配置
RPC_TRACKER_HOST = "127.0.0.1"
RPC_TRACKER_PORT = 9190
RPC_KEY = "snapdragon865"

# 整模型 ONNX
MODELS = {
    "mobilenetv2": {
        "onnx": PROJECT_ROOT / "models/classification/mobilenetv2/mobilenetv2.onnx",
        "input_shape": [1, 3, 224, 224],
    },
}

# 关键单算子
SINGLE_OPS = {
    "Conv1x1_K16_C64_M784":       (1, 64, 28, 28),
    "Conv1x1_K1024_C256_M784":    (1, 256, 28, 28),
    "DWConv_C960_3x3":            (1, 960, 7, 7),
    "MatMul_512x512x512":         (1, 512),
}


def get_rpc_runner():
    """获取 RPC runner — 连接设备测量"""
    from tvm import rpc
    tracker = rpc.connect_tracker(RPC_TRACKER_HOST, RPC_TRACKER_PORT)
    remote = tracker.request(RPC_KEY, priority=0, session_timeout=600)
    return remote


def tune_autotvm(model_name: str, max_trials: int = 800):
    """AutoTVM 模板化调优 — ARM CPU 模板成熟，适合生产环境"""
    import onnx
    onnx_model = onnx.load(str(MODELS[model_name]["onnx"]))
    input_name = onnx_model.graph.input[0].name
    shape = MODELS[model_name]["input_shape"]

    mod, params = from_onnx(onnx_model, shape={input_name: shape}, freeze_params=True)
    print(f"Loaded {model_name}, functions={list(mod.functions.keys())}")

    # 提取 tuning tasks
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=TARGET_ARM, params=params
    )
    print(f"\nAutoTVM: {len(tasks)} tasks found:")

    # 过滤: 只保留 ARM CPU 任务 (去掉 x86 通用模板)
    arm_tasks = [t for t in tasks if "arm_cpu" in t.name or "arm" in t.name.lower()]
    if not arm_tasks:
        arm_tasks = [t for t in tasks if "x86" not in t.name.lower()]
    print(f"  ARM-relevant tasks: {len(arm_tasks)}")

    # RPC runner
    remote = get_rpc_runner()
    print(f"  Device: {remote.cpu()}")

    # 调优配置
    log_file = str(OUTPUT_DIR / "tuning_logs" / f"{model_name}_autotvm.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="ndk", timeout=120),
        runner=autotvm.RPCRunner(
            RPC_KEY,
            host=RPC_TRACKER_HOST,
            port=RPC_TRACKER_PORT,
            number=5,       # 每次测量跑 5 次取平均
            repeat=3,       # 重复 3 组
            timeout=120,    # 单次测量超时
            min_repeat_ms=150,
        ),
    )

    t0 = time.time()
    for i, task in enumerate(arm_tasks):
        prefix = f"[{i+1}/{len(arm_tasks)}]"
        print(f"\n{prefix} Tuning {task.name} (trials={max_trials})...")

        tuner = autotvm.tuner.XGBTuner(task, loss_type="rank-binary")
        tuner.tune(
            n_trial=min(max_trials, len(task.config_space)),
            early_stopping=max_trials // 10,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(min(max_trials, len(task.config_space))),
                autotvm.callback.log_to_file(log_file),
            ],
        )

    elapsed = time.time() - t0
    print(f"\nAutoTVM tuning done in {elapsed/60:.1f} minutes")

    # 应用最佳配置并编译
    print("Compiling tuned model...")
    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=TARGET_ARM, params=params)

    so_path = OUTPUT_DIR / f"{model_name}_autotvm_tuned_tvm.so"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lib.export_library(str(so_path), cc=NDK_CXX)
    print(f"Tuned model: {so_path} ({so_path.stat().st_size/1024/1024:.1f} MB)")
    return so_path


def tune_autoscheduler(model_name: str, num_trials: int = 1000):
    """AutoScheduler (Ansor) 无模板调优 — 搜索空间更大"""
    import onnx
    onnx_model = onnx.load(str(MODELS[model_name]["onnx"]))
    input_name = onnx_model.graph.input[0].name
    shape = MODELS[model_name]["input_shape"]

    mod, params = from_onnx(onnx_model, shape={input_name: shape}, freeze_params=True)
    print(f"Loaded {model_name}")

    # 提取 search tasks
    print("Extracting tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"], params, target=tvm.target.Target(TARGET_ARM)
    )
    print(f"AutoScheduler: {len(tasks)} tasks")

    # RPC runner
    remote = get_rpc_runner()

    log_file = str(OUTPUT_DIR / "tuning_logs" / f"{model_name}_autoscheduler.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        repeat=3, number=5, timeout=120,
        min_repeat_ms=150,
    )

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,
        runner=auto_scheduler.RPCRunner(
            RPC_KEY,
            host=RPC_TRACKER_HOST,
            port=RPC_TRACKER_PORT,
            timeout=120,
            number=5,
            repeat=3,
        ),
        builder=auto_scheduler.LocalBuilder(
            build_func="ndk", timeout=120
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    t0 = time.time()
    auto_scheduler.tune_generic(tasks, task_weights, tune_option)
    elapsed = time.time() - t0
    print(f"AutoScheduler done in {elapsed/60:.1f} minutes")

    # Compile with best schedules
    print("Compiling tuned model...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3,
                config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=TARGET_ARM, params=params)

    so_path = OUTPUT_DIR / f"{model_name}_autoscheduler_tuned_tvm.so"
    lib.export_library(str(so_path), cc=NDK_CXX)
    print(f"Tuned model: {so_path} ({so_path.stat().st_size/1024/1024:.1f} MB)")
    return so_path


def tune_single_op_autotvm(op_name: str, shape: tuple, max_trials: int = 400):
    """AutoTVM 调优单个算子"""
    onnx_path = SINGLE_OPS_DIR / f"{op_name}.onnx"
    if not onnx_path.exists():
        print(f"  ONNX not found: {onnx_path}")
        return None

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    input_name = onnx_model.graph.input[0].name
    mod, params = from_onnx(onnx_model, shape={input_name: list(shape)}, freeze_params=True)

    tasks = autotvm.task.extract_from_program(
        mod["main"], target=TARGET_ARM, params=params
    )
    arm_tasks = [t for t in tasks if "arm_cpu" in t.name]
    if not arm_tasks:
        arm_tasks = [t for t in tasks if "x86" not in t.name.lower()]

    if not arm_tasks:
        print(f"  No ARM tasks for {op_name}")
        return None

    print(f"  {len(arm_tasks)} tasks: {[t.name for t in arm_tasks]}")

    remote = get_rpc_runner()
    log_file = str(OUTPUT_DIR / "tuning_logs" / f"{op_name}_autotvm.log")

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="ndk", timeout=120),
        runner=autotvm.RPCRunner(
            RPC_KEY, host=RPC_TRACKER_HOST, port=RPC_TRACKER_PORT,
            number=5, repeat=3, timeout=120, min_repeat_ms=150,
        ),
    )

    t0 = time.time()
    for task in arm_tasks:
        tuner = autotvm.tuner.XGBTuner(task, loss_type="rank-binary")
        tuner.tune(
            n_trial=min(max_trials, len(task.config_space)),
            early_stopping=60,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(log_file)],
        )

    elapsed = time.time() - t0
    print(f"  Tuned in {elapsed:.1f}s")

    # Compile
    so_path = OUTPUT_DIR / f"{op_name}_autotvm_tuned_tvm.so"
    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=TARGET_ARM, params=params)
    lib.export_library(str(so_path), cc=NDK_CXX)
    print(f"  Compiled: {so_path.name} ({so_path.stat().st_size/1024:.0f} KB)")
    return so_path


def main():
    parser = argparse.ArgumentParser(description="TVM AutoTVM/AutoScheduler RPC 调优")
    parser.add_argument("--autotvm", action="store_true", help="使用 AutoTVM")
    parser.add_argument("--autoscheduler", action="store_true", help="使用 AutoScheduler")
    parser.add_argument("--model", type=str, default=None, help="整模型名 (mobilenetv2)")
    parser.add_argument("--op", type=str, default=None, help="单算子名")
    parser.add_argument("--trials", type=int, default=800, help="最大 trials 数")
    args = parser.parse_args()

    if not args.autotvm and not args.autoscheduler:
        parser.print_help()
        return

    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}, available: {list(MODELS.keys())}")
            return

        if args.autotvm:
            tune_autotvm(args.model, args.trials)
        if args.autoscheduler:
            tune_autoscheduler(args.model, args.trials // 2)  # AS is slower

    elif args.op:
        if args.op not in SINGLE_OPS:
            print(f"Unknown op: {args.op}, available: {list(SINGLE_OPS.keys())}")
            return

        shape = SINGLE_OPS[args.op]
        if args.autotvm:
            tune_single_op_autotvm(args.op, shape, args.trials)
        if args.autoscheduler:
            print("AutoScheduler for single ops: use full model tuning instead")

    else:
        print("Specify --model or --op")


if __name__ == "__main__":
    main()
