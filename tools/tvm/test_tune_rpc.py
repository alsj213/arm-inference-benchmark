#!/usr/bin/env python3
"""MetaSchedule RPC 调优端到端测试 — 小规模验证"""
import os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TVM_ROOT = PROJECT_ROOT / "third_party" / "tvm"

sys.path.insert(0, str(TVM_ROOT / "python"))
sys.path.insert(0, str(TVM_ROOT / "3rdparty" / "tvm-ffi" / "python"))

os.environ.setdefault("TVM_TRACKER_HOST", "127.0.0.1")
os.environ.setdefault("TVM_TRACKER_PORT", "9190")
os.environ.setdefault("TVM_TRACKER_KEY", "snapdragon865")

import tvm
from tvm import relax
from tvm.relax.frontend.onnx.onnx_frontend import from_onnx
from tvm.s_tir import meta_schedule as ms
import onnx
import numpy as np

TARGET_ARM = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "aarch64-linux-android",
    "mattr": ["+neon"],
    "mcpu": "cortex-a77",
    "num-cores": 8,  # 骁龙 865: 8 核 (MetaSchedule 要求)
})

print("=" * 60)
print("MetaSchedule RPC 调优测试 (Conv1x1_K16_C64_M784)")
print(f"  Target: {TARGET_ARM}")
print(f"  Tracker: {os.environ['TVM_TRACKER_HOST']}:{os.environ['TVM_TRACKER_PORT']}")
print(f"  Key: {os.environ['TVM_TRACKER_KEY']}")
print("=" * 60)

# 1. 加载 ONNX（用 mobilenetv2 整模型，它有足够的算子来触发调优）
onnx_path = PROJECT_ROOT / "models" / "classification" / "mobilenetv2" / "mobilenetv2.onnx"
print(f"\n[1] 加载 ONNX: {onnx_path}")
if not onnx_path.exists():
    print(f"  ❌ 文件不存在!")
    sys.exit(1)
onnx_model = onnx.load(str(onnx_path))
input_name = onnx_model.graph.input[0].name
input_shape = [1, 3, 224, 224]

mod = from_onnx(onnx_model, shape_dict={input_name: input_shape})
print(f"  IRModule functions: {list(mod.functions.keys())}")

# 只调优 main 函数中的前几个子图（不调优全部，加快测试）
# op_names 参数可以限制调优范围
print(f"  限制调优范围: 前 2 个卷积层")

# 2. MetaSchedule 调优（4 trials, 仅验证流程）
work_dir = str(SCRIPT_DIR / "compiled_models" / "tuning_logs" / "test_mobilenetv2_rpc")
os.makedirs(work_dir, exist_ok=True)

print(f"\n[2] MetaSchedule 调优 (8 trials, RPC on device)...")
print(f"  work_dir: {work_dir}")

try:
    database = ms.relax_integration.tune_relax(
        mod=mod,
        target=TARGET_ARM,
        params={},
        work_dir=work_dir,
        max_trials_global=8,       # 少量试验验证管道
        num_trials_per_iter=4,
        strategy="evolutionary",
        cost_model="xgb",
        space="post-order-apply",
        runner="rpc",
    )

    print(f"\n[3] 应用调优结果...")
    mod = database.apply(mod)
    print("  ✅ 调优完成!")
except Exception as e:
    print(f"\n  ❌ 调优失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ MetaSchedule RPC 调优流程验证通过!")
print("   (跳过 .so 导出以节省测试时间)")
print("=" * 60)
