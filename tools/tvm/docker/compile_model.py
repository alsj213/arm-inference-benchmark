#!/usr/bin/env python3
"""
TVM Docker 内编译脚本 — ONNX → ARM .so
用法 (Docker 内):
  python3 compile_model.py --model mobilenetv2 --onnx /workspace/models/classification/mobilenetv2/mobilenetv2.onnx
用法 (Docker 外):
  docker run --rm -v $(pwd):/workspace tvm-codegen --model mobilenetv2
"""

import argparse, os, sys, json, time

# ═══ TVM 路径设置 ═══
TVM_HOME = os.environ.get("TVM_HOME", "/tvm")
sys.path.insert(0, os.path.join(TVM_HOME, "python"))
sys.path.insert(0, os.path.join(TVM_HOME, "topi", "python"))
sys.path.insert(0, os.path.join(TVM_HOME, "3rdparty", "tvm-ffi", "python"))
_BUILD = os.path.join(TVM_HOME, "build")
os.environ.setdefault("LD_LIBRARY_PATH", "")
if _BUILD not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = _BUILD + ":" + os.environ["LD_LIBRARY_PATH"]

# ═══ Patch: importlib.metadata + libinfo 绕过 (必须在任何 tvm 导入前) ═══
import importlib.metadata as _im
import ctypes as _ctypes

_BUILD = os.path.join(TVM_HOME, "build")
_LIB_DIR = _BUILD

# ── 1. Patch importlib.metadata.distribution ──
_orig_distribution = _im.distribution
from pathlib import Path as _Path

class _FakeDist:
    """假 Distribution — 提供 libinfo 所需的最小接口"""
    # libinfo 用 dist._path.parent / record_entry 解析路径
    # 所以 _path 需要设为 build 的 子目录，让 parent = build
    _path = _Path(_BUILD) / "dummy"
    def read_text(self, path):
        return f"libtvm_ffi.so,,\n"
    def locate_file(self, path):
        return str(_Path(_BUILD) / path)

def _patched_distribution(package):
    if package in ("apache-tvm-ffi", "tvm-ffi"):
        return _FakeDist()
    return _orig_distribution(package)

_im.distribution = _patched_distribution

# ── 2. 导入 tvm_ffi — FakeDist 会让它找到 build/ 下的 lib ──
import tvm_ffi.libinfo as _libinfo

# ── 3. 兜底：patch _find_library_by_basename 直接给路径 ──
_orig_find = _libinfo._find_library_by_basename
def _direct_find(package, target_name):
    for c in [
        os.path.join(_BUILD, f"lib{target_name}.so"),
        os.path.join(_BUILD, f"lib{target_name}.so.0"),
    ]:
        p = _Path(c)
        if p.exists():
            return p.resolve()
    return _orig_find(package, target_name)
_libinfo._find_library_by_basename = _direct_find

import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

# ── 模型配置 ──
MODEL_CONFIGS = {
    "mobilenetv2": {
        "onnx_path": "models/classification/mobilenetv2/mobilenetv2.onnx",
        "input_name": "input",
        "input_shape": [1, 3, 224, 224],
    },
    "resnet50": {
        "onnx_path": "models/classification/resnet50/resnet50.onnx",
        "input_name": "input",
        "input_shape": [1, 3, 224, 224],
    },
    "bert": {
        "onnx_path": "models/nlp/bert/bert.onnx",
        "input_name": "input_ids",
        "input_shape": [1, 128],
    },
    "yolov8n": {
        "onnx_path": "models/detection/yolov8n/yolov8n.onnx",
        "input_name": "images",
        "input_shape": [1, 3, 640, 640],
    },
}

# ── ARM 编译 target ──
# 骁龙 865: ARMv8.2-A, Cortex-A77, 支持 FP16 + SDOT
ARM_TARGET = (
    "llvm -device=arm_cpu "
    "-mtriple=aarch64-linux-android "
    "-mattr=+v8.2a,+fp-armv8,+neon,+dotprod,+fp16fml"
)
ARM_HOST = "llvm -mtriple=aarch64-linux-android"


def compile_model(model_name, onnx_path, input_name, input_shape, output_dir, opt_level=3):
    """ONNX → Relax → TVM .so (ARM Android)"""
    t_start = time.time()

    # ── 1. 加载 ONNX ──
    print(f"[1/4] Loading ONNX: {onnx_path}")
    mod = from_onnx(onnx_path, keep_params_in_ir=True)
    print(f"  Relax IR imported")

    # ── 2. 编译到 ARM ──
    print(f"[2/4] Building for ARM target (opt_level={opt_level}): {ARM_TARGET}")
    target = tvm.target.Target(ARM_TARGET, host=ARM_HOST)
    ex = relax.build(mod, target=target)
    lib = ex.mod

    # ── 3. 导出 .so ──
    os.makedirs(output_dir, exist_ok=True)
    so_path = os.path.join(output_dir, f"{model_name}_tvm.so")
    print(f"[3/4] Exporting to: {so_path}")
    ex.export_library(so_path, cc="aarch64-linux-android29-clang")

    # ── 4. 元数据 ──
    meta = {
        "model": model_name,
        "input_name": input_name,
        "input_shape": input_shape,
        "target": ARM_TARGET,
        "opt_level": opt_level,
        "tvm_version": tvm.__version__,
        "compile_time_sec": round(time.time() - t_start, 1),
        "files": {
            "so": so_path,
        },
    }
    meta_path = os.path.join(output_dir, f"{model_name}_tvm_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    so_size_mb = os.path.getsize(so_path) / (1024 * 1024)
    print(f"[4/4] Done! ({meta['compile_time_sec']}s, .so = {so_size_mb:.1f} MB)")
    print(f"  Output: {so_path}")
    return so_path


def main():
    parser = argparse.ArgumentParser(description="TVM ONNX → ARM .so compiler")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Model name")
    parser.add_argument("--onnx", default=None,
                        help="ONNX model path (auto-detected from --model)")
    parser.add_argument("--opt-level", type=int, default=3,
                        help="TVM optimization level (default: 3)")
    parser.add_argument("--output-dir", default="tools/tvm/compiled_models",
                        help="Output directory")

    args = parser.parse_args()
    config = MODEL_CONFIGS[args.model]

    onnx_path = args.onnx or config["onnx_path"]
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX not found: {onnx_path}")
        print(f"  Make sure you mounted the project dir: -v $(pwd):/workspace")
        sys.exit(1)

    compile_model(
        model_name=args.model,
        onnx_path=onnx_path,
        input_name=config["input_name"],
        input_shape=config["input_shape"],
        output_dir=args.output_dir,
        opt_level=args.opt_level,
    )


if __name__ == "__main__":
    main()
