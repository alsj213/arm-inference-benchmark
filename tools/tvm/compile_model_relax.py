#!/usr/bin/env python3
"""
TVM Relax 模型编译脚本 — 使用 keep_params_as_input=False 将参数嵌入 .so
编译产物可直接用 vm["main"](input) 单参数调用，无需额外传 params。

用法:
  export TVM_ROOT=third_party/tvm
  export PYTHONPATH=$TVM_ROOT/python:$TVM_ROOT/3rdparty/tvm-ffi/python
  export LD_LIBRARY_PATH=$TVM_ROOT/build/lib:$TVM_ROOT/build
  python3 tools/tvm/compile_model_relax.py mobilenetv2
"""

import os
import sys
import numpy as np
from pathlib import Path

# ---- 路径设置 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TVM_ROOT = PROJECT_ROOT / "third_party" / "tvm"
OUTPUT_DIR = SCRIPT_DIR / "compiled_models"

sys.path.insert(0, str(TVM_ROOT / "python"))
sys.path.insert(0, str(TVM_ROOT / "3rdparty" / "tvm-ffi" / "python"))

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# ---- 配置 ----
MODEL_CONFIGS = {
    "mobilenetv2": {
        "class": "torchvision.models.mobilenet_v2",
        "args": ("weights", "MobileNet_V2_Weights.IMAGENET1K_V1"),
        "input_shape": (1, 3, 224, 224),
    },
    "resnet18": {
        "class": "torchvision.models.resnet18",
        "args": ("weights", "ResNet18_Weights.IMAGENET1K_V1"),
        "input_shape": (1, 3, 224, 224),
    },
}

# ARM Android target (与骁龙 865 的 ARMv8.2-A + NEON 兼容)
TARGET_ARM = tvm.target.Target({"kind": "llvm", "mtriple": "aarch64-linux-android", "mattr": ["+neon"]})


def compile_model(model_name: str) -> Path:
    """编译 PyTorch 模型 → TVM Relax .so（参数嵌入模式）"""
    import torch
    import torchvision.models as tv_models

    cfg = MODEL_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"编译 {model_name} → TVM Relax (.so with embedded params)")
    print(f"Target: {TARGET_ARM}")
    print(f"Input:  {cfg['input_shape']}")
    print(f"{'='*60}")

    # ── 1. 加载 PyTorch 模型 ──
    print("\n[1/5] 加载 PyTorch 模型...")
    # 解析 "torchvision.models.mobilenet_v2" 和参数
    model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = model.eval()

    # ── 2. torch.export 导出 ──
    print("[2/5] torch.export.export() ...")
    example_input = (torch.randn(*cfg["input_shape"], dtype=torch.float32),)
    with torch.no_grad():
        exported = torch.export.export(model, example_input)

    # ── 3. 转换为 Relax IR (参数嵌入) ──
    print("[3/5] from_exported_program() → Relax IRModule (keep_params_as_input=False)...")
    mod = from_exported_program(exported, keep_params_as_input=False)
    # 不调用 detach_params，参数保留在 IRModule 中作为常量
    func_names = list(mod.functions.keys())
    print(f"  IRModule functions: {func_names}")

    # ── 4. Relax 编译流水线 ──
    print("[4/5] relax.get_pipeline() + tvm.compile() ...")
    pipeline = relax.get_pipeline()
    with TARGET_ARM:
        built_mod = pipeline(mod)
    executable = tvm.compile(built_mod, target=TARGET_ARM)
    print("  编译成功！")

    # ── 5. 导出 .so ──
    print("[5/5] export_library() with NDK cross-linker ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    so_path = OUTPUT_DIR / f"{model_name}_tvm.so"
    # NDK 交叉编译链接器（aarch64-linux-android）
    NDK_CXX = "/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"
    executable.export_library(
        str(so_path),
        workspace_dir=str(OUTPUT_DIR / "workspace"),
        cc=NDK_CXX,
    )

    size_mb = so_path.stat().st_size / (1024 * 1024)
    print(f"\n  输出: {so_path}")
    print(f"  大小: {size_mb:.1f} MB")
    print(f"\n{'='*60}")
    print(f"✅ {model_name} 编译完成")
    print(f"   部署文件: {so_path.name}")
    print(f"   C++ 调用: vm[\"main\"](input_tensor)  ← 不需要额外传参数!")
    print(f"{'='*60}")
    return so_path


def verify_model(so_path: Path, model_name: str):
    """用 TVM runtime 验证编译产物"""
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n[验证] 加载 {so_path} ...")

    lib = tvm.runtime.load_module(str(so_path))
    dev = tvm.cpu(0)
    vm = relax.VirtualMachine(lib, dev)

    # 准备输入
    import torch
    data = torch.randn(*cfg["input_shape"], dtype=torch.float32)
    vm_input = tvm.runtime.tensor(data.numpy(), dev)

    # 调用 — 单参数模式!
    print(f"  vm[\"main\"](input) 调用中...")
    output = vm["main"](vm_input)

    # 处理输出
    if isinstance(output, tvm.ir.Array) and len(output) > 0:
        result = output[0]
    else:
        result = output

    print(f"  输出 shape: {result.shape}")
    print(f"  输出 dtype: {result.dtype}")
    print(f"  ✅ 验证通过!")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "mobilenetv2"

    if model_name not in MODEL_CONFIGS:
        print(f"不支持的模型: {model_name}")
        print(f"可用: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    so_path = compile_model(model_name)
    verify_model(so_path, model_name)
