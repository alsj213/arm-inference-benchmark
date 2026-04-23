#!/usr/bin/env python3
"""
TVM Model Cross-Compiler for Android AArch64
Standardized & reproducible TVM model build flow

Usage:
    PYTHONPATH=../third_party/tvm/python python build_tvm_model.py mobilenetv2
"""
import os
import sys
import json
import numpy as np

# Add TVM to path
TVM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "tvm")
sys.path.insert(0, os.path.join(TVM_ROOT, "python"))
sys.path.insert(0, os.path.join(TVM_ROOT, "3rdparty", "tvm-ffi", "python"))

# Set library path for TVM
os.environ["LD_LIBRARY_PATH"] = f"{TVM_ROOT}/build/lib:{TVM_ROOT}/build:" + os.environ.get("LD_LIBRARY_PATH", "")

try:
    import tvm
    from tvm import relay
    print(f"TVM version: {tvm.__version__}")
except ImportError as e:
    print(f"TVM import failed: {e}")
    print("Falling back to ONNX + optimized C++ kernel generation")
    sys.exit(1)

import torch
import torchvision.models as models


def build_mobilenetv2(target="llvm -mtriple=aarch64-linux-android"):
    """Build MobileNetV2 for Android AArch64"""
    print("\n=== Building MobileNetV2 for TVM ===")
    print(f"Target: {target}")

    # Load model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = model.eval()

    # Input shape
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)

    # Trace and convert to TVM Relay
    print("Tracing model...")
    scripted_model = torch.jit.trace(model, input_data).eval()

    # Convert to Relay using PyTorch frontend
    print("Converting to TVM Relay...")
    shape_list = [("input", input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # Build with optimization level 3
    print(f"Building for {target} (opt_level=3)...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    print("Build successful!")
    return lib


def save_lib(lib, output_dir, model_name):
    """Save TVM compiled library in benchmark-compatible format"""
    os.makedirs(output_dir, exist_ok=True)

    # Save .so for Android
    lib_path = os.path.join(output_dir, f"{model_name}_TVM.so")
    lib.export_library(lib_path)
    print(f"Exported: {lib_path}")

    # Save metadata for benchmark
    meta = {
        "model": model_name,
        "input_name": "input",
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 1000],
        "tvm_version": tvm.__version__,
    }

    meta_path = os.path.join(output_dir, f"{model_name}_TVM.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported: {meta_path}")

    return lib_path


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "mobilenetv2"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "models", "classification", model_name)

    if model_name == "mobilenetv2":
        lib = build_mobilenetv2()
        path = save_lib(lib, output_dir, model_name)
        print(f"\n=== TVM model build complete! ===")
        print(f"Push to device:")
        print(f"  adb push {path} /data/local/tmp/models/classification/mobilenetv2/")
    else:
        print(f"Model {model_name} not supported yet")
        sys.exit(1)


if __name__ == "__main__":
    main()
