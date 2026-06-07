#!/usr/bin/env python3
"""
Compile MobileNetV2 for Android AArch64 using TVM (ONNX version)
Usage: PYTHONPATH=../third_party/tvm/python python compile_tvm_mobilenetv2.py
"""

import os
import sys

# Add TVM to path
TVM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "tvm")
sys.path.insert(0, os.path.join(TVM_ROOT, "python"))

try:
    import tvm
    from tvm import relay
    print(f"TVM version: {tvm.__version__}")
except ImportError as e:
    print(f"TVM Import error: {e}")
    sys.exit(1)

try:
    import onnx
    print(f"ONNX version: {onnx.__version__}")
except ImportError as e:
    print(f"ONNX Import error: {e}")
    print("Install: pip install onnx")
    sys.exit(1)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "classification", "mobilenetv2")
os.makedirs(OUTPUT_DIR, exist_ok=True)
ONNX_MODEL = os.path.join(OUTPUT_DIR, "mobilenetv2.onnx")

def main():
    print("=" * 60)
    print("TVM MobileNetV2 Compiler for Android ARM64")
    print("=" * 60)

    print(f"\nLoading ONNX model: {ONNX_MODEL}")
    onnx_model = onnx.load(ONNX_MODEL)
    print("ONNX model loaded successfully")

    # Input shape - TVM expects NCHW format
    input_shape = (1, 3, 224, 224)
    shape_dict = {"input": input_shape}
    dtype_dict = {"input": "float32"}

    print(f"\nConverting ONNX to Relay IR...")
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)
    print("Relay IR conversion successful")

    # Target: Android ARM64 with NEON
    target = tvm.target.Target("llvm -mtriple=aarch64-linux-android -mattr=+neon")
    print(f"\nTarget: {target}")

    # Build with optimization level 3
    print("\nBuilding model (this may take a few minutes)...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    print("Build successful!")

    # Export as shared library
    output_path = os.path.join(OUTPUT_DIR, "mobilenetv2_TVM.so")
    print(f"\nExporting to: {output_path}")
    lib.export_library(output_path)
    print(f"Model exported successfully!")

    # Verify file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("Compilation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
