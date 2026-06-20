#!/usr/bin/env python3
"""Convert PyTorch classification models to ncnn format via PNNX.

PNNX is the recommended PyTorch->ncnn converter, built into ncnn's tools/pnnx.
It skips ONNX as an intermediate step and applies 300+ optimization passes,
producing smaller and faster ncnn models than onnx2ncnn.

Usage:
    python3 scripts/convert_pnnx.py                       # Convert all models
    python3 scripts/convert_pnnx.py --model mobilenetv2   # Single model
    python3 scripts/convert_pnnx.py --list                # List models
    python3 scripts/convert_pnnx.py --symlinks-only       # Fix symlinks only

Output:
    {model_dir}/{name}.ncnn.param   - ncnn param (PNNX output)
    {model_dir}/{name}.ncnn.bin     - ncnn weights (PNNX output)
    {model_dir}/{name}_ncnn.param   - symlink to .ncnn.param (for model_info.h)
    {model_dir}/{name}_ncnn.bin     - symlink to .ncnn.bin   (for model_info.h)
"""

import argparse
import os
import sys
import warnings
import torch
import torchvision.models as models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_symlinks(model_dir, name):
    """Create symlinks so model_info.h naming matches PNNX output.

    model_info.h expects:  {name}_ncnn.param and {name}_ncnn.bin
    PNNX outputs:          {name}.ncnn.param   and {name}.ncnn.bin

    Creates: {name}_ncnn.param -> {name}.ncnn.param
             {name}_ncnn.bin   -> {name}.ncnn.bin
    """
    for ext in ("param", "bin"):
        src = os.path.join(model_dir, f"{name}.ncnn.{ext}")
        dst = os.path.join(model_dir, f"{name}_ncnn.{ext}")

        if not os.path.exists(src):
            print(f"  WARNING: {src} not found, skipping symlink")
            continue

        # Remove existing symlink or file at dst
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)

        os.symlink(os.path.basename(src), dst)
        print(f"  symlink: {os.path.basename(dst)} -> {os.path.basename(src)}")


def convert_model(name, model_fn, input_shape, model_dir):
    """Convert a PyTorch model to ncnn via PNNX.

    Args:
        name: Model name (e.g. 'mobilenetv2')
        model_fn: Callable returning a torch.nn.Module
        input_shape: Input tensor shape (e.g. [1, 3, 224, 224])
        model_dir: Output directory for model files
    """
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Converting {name}...")
    print(f"{'='*60}")

    try:
        model = model_fn()
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return False

    model.eval()

    # Build example input
    x = torch.rand(*input_shape)

    # PNNX export: the name/path determines output file names
    # PNNX appends .ncnn.param / .ncnn.bin etc. to this prefix
    export_target = os.path.join(model_dir, name)

    try:
        import pnnx
        pnnx.export(model, export_target, x)
        print(f"  PNNX export complete")
    except Exception as e:
        print(f"  ERROR during PNNX export: {e}")
        return False

    # Create symlinks for model_info.h naming convention
    create_symlinks(model_dir, name)

    # List generated ncnn files
    for f in sorted(os.listdir(model_dir)):
        fp = os.path.join(model_dir, f)
        is_link = os.path.islink(fp)
        if is_link or f.endswith((".ncnn.param", ".ncnn.bin", ".pnnx.param", ".pnnx.bin")):
            if os.path.isfile(fp) or is_link:
                size = os.path.getsize(fp) if os.path.isfile(fp) else 0
                if size > 1024 * 1024:
                    s = f"{size / 1024 / 1024:.1f}M"
                elif size > 1024:
                    s = f"{size / 1024:.1f}K"
                else:
                    s = str(size)
                link_info = f" -> {os.readlink(fp)}" if is_link else ""
                print(f"    {f:45s} {s:>8s}{link_info}")

    return True


def get_all_models():
    """Return dict of model_name -> (factory_fn, input_shape, model_dir)."""
    result = {}
    cls_base = os.path.join(BASE_DIR, "models", "classification")

    result["mobilenetv2"] = (
        lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        [1, 3, 224, 224],
        os.path.join(cls_base, "mobilenetv2"),
    )

    result["resnet50"] = (
        lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        [1, 3, 224, 224],
        os.path.join(cls_base, "resnet50"),
    )

    result["shufflenet_v2_x0_5"] = (
        lambda: models.shufflenet_v2_x0_5(
            weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        ),
        [1, 3, 224, 224],
        os.path.join(cls_base, "shufflenet_v2"),
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to ncnn format via PNNX"
    )
    parser.add_argument(
        "--model",
        choices=list(get_all_models().keys()),
        help="Convert only this model",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--symlinks-only",
        action="store_true",
        help="Only recreate symlinks, skip PNNX conversion",
    )
    args = parser.parse_args()

    all_models = get_all_models()

    if args.list:
        print("Available models for PNNX conversion:")
        for name, (_, shape, path) in sorted(all_models.items()):
            rel_path = os.path.relpath(path, BASE_DIR)
            print(f"  {name:25s} shape={str(shape):20s} -> {rel_path}")
        return

    if args.symlinks_only:
        print("Recreating symlinks...")
        for name, (_, _, path) in sorted(all_models.items()):
            if os.path.exists(path):
                print(f"  {name}:")
                create_symlinks(path, name)
        return

    if args.model:
        names = [args.model]
    else:
        names = sorted(all_models.keys())

    results = []
    for name in names:
        fn, shape, path = all_models[name]
        ok = convert_model(name, fn, shape, path)
        results.append((name, ok))

    print(f"\n{'='*60}")
    print("Summary:")
    for name, ok in results:
        status = "OK" if ok else "FAIL"
        print(f"  {name:35s} [{status}]")
    print(f"{'='*60}")

    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
