#!/usr/bin/env python3
"""
Download pretrained models in ONNX format for benchmarking.
Downloads: MobileNetV2, ResNet50, YOLOv8n, BERT
"""
import os
import sys
import subprocess

try:
    import torch
    import torchvision.models as models
    import onnx
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    print("Install: pip install torch torchvision transformers")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")


def export_mobilenetv2():
    """Export MobileNetV2 to ONNX."""
    output_dir = os.path.join(MODELS_ROOT, "classification", "mobilenetv2")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mobilenetv2.onnx")

    if os.path.exists(output_path):
        print(f"MobileNetV2 already exists: {output_path}")
        return

    print("Exporting MobileNetV2 to ONNX...")
    model = models.mobilenet_v2(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Success: {output_path}")


def export_resnet50():
    """Export ResNet50 to ONNX."""
    output_dir = os.path.join(MODELS_ROOT, "classification", "resnet50")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "resnet50.onnx")

    if os.path.exists(output_path):
        print(f"ResNet50 already exists: {output_path}")
        return

    print("Exporting ResNet50 to ONNX...")
    model = models.resnet50(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Success: {output_path}")


def export_yolov8n():
    """Export YOLOv8n to ONNX using ultralytics."""
    output_dir = os.path.join(MODELS_ROOT, "detection", "yolov8n")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "yolov8n.onnx")

    if os.path.exists(output_path):
        print(f"YOLOv8n already exists: {output_path}")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Skipping YOLOv8n.")
        print("Install: pip install ultralytics")
        return

    print("Exporting YOLOv8n to ONNX...")
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", opset=12, simplify=True)

    # Move the exported file
    exported_path = "yolov8n.onnx"
    if os.path.exists(exported_path):
        os.rename(exported_path, output_path)
        print(f"Success: {output_path}")
    else:
        print(f"Warning: Could not find exported YOLOv8n ONNX")


def export_bert():
    """Export BERT-base to ONNX."""
    output_dir = os.path.join(MODELS_ROOT, "nlp", "bert")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "bert.onnx")

    if os.path.exists(output_path):
        print(f"BERT already exists: {output_path}")
        return

    try:
        from transformers import BertModel, BertTokenizer
    except ImportError:
        print("Transformers not installed. Skipping BERT.")
        return

    print("Exporting BERT to ONNX...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    dummy_input = tokenizer(
        "Hello, world!",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"}
        }
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Success: {output_path}")


def main():
    print("=== Downloading Pretrained Models ===")
    print(f"Models directory: {MODELS_ROOT}")
    print()

    os.makedirs(MODELS_ROOT, exist_ok=True)

    export_mobilenetv2()
    export_resnet50()
    export_yolov8n()
    export_bert()

    print()
    print("=== Done ===")
    print()
    print("Next step: Convert models")
    print("  ./scripts/convert_models.sh")


if __name__ == "__main__":
    main()
