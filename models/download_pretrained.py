#!/usr/bin/env python3
"""
Download pretrained models and convert them to ONNX format.
After download, you need to convert ONNX to each framework's format.
"""

import os
import urllib.request
import subprocess

# Model URLs - using working mirrors
MODELS = {
    "classification": {
        "mobilenetv2": {
            "onnx_url": "https://huggingface.co/onnx/mobilenet-v2/resolve/main/mobilenet-v2-1.4_224.onnx",
            "onnx_name": "mobilenetv2.onnx"
        },
        "resnet50": {
            "onnx_url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx",
            "onnx_name": "resnet50.onnx"
        }
    },
    "detection": {
        "yolov8n": {
            "onnx_url": "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.onnx",
            "onnx_name": "yolov8n.onnx"
        }
    },
    "nlp": {
        "bert": {
            "onnx_url": "https://huggingface.co/optimum/bert-base-uncased-onnx/resolve/main/bert-base-uncased.onnx",
            "onnx_name": "bert.onnx"
        }
    }
}

def download_file(url, dest_path):
    """Download file with progress"""
    if os.path.exists(dest_path):
        print(f"Already exists: {dest_path}")
        return True

    print(f"Downloading: {url} -> {dest_path}")
    try:
        urllib.request.urlretrieve(url, dest_path);
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for category, models in MODELS.items():
        for model_name, model_info in models.items():
            out_dir = os.path.join(base_dir, category, model_name)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, model_info["onnx_name"])
            download_file(model_info["onnx_url"], out_path)

    print("\nDownload complete!")
    print("\nNext steps:")
    print("1. Convert ONNX to each framework's format using the framework tools")
    print("2. For ncnn: use onnx2ncnn")
    print("3. For MNN: use onnx2mnn")
    print("4. For TNN: use onnx2tnn")
    print("5. For TFLite: use tf converter")
    print("6. For QNN: use qnn-onnx-converter")

if __name__ == "__main__":
    main()
