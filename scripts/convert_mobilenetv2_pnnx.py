#!/usr/bin/env python3
"""
Convert MobileNetV2 from PyTorch to ncnn format using PNNX
This gives better conversion results than onnx2ncnn
"""

import torch
import torchvision.models as models
import pnnx

# Load pretrained MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()

# Example input
x = torch.randn(1, 3, 224, 224)

# Export via PNNX
pnnx.export(
    model,
    "mobilenetv2.pt",
    x,
    input_shapes=[1, 3, 224, 224],
    input_types="torch.float32",
    ncnnparam="mobilenetv2.ncnn.param",
    ncnnbin="mobilenetv2.ncnn.bin",
    fp16=False
)

print("PNNX conversion complete!")
print("Output files:")
print("  mobilenetv2.ncnn.bin")
print("  mobilenetv2.ncnn.param")
