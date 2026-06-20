#!/usr/bin/env python3
"""
Convert ResNet50 from PyTorch to ncnn format using PNNX
"""

import torch
import torchvision.models as models
import pnnx

# Load pretrained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Example input
x = torch.randn(1, 3, 224, 224)

# Export via PNNX
pnnx.export(
    model,
    "resnet50.pt",
    x,
    input_shapes=[1, 3, 224, 224],
    input_types="torch.float32",
    ncnnparam="resnet50.ncnn.param",
    ncnnbin="resnet50.ncnn.bin",
    fp16=False
)

print("PNNX conversion complete!")
print("Output files:")
print("  resnet50.ncnn.bin")
print("  resnet50.ncnn.param")
