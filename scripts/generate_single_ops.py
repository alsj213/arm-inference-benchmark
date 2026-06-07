#!/usr/bin/env python3
"""
Generate single-operator ONNX models using PyTorch.
This provides more reliable and correct ONNX export.
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx


# 算子配置 - 简化版
OPERATOR_CONFIGS = {
    # ========== 卷积类 ==========
    "Conv2d_1x1_s1": {
        "class": nn.Conv2d,
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_shape": (1, 64, 56, 56),
    },
    "Conv2d_3x3_s1": {
        "class": nn.Conv2d,
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input_shape": (1, 64, 56, 56),
    },
    "Conv2d_3x3_s2": {
        "class": nn.Conv2d,
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "input_shape": (1, 64, 56, 56),
    },
    "DepthwiseConv_3x3_s1": {
        "class": nn.Conv2d,
        "in_channels": 64,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "groups": 64,
        "input_shape": (1, 64, 56, 56),
    },

    # ========== 池化类 ==========
    "MaxPool2d_2x2_s2": {
        "class": nn.MaxPool2d,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "input_shape": (1, 64, 56, 56),
    },
    "AvgPool2d_2x2_s2": {
        "class": nn.AvgPool2d,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "input_shape": (1, 64, 56, 56),
    },
    "AdaptiveAvgPool": {
        "class": nn.AdaptiveAvgPool2d,
        "output_size": (1, 1),
        "input_shape": (1, 128, 7, 7),
    },

    # ========== 激活函数 ==========
    "ReLU": {
        "class": nn.ReLU,
        "input_shape": (1, 128, 28, 28),
    },
    "ReLU6": {
        "class": nn.ReLU6,
        "input_shape": (1, 128, 28, 28),
    },
    "Sigmoid": {
        "class": nn.Sigmoid,
        "input_shape": (1, 128, 28, 28),
    },
    "Tanh": {
        "class": nn.Tanh,
        "input_shape": (1, 128, 28, 28),
    },
    "LeakyReLU": {
        "class": nn.LeakyReLU,
        "negative_slope": 0.1,
        "input_shape": (1, 128, 28, 28),
    },
    "PReLU": {
        "class": nn.PReLU,
        "num_parameters": 128,
        "input_shape": (1, 128, 28, 28),
    },
    "Hardswish": {
        "class": nn.Hardswish,
        "input_shape": (1, 128, 28, 28),
    },
    "GELU": {
        "class": nn.GELU,
        "input_shape": (1, 128, 28, 28),
    },

    # ========== 归一化 ==========
    "BatchNorm2d": {
        "class": nn.BatchNorm2d,
        "num_features": 128,
        "input_shape": (1, 128, 28, 28),
    },
    "InstanceNorm2d": {
        "class": nn.InstanceNorm2d,
        "num_features": 128,
        "input_shape": (1, 128, 28, 28),
    },
    "LayerNorm": {
        "class": nn.LayerNorm,
        "normalized_shape": (512,),
        "input_shape": (1, 512),
    },

    # ========== 矩阵运算 ==========
    "Linear_512_512": {
        "class": nn.Linear,
        "in_features": 512,
        "out_features": 512,
        "input_shape": (1, 512),
    },
    "Linear_1024_1024": {
        "class": nn.Linear,
        "in_features": 1024,
        "out_features": 1024,
        "input_shape": (1, 1024),
    },

    # ========== 逐元素运算 ==========
    "Add": {
        "class": "ElementWiseAdd",
        "input_shape": (1, 128, 28, 28),
    },
    "Mul": {
        "class": "ElementWiseMul",
        "input_shape": (1, 128, 28, 28),
    },
    "Div": {
        "class": "ElementWiseDiv",
        "input_shape": (1, 128, 28, 28),
    },
    "ElementWise_Pow": {
        "class": "ElementWisePow",
        "input_shape": (1, 128, 28, 28),
    },
    "Sqrt": {
        "class": "ElementWiseSqrt",
        "input_shape": (1, 128, 28, 28),
    },
    "Exp": {
        "class": "ElementWiseExp",
        "input_shape": (1, 128, 28, 28),
    },

    # ========== 形状变换 ==========
    "Upsample_Bilinear": {
        "class": nn.Upsample,
        "scale_factor": 2,
        "mode": "bilinear",
        "align_corners": False,
        "input_shape": (1, 64, 28, 28),
    },
    "Upsample_Nearest": {
        "class": nn.Upsample,
        "scale_factor": 2,
        "mode": "nearest",
        "input_shape": (1, 64, 28, 28),
    },
    "Reshape_Flatten": {
        "class": nn.Flatten,
        "start_dim": 1,
        "input_shape": (1, 128, 7, 7),
    },
    "ConvTranspose_2x2": {
        "class": nn.ConvTranspose2d,
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "input_shape": (1, 64, 28, 28),
    },

    # ========== Padding ==========
    "ReflectionPad2d": {
        "class": nn.ReflectionPad2d,
        "padding": (1, 1, 1, 1),
        "input_shape": (1, 64, 28, 28),
    },
    "ZeroPad2d": {
        "class": nn.ZeroPad2d,
        "padding": (1, 1, 1, 1),
        "input_shape": (1, 64, 28, 28),
    },

    # ========== Reduce ==========
    "ReduceMean_Spatial": {
        "class": "ReduceMean",
        "dim": (2, 3),
        "keepdim": False,
        "input_shape": (1, 128, 28, 28),
    },
    "ReduceSum_Spatial": {
        "class": "ReduceSum",
        "dim": (2, 3),
        "keepdim": False,
        "input_shape": (1, 128, 28, 28),
    },

    # ========== Softmax ==========
    "Softmax": {
        "class": nn.Softmax,
        "dim": 1,
        "input_shape": (1, 1000),
    },

    # ========== 组合算子 ==========
    "Conv_BN_ReLU": {
        "class": "ConvBnReLU",
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input_shape": (1, 64, 56, 56),
    },
    "DWConv_PWConv": {
        "class": "DWConvPWConv",
        "in_channels": 64,
        "dw_channels": 64,
        "pw_channels": 128,
        "input_shape": (1, 64, 56, 56),
    },

    # ===== 形状分桶: Conv1x1 M维极端 =====
    # 访存密集端：M=49 (7×7) — 数据量很小，kernel launch 开销显著
    "Conv1x1_M49_C32_K64": {
        "class": nn.Conv2d, "in_channels": 32, "out_channels": 64,
        "kernel_size": 1, "stride": 1, "padding": 0,
        "input_shape": (1, 32, 7, 7),
    },
    "Conv1x1_M49_C256_K512": {
        "class": nn.Conv2d, "in_channels": 256, "out_channels": 512,
        "kernel_size": 1, "stride": 1, "padding": 0,
        "input_shape": (1, 256, 7, 7),
    },
    # 平衡态：M=784 (28×28) — MobileNetV2 主干
    "Conv1x1_M784_C32_K64": {
        "class": nn.Conv2d, "in_channels": 32, "out_channels": 64,
        "kernel_size": 1, "stride": 1, "padding": 0,
        "input_shape": (1, 32, 28, 28),
    },
    # 计算密集端：M=3136 (56×56) — 大量计算，期望接近峰值算力
    "Conv1x1_M3136_C64_K128": {
        "class": nn.Conv2d, "in_channels": 64, "out_channels": 128,
        "kernel_size": 1, "stride": 1, "padding": 0,
        "input_shape": (1, 64, 56, 56),
    },

    # ===== K维极端 =====
    "Conv1x1_K16_C64_M784": {
        "class": nn.Conv2d, "in_channels": 64, "out_channels": 16,
        "kernel_size": 1, "stride": 1, "padding": 0,
        "input_shape": (1, 64, 28, 28),
    },
    "Conv1x1_K1024_C256_M784": {
        "class": nn.Conv2d, "in_channels": 256, "out_channels": 1024,
        "kernel_size": 1, "stride": 1, "padding": 0,
        "input_shape": (1, 256, 28, 28),
    },

    # ===== NCHW4c 对齐退化 =====
    # C%4≠0: MNN 的 NCHW4c 布局需要 padding → 额外开销
    "Conv1x1_Misaligned_C31_K64": {
        "class": nn.Conv2d, "in_channels": 31, "out_channels": 64,
        "kernel_size": 1, "stride": 1, "padding": 0,
        "input_shape": (1, 31, 56, 56),
    },
    "Conv1x1_Misaligned_C33_K64": {
        "class": nn.Conv2d, "in_channels": 33, "out_channels": 64,
        "kernel_size": 1, "stride": 1, "padding": 0,
        "input_shape": (1, 33, 56, 56),
    },

    # ===== DWConv 极端通道 =====
    # DWConv 大通道（C=960, MobileNetV2 最后一层）
    "DWConv_C960_3x3": {
        "class": nn.Conv2d,
        "in_channels": 960, "out_channels": 960,
        "kernel_size": 3, "stride": 1, "padding": 1,
        "input_shape": (1, 960, 7, 7),
        "groups": 960,
    },
    # DWConv 小通道（C=16）
    "DWConv_C16_3x3": {
        "class": nn.Conv2d,
        "in_channels": 16, "out_channels": 16,
        "kernel_size": 3, "stride": 1, "padding": 1,
        "input_shape": (1, 16, 112, 112),
        "groups": 16,
    },

    # ===== MatMul 方阵 vs 长矩阵对比 =====
    # BERT Attention MatMul (方阵)
    "MatMul_768x768x768": {
        "class": nn.Linear, "in_features": 768, "out_features": 768,
        "input_shape": (1, 768),
    },
    "MatMul_512x512x512": {
        "class": nn.Linear, "in_features": 512, "out_features": 512,
        "input_shape": (1, 512),
    },
    # BERT FFN MatMul (长矩阵)
    "MatMul_768x3072": {
        "class": nn.Linear, "in_features": 768, "out_features": 3072,
        "input_shape": (1, 768),
    },
    "MatMul_3072x768": {
        "class": nn.Linear, "in_features": 3072, "out_features": 768,
        "input_shape": (1, 3072),
    },
}


# 自定义模块
class ElementWiseAdd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x


class ElementWiseMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x


class ElementWiseDiv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (x + 1e-8)


class ElementWisePow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.pow(2)


class ElementWiseSqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.abs().sqrt()


class ElementWiseExp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x / 100).exp()


class ReduceMean(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class ReduceSum(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.sum(dim=self.dim, keepdim=self.keepdim)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWConvPWConv(nn.Module):
    def __init__(self, in_channels, dw_channels, pw_channels):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, dw_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.pwconv = nn.Conv2d(dw_channels, pw_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x


def build_module(config):
    """根据配置构建 PyTorch 模块"""
    cls_or_name = config["class"]

    # 自定义模块
    if isinstance(cls_or_name, str):
        if cls_or_name == "ElementWiseAdd":
            return ElementWiseAdd()
        elif cls_or_name == "ElementWiseMul":
            return ElementWiseMul()
        elif cls_or_name == "ElementWiseDiv":
            return ElementWiseDiv()
        elif cls_or_name == "ElementWisePow":
            return ElementWisePow()
        elif cls_or_name == "ElementWiseSqrt":
            return ElementWiseSqrt()
        elif cls_or_name == "ElementWiseExp":
            return ElementWiseExp()
        elif cls_or_name == "ReduceMean":
            return ReduceMean(dim=config["dim"], keepdim=config.get("keepdim", False))
        elif cls_or_name == "ReduceSum":
            return ReduceSum(dim=config["dim"], keepdim=config.get("keepdim", False))
        elif cls_or_name == "ConvBnReLU":
            return ConvBnReLU(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                padding=config["padding"],
            )
        elif cls_or_name == "DWConvPWConv":
            return DWConvPWConv(
                in_channels=config["in_channels"],
                dw_channels=config["dw_channels"],
                pw_channels=config["pw_channels"],
            )
        else:
            raise ValueError(f"Unknown custom module: {cls_or_name}")

    # PyTorch 内置模块
    kwargs = {k: v for k, v in config.items() if k not in ["class", "input_shape"]}
    return cls_or_name(**kwargs)


def export_to_onnx(model, input_shape, output_path, model_name):
    """导出 PyTorch 模型为 ONNX"""
    dummy_input = torch.randn(*input_shape)

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
    )

    # 验证
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # 统计参数量
    param_count = sum(p.numel() for p in model.parameters())

    return param_count


def main():
    parser = argparse.ArgumentParser(description="Generate single-operator ONNX models")
    parser.add_argument("--output_dir", default="models/single_ops",
                        help="Output directory for generated models")
    parser.add_argument("--ops", nargs="+", help="Specific operators to generate (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ops_to_generate = args.ops if args.ops else OPERATOR_CONFIGS.keys()

    print(f"📦 Generating single-operator ONNX models...")
    print(f"   Output directory: {args.output_dir}")
    print()

    success_count = 0
    fail_count = 0

    for model_name in sorted(ops_to_generate):
        if model_name not in OPERATOR_CONFIGS:
            print(f"⚠️  Unknown operator: {model_name}")
            continue

        config = OPERATOR_CONFIGS[model_name]

        try:
            # 构建模型
            model = build_module(config)
            model.eval()

            # 导出 ONNX
            output_path = os.path.join(args.output_dir, f"{model_name}.onnx")
            param_count = export_to_onnx(model, config["input_shape"], output_path, model_name)

            print(f"✅ {model_name:30s} -> {param_count:8d} parameters")
            success_count += 1

        except Exception as e:
            print(f"❌ {model_name:30s} -> Failed: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print()
    print(f"📊 Summary: {success_count} success, {fail_count} failed")
    print(f"📁 Models saved to: {args.output_dir}")
    print()
    print("Next steps:")
    print("  1. Convert to all framework formats: ./scripts/convert_models.sh")
    print("  2. Run benchmark on device: adb push and run inference")


if __name__ == "__main__":
    main()
