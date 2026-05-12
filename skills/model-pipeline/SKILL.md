---
name: model-pipeline
description: Use when needing to prepare models for benchmark testing — download pretrained ONNX models, convert to MNN format, or verify model files in correct directory structure
---

# Model Pipeline

## Overview

管理 benchmark 测试所需模型的下载、转换和路径验证流程。当前仅覆盖 MNN 和 ONNX Runtime。

## 已支持的模型

| 模型 | 分类 | MNN | ORT | 状态 |
|------|------|-----|-----|------|
| MobileNetV2 | classification | .mnn | .onnx | 就绪 |
| ResNet50 | classification | .mnn | .onnx | 就绪 |
| ShuffleNetV2 x0.5 | classification | .mnn | .onnx | 就绪 |
| MobileViT-S | classification | — | — | 待下载 |
| YOLOv8n | detection | .mnn | .onnx/.ort | 就绪 |
| BERT | nlp | — | — | 待下载 |

## 模型目录结构

```
models/
├── classification/
│   ├── mobilenetv2/        # .onnx, .mnn
│   ├── resnet50/           # .onnx, .mnn
│   ├── shufflenet_v2/      # .onnx, .mnn
│   ├── squeezenet/         # 仅 .tnn（已停用）
│   └── mobilevit_s/        # .onnx, .mnn
├── detection/yolov8n/      # .onnx, .ort, .mnn
├── nlp/bert/               # 空（待下载）
└── speech/                 # 空
```

## 工作流

### 1. 下载预训练 ONNX 模型

```bash
cd /home/liu/project/newwork/benchmark
python scripts/download_pretrained.py
```

导出模型：MobileNetV2、ResNet50、ShuffleNetV2、MobileViT-S、YOLOv8n、BERT。

### 2. MNN 模型转换

编译主机侧工具：

```bash
./scripts/build_host_tools.sh
```

转换示例：

```bash
./tools/bin/MNNConvert -f ONNX \
  --modelFile models/classification/mobilenetv2/mobilenetv2.onnx \
  --MNNModel models/classification/mobilenetv2/mobilenetv2_MNN.mnn \
  --bizCode benchmark
```

### 3. ONNX Runtime

ORT 直接使用 `.onnx` 格式，无需转换。YOLOv8n 的 `.ort` 格式已存在于 `models/detection/yolov8n/`，但当前 benchmark 代码使用 `.onnx` 路径加载。

## 常见问题

- **MNNConvert 找不到**: 先运行 `build_host_tools.sh`，产物在 `tools/bin/MNNConvert`
- **模型路径不匹配**: 代码中通过 `ModelInfo` 结构体管理路径，保持目录结构一致
- **模型文件损坏**: 重新运行 `download_pretrained.py`