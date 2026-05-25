---
name: benchmark-model-prep
description: Use when preparing models for benchmark testing — download pretrained ONNX models, convert to framework-specific formats, and cross-compile the Android benchmark binary
---

# Benchmark Model Prep

## Overview

管理 benchmark 测试所需的模型准备和二进制编译流程：模型下载、格式转换（MNN）、NDK 交叉编译。

配置通过 `.benchmarkrc.yml` 读取（Android NDK 路径等）。

## 模型支持状态

| 模型 | 分类 | MNN | ORT | 状态 |
|------|------|-----|-----|------|
| MobileNetV2 | classification | .mnn | .onnx | 就绪 |
| ResNet50 | classification | .mnn | .onnx | 就绪 |
| YOLOv8n | detection | .mnn | .onnx/.ort | 就绪 |
| BERT | nlp | — | — | 待下载 |

## 模型目录结构

```
models/
├── classification/
│   ├── mobilenetv2/        # .onnx, .mnn
│   ├── resnet50/           # .onnx, .mnn
│   └── squeezenet/         # 仅 .tnn（已停用）
├── detection/yolov8n/      # .onnx, .ort, .mnn
├── nlp/bert/               # 空（待下载）
└── speech/                 # 空
```

## 下载预训练 ONNX 模型

```bash
cd /home/liu/project/newwork/benchmark
python scripts/download_pretrained.py
```

导出模型：MobileNetV2、ResNet50、YOLOv8n、BERT。

## MNN 模型转换

### 编译主机侧工具

```bash
./scripts/build_host_tools.sh
```

产物在 `tools/bin/MNNConvert`。

### 转换命令

```bash
./tools/bin/MNNConvert -f ONNX \
  --modelFile models/classification/mobilenetv2/mobilenetv2.onnx \
  --MNNModel models/classification/mobilenetv2/mobilenetv2_MNN.mnn \
  --bizCode benchmark
```

### ONNX Runtime

ORT 直接使用 `.onnx` 格式，无需转换。YOLOv8n 的 `.ort` 格式已存在于 `models/detection/yolov8n/`，但当前 benchmark 代码使用 `.onnx` 路径加载。

## Android NDK 交叉编译

### 前置条件

- `ANDROID_NDK` 环境变量已设置
- 三方依赖子模块已初始化

```bash
# 从 .benchmarkrc.yml 读取 NDK 路径
export ANDROID_NDK=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['env']['android_ndk'])")
```

### 构建类型

| 类型 | 脚本 | 输出目录 |
|------|------|---------|
| Release | `./scripts/build_android.sh` | `build_android/` |
| Debug | `./scripts/build_android.sh --debug` | `build_android_debug/` |
| 主机侧工具 | `./scripts/build_host_tools.sh` | `build_host_tools/` |

### 编译流程

```bash
# 安装依赖
./scripts/setup_deps.sh

# Release 版本
./scripts/build_android.sh

# Debug 版本（profiling 需要符号信息）
./scripts/build_android.sh --debug
```

脚本内部执行 cmake 命令：

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-29 \
      -DCMAKE_BUILD_TYPE=Release \
      -DBENCHMARK_MNN=ON \
      -DBENCHMARK_ORT=ON \
      -DBENCHMARK_NCNN=OFF \
      -DBENCHMARK_TFLITE=OFF \
      -DBENCHMARK_TNN=OFF \
      -DBENCHMARK_QNN=OFF \
      -DBENCHMARK_TVM=OFF \
      -DBENCHMARK_LLAMACPP=OFF
make -j$(nproc)
```

## CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `BENCHMARK_MNN` | ON | MNN 后端（共享库 libMNN.so） |
| `BENCHMARK_ORT` | ON | ONNX Runtime 后端（动态库） |
| `BENCHMARK_NCNN` | OFF | ncnn（已停用） |
| `BENCHMARK_TFLITE` | OFF | TFLite（已停用） |
| `BENCHMARK_TNN` | OFF | TNN（已停用） |
| `BENCHMARK_QNN` | OFF | QNN（已停用） |
| `BENCHMARK_TVM` | OFF | TVM（已停用） |
| `BENCHMARK_LLAMACPP` | OFF | llama.cpp（已停用） |

## 常见问题

- **MNNConvert 找不到**: 先运行 `build_host_tools.sh`，产物在 `tools/bin/MNNConvert`
- **模型路径不匹配**: 代码中通过 `ModelInfo` 结构体管理路径，保持目录结构一致
- **模型文件损坏**: 重新运行 `download_pretrained.py`
- **ANDROID_NDK 未设置**: 从 `.benchmarkrc.yml` 读取或手动 `export`
- **ORT 头文件找不到**: 确保 ONNX Runtime 子模块已初始化并单独编译
- **MNN 编译慢**: 首次编译共享库较慢，后续增量编译很快
