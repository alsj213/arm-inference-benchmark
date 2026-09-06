---
name: mobile-bench-model-prep
description: Use when preparing models for benchmark testing — download pretrained ONNX models, convert to framework-specific formats, and cross-compile the Android benchmark binary
---

# Mobile Bench Model Prep

## Overview

管理 benchmark 测试所需的模型准备和二进制编译流程：模型下载、格式转换（MNN）、NDK 交叉编译。

配置通过项目根目录的 `.benchmarkrc.yml` 读取（Android NDK 路径等）。

## 模型支持状态

| 模型 | 分类 | ONNX | MNN | ORT | NCNN | TVM | 状态 |
|------|------|------|-----|-----|------|-----|------|
| MobileNetV2 | classification | .onnx | .mnn | ✅ | .param/.bin | .so | 就绪 |
| ResNet50 | classification | .onnx | .mnn | ✅ | .param/.bin | — | 就绪 |
| ResNet18 | classification | .onnx | — | ✅ | — | — | 就绪 |
| ShuffleNetV2 | classification | .onnx | — | ✅ | — | — | 就绪 |
| SqueezeNet | classification | .onnx | — | ✅ | — | — | 就绪 |
| EfficientNet-Lite0 | classification | .onnx | — | ✅ | — | — | 就绪 |
| MobileViT-S | classification | .onnx | — | ✅ | — | — | 就绪 |
| YOLOv8n | detection | .onnx | .mnn | ✅ | — | — | 就绪 |
| BERT | nlp | .onnx | — | ✅ | — | — | 就绪 |
| BERT-Base | nlp | .onnx | — | ✅ | — | — | 就绪 |
| Qwen2-0.5B | nlp (LLM) | .onnx | — | ✅ | — | — | 就绪 |
| Qwen2-1.5B | nlp (LLM) | .onnx | — | ✅ | — | — | 就绪 |
| Qwen2.5-1.5B | nlp (LLM) | .onnx | — | ✅ | — | — | 就绪 |

## 模型目录结构

```
models/
├── classification/
│   ├── mobilenetv2/        # .onnx, .mnn
│   ├── resnet50/           # .onnx, .mnn
│   ├── resnet18/           # .onnx
│   ├── shufflenet_v2/      # .onnx
│   ├── squeezenet/         # .onnx
│   ├── efficientnet_lite0/ # .onnx
│   └── mobilevit_s/        # .onnx
├── detection/
│   └── yolov8n/            # .onnx, .ort, .mnn
├── nlp/
│   ├── bert/               # .onnx
│   ├── bert_base/          # .onnx
│   ├── qwen2_0.5b/         # .onnx
│   ├── qwen2_1.5b/         # .onnx
│   └── qwen2.5_1.5b/       # .onnx
├── speech/                 # 语音模型
└── single_ops/             # 单算子测试
```

## 下载预训练 ONNX 模型

```bash
python3 scripts/convert/download-pretrained.py
```

支持导出的模型：MobileNetV2、ResNet50、YOLOv8n、BERT。

## MNN 模型转换

### 编译主机侧工具

```bash
./scripts/build/build-host-tools.sh
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

ORT 直接使用 `.onnx` 格式，无需转换。

### TVM 模型编译

TVM 使用 Relax 编译管道将模型编译为 `.so` 动态库（真实入口在 `tools/tvm/`，参数已内嵌进脚本 config）：

```bash
# 单模型 Relax 编译（产物 tools/tvm/compiled_models/，keep_params_as_input=False 参数已嵌入 .so）
export TVM_ROOT=third_party/tvm
export PYTHONPATH=$TVM_ROOT/python:$TVM_ROOT/3rdparty/tvm-ffi/python
export LD_LIBRARY_PATH=$TVM_ROOT/build/lib:$TVM_ROOT/build
python3 tools/tvm/compile_model_relax.py mobilenetv2   # 支持列表见脚本 MODEL_CONFIGS
# 批量/调优编译见 tools/tvm/compile_all_models.py / tune_and_compile.py
```

运行时通过 `TVMModuleLoadFromFile` 加载编译出的 `.so`。

### MindSpore Lite 模型转换

MindSpore Lite 使用 `.ms` 格式，需通过 MindSpore Lite Converter 工具将 ONNX 模型转换（converter_lite 为**外部 MindSpore Lite SDK 工具**，不随本仓提供，路径以实际 SDK 为准）：

```bash
# 使用 MindSpore Lite 提供的 converter_lite 工具
./converter_lite --fmk=ONNX --modelFile=models/classification/mobilenetv2/mobilenetv2.onnx \
  --outputFile=models/classification/mobilenetv2/mobilenetv2.ms
# 注: converter_lite 是 MindSpore Lite SDK 提供的外部工具,不在本仓;示例仅演示调用方式,实际路径以 SDK 安装为准。
```

### llama.cpp GGUF 模型

llama.cpp 使用 GGUF 格式，需将 HuggingFace 模型转换为 GGUF：

```bash
# 使用 llama.cpp 提供的 convert_hf_to_gguf.py
python3 third_party/llama.cpp/convert_hf_to_gguf.py models/nlp/qwen2.5_1.5b/ \
  --outfile models/nlp/qwen2.5_1.5b/qwen2.5_1.5b.gguf
```

## Android NDK 交叉编译

### 前置条件

- `ANDROID_NDK` 环境变量已设置
- 三方依赖子模块已初始化

```bash
# 从 .benchmarkrc.yml 读取 NDK 路径
export ANDROID_NDK=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['ndk']['path'])")
```

### 构建类型

| 类型 | 脚本 | 输出目录 |
|------|------|---------|
| Release | `./scripts/build/build-android.sh` | `build_android/` |
| Debug | `./scripts/build/build-android.sh --debug` | `build_android_debug/` |
| 主机侧工具 | `./scripts/build/build-host-tools.sh` | `build_host_tools/` |

### 编译流程

```bash
# 安装依赖
./scripts/setup/setup-deps.sh

# Release 版本
./scripts/build/build-android.sh

# Debug 版本（profiling 需要符号信息）
./scripts/build/build-android.sh --debug
```

脚本内部执行 cmake 命令：

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-29 \
      -DCMAKE_BUILD_TYPE=Release \
      -DBENCHMARK_MNN=ON \
      -DBENCHMARK_ORT=ON \
      -DBENCHMARK_NCNN=ON \
      -DBENCHMARK_TVM=ON \
      -DBENCHMARK_MINDSPORE_LITE=ON \
      -DBENCHMARK_LLAMACPP=OFF \
      -DBENCHMARK_TFLITE=OFF \
      -DBENCHMARK_TNN=OFF \
      -DBENCHMARK_QNN=OFF
make -j$(nproc)
```

## CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `BENCHMARK_MNN` | ON | MNN 后端（共享库 libMNN.so） |
| `BENCHMARK_ORT` | ON | ONNX Runtime 后端（动态库） |
| `BENCHMARK_NCNN` | ON | ncnn 后端（腾讯） |
| `BENCHMARK_TVM` | ON | Apache TVM 后端（Relax VM） |
| `BENCHMARK_MINDSPORE_LITE` | ON | MindSpore Lite 后端（华为，需手动下载 SDK） |
| `BENCHMARK_LLAMACPP` | OFF | llama.cpp LLM 推理引擎（GGUF 格式） |
| `BENCHMARK_TFLITE` | OFF | TensorFlow Lite（已停用，可恢复） |
| `BENCHMARK_TNN` | OFF | TNN（已停用，可恢复） |
| `BENCHMARK_QNN` | OFF | Qualcomm QNN（已停用，可恢复） |

## 常见问题

- **MNNConvert 找不到**: 先运行 `./scripts/build/build-host-tools.sh`，产物在 `tools/bin/MNNConvert`
- **模型路径不匹配**: 代码中通过 `ModelInfo` 结构体管理路径，保持目录结构一致
- **模型文件损坏**: 重新运行 `scripts/convert/download-pretrained.py`
- **ANDROID_NDK 未设置**: 从 `.benchmarkrc.yml` 读取或手动 `export`
- **ORT 头文件找不到**: 确保 ONNX Runtime 子模块已初始化并单独编译
- **MNN 编译慢**: 首次编译共享库较慢，后续增量编译很快

## 关联 Skill

- [mobile-bench-run](../mobile-bench-run/SKILL.md) — 完整基准测试流程
- [mobile-bench-profiling](../mobile-bench-profiling/SKILL.md) — 性能 profiling
- [mobile-bench-integrate](../mobile-bench-integrate/SKILL.md) — 集成新推理框架后端
