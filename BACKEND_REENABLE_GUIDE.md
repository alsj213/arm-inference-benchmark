# 后端重新启用指南

## 概述

ncnn、TFLite、TNN、QNN、TVM 和 llama.cpp 后端的源代码已完整保留在 `src/backends/` 和 `third_party/` 中。
当前只有 **MNN** 和 **ONNX Runtime** 处于活跃状态。要重新启用其他后端，请按以下步骤操作。

## 通用步骤

1. 在根目录 `CMakeLists.txt` 中将对应后端的 `option()` 改为 `ON`
2. 重新编译：`./scripts/build_android.sh`
3. 恢复模型转换（需要对应转换工具）

---

## 各后端详情

### ncnn（腾讯）

| 项目 | 内容 |
|------|------|
| CMake 选项 | `BENCHMARK_NCNN` |
| 第三方依赖 | `third_party/ncnn`（git 子模块，已检出） |
| 模型转换 | `third_party/ncnn/tools/onnx/onnx2ncnn` |
| 恢复步骤 | `scripts/build_host_tools.sh` 中取消注释 ncnn 构建块，然后 `scripts/convert_models.sh` |

### TFLite（Google TensorFlow Lite）

| 项目 | 内容 |
|------|------|
| CMake 选项 | `BENCHMARK_TFLITE` |
| 第三方依赖 | `third_party/tflite_extracted/`（从 AAR 提取的 .so + 头文件） |
| 模型转换 | 需要 TensorFlow + onnx-tf Python 包：`pip install tensorflow onnx onnx-tf` |
| 恢复步骤 | `scripts/run_benchmark_android.sh` 中取消注释 TFLite .so 推送 |

### TNN（字节跳动）

| 项目 | 内容 |
|------|------|
| CMake 选项 | `BENCHMARK_TNN` |
| 第三方依赖 | `third_party/TNN`（git 子模块，已检出） |
| 模型转换 | `third_party/TNN/tools/onnx2tnn/onnx-converter` |
| 注意 | TNN 的 CMake 构建支持不完整，首次启用可能需要补全 CMake 配置 |

### QNN（Qualcomm QNN SDK）

| 项目 | 内容 |
|------|------|
| CMake 选项 | `BENCHMARK_QNN` |
| 第三方依赖 | `third_party/QNN/`（需从 Qualcomm 官网手动下载 SDK） |
| 参考 | `QNN_SETUP_GUIDE.md` |
| 注意 | 需要 Qualcomm 设备（骁龙平台）和 Qualcomm 开发者账号 |

### TVM（Apache TVM）

| 项目 | 内容 |
|------|------|
| CMake 选项 | `BENCHMARK_TVM` |
| 第三方依赖 | `third_party/tvm`（git 子模块，已检出） |
| 运行时 | 需要先编译 `libtvm_runtime.so` |
| 恢复步骤 | `scripts/run_benchmark_android.sh` 中取消注释 TVM .so 推送 |

### llama.cpp（LLM 推理）

| 项目 | 内容 |
|------|------|
| CMake 选项 | `BENCHMARK_LLAMACPP` |
| 第三方依赖 | `third_party/llama.cpp`（git 子模块，已检出） |
| 模型格式 | GGUF 格式 |

---

## 完整的多后端配置

原始的多后端完整配置已备份在 `backup/all-backends` 分支中：

```bash
git checkout backup/all-backends
```

对比当前分支与备份分支可查看所有脚本的原始内容。