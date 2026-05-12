---
name: android-cross-compile
description: Use when compiling benchmark binary for Android ARM64 — configure NDK, build MNN as shared library, prepare ORT dynamic library, select Release/Debug mode
---

# Android Cross Compile

## Overview

将 benchmark 交叉编译到 Android ARM64 (arm64-v8a)。MNN 通过 git 子模块 + add_subdirectory 编译为共享库 (libMNN.so)，ONNX Runtime 为预编译动态库 (libonnxruntime.so)。

## 前置条件

- `ANDROID_NDK` 环境变量已设置（如 `export ANDROID_NDK=~/android-ndk-r25c`）
- 三方依赖子模块已初始化

## 构建类型

| 类型 | 脚本 | 输出目录 |
|------|------|---------|
| Release | `./scripts/build_android.sh` | `build_android/` |
| Debug | `./scripts/build_android.sh --debug` | `build_android_debug/` |
| 主机侧工具 | `./scripts/build_host_tools.sh` | `build_host_tools/` |

## 工作流

### 1. 设置 NDK 环境变量

```bash
export ANDROID_NDK=~/android-ndk-r25c
```

### 2. 安装依赖

```bash
./scripts/setup_deps.sh
```

### 3. 编译 benchmark

```bash
# Release 版本（默认）
./scripts/build_android.sh

# Debug 版本（profiling 需要符号信息）
./scripts/build_android.sh --debug
```

脚本内部执行：
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

### 4. 主机侧工具（MNNConvert）

```bash
./scripts/build_host_tools.sh
```

产物在 `tools/bin/MNNConvert`，用于将 ONNX 模型转换为 MNN 格式。

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

- **ANDROID_NDK 未设置**: `export ANDROID_NDK=~/android-ndk-r25c`，NDK r25c 或更新版本
- **ORT 头文件找不到**: 确保 ONNX Runtime 子模块已初始化（`git submodule update --init third_party/onnxruntime`），并已单独编译出 `libonnxruntime.so`
- **MNN 编译慢**: 首次编译共享库较慢，后续增量编译很快