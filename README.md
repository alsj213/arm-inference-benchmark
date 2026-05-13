# ARM Inference Benchmark

端侧深度学习推理框架性能基准测试项目，针对 ARM 架构手机芯片进行性能对比。

![Platform](https://img.shields.io/badge/platform-Android-lightgrey)
![Architecture](https://img.shields.io/badge/architecture-ARM64-brightgreen)

## 特性

- **多框架支持**: MNN + ONNX Runtime（活跃），ncnn / TFLite / TNN / QNN / TVM / llama.cpp（完整保留，可恢复）
- **性能指标**: 延迟 P50/P90/P99、吞吐量 FPS、初始化时间、峰值内存
- **精度对比**: 以 ONNX Runtime 为标杆，自动计算余弦相似度、平均绝对/相对误差
- **CPU Profiling**: simpleperf 火焰图、atrace/perfetto 系统 trace、MNN/ORT 逐算子分析
- **测试模型**: MobileNetV2 / ResNet50 / YOLOv8n / BERT
- **环境控制**: CPU 锁频 + 缓存清理，保证结果可复现

## 测试平台

| 设备 | 芯片 | CPU | GPU | ISA |
|------|------|-----|-----|-----|
| 红米 K30 Pro | 骁龙 865 (SM8250) | 1×A77@2.84GHz + 3×A77@2.42GHz + 4×A55@1.8GHz | Adreno 650 | ARMv8.2-A, FP16 |

## 基准测试结果

### 4 线程性能 (FP32)

| 模型 | 框架 | Init(ms) | P50(ms) | P90(ms) | FPS | 加速比 |
|------|------|----------|---------|---------|-----|--------|
| **MobileNetV2** | ORT | 49.05 | 18.16 | 18.36 | 55.0 | - |
| **MobileNetV2** | MNN | 29.53 | **8.70** | 8.96 | **114.9** | **2.09x** |
| **ResNet50** | ORT | 388.38 | 84.48 | 86.94 | 11.77 | - |
| **ResNet50** | MNN | 497.68 | **82.54** | 85.27 | 11.95 | **1.02x** |
| **YOLOv8n** | ORT | 73.27 | 106.55 | 108.68 | 9.38 | - |
| **YOLOv8n** | MNN | 105.73 | **79.60** | 86.18 | **12.19** | **1.34x** |

### 精度对比 (MNN vs ORT 标杆)

| 模型 | Cosine Similarity | Mean Abs Error | Mean Rel Error |
|------|-------------------|----------------|----------------|
| **MobileNetV2** | 1.000000 | 0.000002 | 0.0005% |
| **ResNet50** | 1.000000 | 0.000070 | 0.0227% |
| **YOLOv8n** | 1.000000 | 0.000082 | 0.8522% |
| **BERT** | 0.990439 | 0.068855 | — |

### 1 线程性能 (FP32) — NLP 模型

| 模型 | 框架 | Init(ms) | P50(ms) | P90(ms) | FPS |
|------|------|----------|---------|---------|-----|
| **BERT** | ORT | 645.88 | **598.02** | 600.39 | 1.67 |
| **BERT** | MNN | 788.98 | 689.49 | 690.02 | 1.45 |

> 完整结果见 [docs/results_sm8250.md](docs/results_sm8250.md)

## 快速开始

### 前置条件

- Android NDK (本机路径: `/home/liu/android-ndk`)
- Android 设备 (Android 10+, ARM64)
- ADB (Windows 路径: `/mnt/e/andorid/adb/adb.exe`)
- Python 3.8+

### 1. 初始化子模块

```bash
git submodule update --init --recursive
```

### 2. 编译 ONNX Runtime（首次需单独编译）

```bash
cd third_party/onnxruntime
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  -u LD_LIBRARY_PATH -u LDFLAGS -u PKG_CONFIG_PATH \
  ./build.sh \
  --android --android_abi arm64-v8a --android_api 29 \
  --android_sdk_path $ANDROID_SDK --android_ndk_path $ANDROID_NDK \
  --build_shared_lib --config Release --use_nnapi \
  --skip_tests --parallel --skip_submodule_sync
cd ../..
```

产物: `third_party/onnxruntime/build/Android/Release/libonnxruntime.so`

### 3. 编译 benchmark

```bash
export ANDROID_NDK=/home/liu/android-ndk
./scripts/build_android.sh       # Release
# ./scripts/build_android.sh --debug   # Debug（profiling 需要符号信息）
```

产物: `build_android/src/benchmark_inference` (约 1.8MB，MNN 为共享库)

### 4. 下载模型

```bash
python scripts/download_pretrained.py
./scripts/build_host_tools.sh && ./scripts/convert_models.sh
```

### 5. 运行测试

```bash
./scripts/run_benchmark_android.sh --backend mnn --model mobilenetv2 --threads 4 --runs 50
```

脚本自动执行：锁频 → 推送二进制/模型/so → 运行 benchmark → 恢复环境。

### 命令行参数

```
--backend  <mnn|onnxrt|ort|all>  后端 (默认: all)
--model    <模型名|all>           模型 (默认: all)
--precision <fp32|fp16|int8>     精度 (默认: fp32)
--threads  <num>                 线程数 (默认: 1)
--warmup   <num>                 warmup 次数 (默认: 10)
--runs     <num>                 测试次数 (默认: 100)
--help                            帮助
```

支持的模型: `mobilenetv2`, `resnet50`, `yolov8n`, `bert`

## Profiling

```bash
# MNN 逐算子
adb shell "cd /data/local/tmp/benchmark && MNN_PROFILING=1 LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2 --backend mnn"

# simpleperf 火焰图
./scripts/simpleperf_profile.sh --backend mnn --model mobilenetv2 --duration 10

# 集成 profiling 入口
./scripts/profile_benchmark.sh --backend mnn --model mobilenetv2 --profile simpleperf
```

## 项目结构

```
benchmark/
├── CLAUDE.md                  # 项目配置文件（文档入口）
├── CMakeLists.txt             # 顶层 CMake（控制后端开关）
├── src/                       # 源代码
│   ├── main.cpp              # 入口 + 参数解析
│   ├── common/               # 基类、配置、工具函数
│   ├── backends/             # 8 个后端实现
│   ├── models/               # 6 个模型信息定义
│   ├── single_op_benchmark.cpp # 单算子测试
│   └── llm_benchmark.cpp     # LLM 推理测试
├── scripts/                  # 19 个脚本
├── models/                   # 模型文件
├── third_party/              # 第三方依赖（git 子模块）
│   ├── MNN/                 # libMNN.so（共享库）
│   └── onnxruntime/         # libonnxruntime.so（需单独编译）
├── skills/                   # 项目技能文档（8 个 SKILL.md）
├── docs/                     # 文档 + 测试结果
├── results/                  # 测试结果输出
└── tools/                    # MNNConvert 等转换工具
```

## 后端状态

| 框架 | CMake 选项 | 状态 |
|------|-----------|------|
| **MNN** | `BENCHMARK_MNN=ON` | 活跃（共享库 libMNN.so） |
| **ONNX Runtime** | `BENCHMARK_ORT=ON` | 活跃（动态库 libonnxruntime.so） |
| ncnn | `BENCHMARK_NCNN=OFF` | 已停用，可恢复 |
| TFLite | `BENCHMARK_TFLITE=OFF` | 已停用，可恢复 |
| TNN | `BENCHMARK_TNN=OFF` | 已停用，可恢复 |
| QNN | `BENCHMARK_QNN=OFF` | 已停用，可恢复 |
| TVM | `BENCHMARK_TVM=OFF` | 已停用，可恢复 |
| llama.cpp | `BENCHMARK_LLAMACPP=OFF` | 已停用，可恢复 |

## 详细指南

本项目包含 8 个 `skills/` 技能文档，覆盖各操作环节的详细步骤：

- **模型准备**: [skills/model-pipeline/SKILL.md](skills/model-pipeline/SKILL.md)
- **交叉编译**: [skills/android-cross-compile/SKILL.md](skills/android-cross-compile/SKILL.md)
- **运行测试**: [skills/run-benchmark/SKILL.md](skills/run-benchmark/SKILL.md)
- **性能分析**: [skills/performance-profiling/SKILL.md](skills/performance-profiling/SKILL.md)
- **集成框架**: [skills/integrate-framework/SKILL.md](skills/integrate-framework/SKILL.md)
- **设备操作**: [skills/android-device-ops/SKILL.md](skills/android-device-ops/SKILL.md)
- **环境控制**: [skills/test-environment-control/SKILL.md](skills/test-environment-control/SKILL.md)
- **结果处理**: [skills/result-processor/SKILL.md](skills/result-processor/SKILL.md)

## 许可证

MIT License
