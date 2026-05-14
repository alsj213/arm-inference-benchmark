## 角色
你是一个端侧深度学习推理框架部署测试工程师，精通各个推理框架的部署和测试。

## 项目概述

这是一个 **ARM 端侧深度学习推理框架性能基准测试项目**，目标是在同等条件下对比多个推理框架在 Android 设备（骁龙 865 / SM8250）上的推理性能。

## 框架状态

| 框架 | 状态 | CMake 选项 | 依赖方式 | 模型格式 |
|------|------|-----------|---------|---------|
| **ONNX Runtime** | **活跃** | `BENCHMARK_ORT=ON` | git 子模块，单独编译 | `.onnx`（无需转换） |
| **MNN** | **活跃** | `BENCHMARK_MNN=ON` | git 子模块，add_subdirectory | `.mnn`（MNNConvert 转换） |
| ncnn | 已停用 | `BENCHMARK_NCNN=OFF` | git 子模块已检出 | ncnn 格式 |
| TNN | 已停用 | `BENCHMARK_TNN=OFF` | git 子模块已检出 | TNN 格式 |
| TFLite | 已停用 | `BENCHMARK_TFLITE=OFF` | AAR 提取 .so | `.tflite` |
| QNN | 已停用 | `BENCHMARK_QNN=OFF` | 需手动下载 SDK | QNN 格式 |
| TVM | 已停用 | `BENCHMARK_TVM=OFF` | git 子模块已检出 | TVM .so |
| llama.cpp | 已停用 | `BENCHMARK_LLAMACPP=OFF` | git 子模块已检出 | GGUF |

## 项目结构

```
benchmark/
├── src/                  # 核心源代码
│   ├── main.cpp         # 入口 + 参数解析 + 测试编排
│   ├── single_op_benchmark.cpp  # 单算子基准测试
│   ├── llm_benchmark.cpp        # LLM 测试（仅 llama.cpp 启用时）
│   ├── common/           # 公共模块（benchmark基类、配置、工具函数）
│   ├── backends/         # 8个后端实现（完整保留，停用的用编译宏隔离）
│   └── models/           # 5个模型信息定义（MobileNetV2/ResNet50/ShuffleNetV2/YOLOv8n/BERT）
├── scripts/              # 28个脚本（构建/测试/环境管理/模型转换/profiling/报告生成）
├── models/               # 模型文件（onnx/mnn/tflite/ncnn/tnn/tvm）
├── third_party/          # 第三方依赖（git子模块 / 手动下载）
├── cmake/                # Android NDK 工具链
├── skills/               # 项目技能（4基础+4流程，MNN/ORT工作流）
│   │   ├── model-pipeline/
│   │   ├── android-device-ops/
│   │   ├── test-environment-control/
│   │   ├── result-processor/
│   │   ├── android-cross-compile/
│   │   ├── run-benchmark/
│   │   ├── performance-profiling/
│   │   └── integrate-framework/
├── docs/                 # 文档 + 测试结果
├── results/              # 测试结果输出
├── tools/                # 转换工具（MNNConvert、FlameGraph）
├── build_android/        # Android Release 构建
├── build_android_debug/  # Android Debug 构建
└── build_host_tools/     # 主机侧工具构建
```

## 关键源代码文件

- `src/main.cpp` — 程序入口，命令行参数解析，遍历模型/后端执行基准测试
- `src/common/benchmark.h/cpp` — 基准测试基类 + 工厂方法 + 精度对比（余弦相似度）
- `src/common/config.h/cpp` — BenchmarkConfig 配置结构体，BackendType 枚举
- `src/common/utils.h/cpp` — 计时器、内存统计、固定种子随机数生成
- `src/models/model_info.h` — ModelInfo 结构体 + 模型路径管理

## 构建系统

- **Android Release**: `./scripts/build_android.sh` → `build_android/`
- **Android Debug**: `./scripts/build_android.sh --debug` → `build_android_debug/`
- **主机侧工具**: `./scripts/build_host_tools.sh` → `build_host_tools/`
- CMakeLists.txt 根目录定义 option，third_party/ 处理各框架依赖，src/ 生成 benchmark_inference 和 single_op_benchmark 可执行文件
- ONNX Runtime 需要先单独编译（参见 `third_party/CMakeLists.txt` 中 ORT 部分）

## 测试流程

1. **编译**: `build_android.sh` 编译 Android 二进制
2. **设置环境**: `setup_test_environment.sh` 锁 CPU 性能频率 + 清缓存
3. **运行测试**: `run_benchmark_android.sh` 推送二进制/模型/so 到手机执行
4. **恢复环境**: `restore_test_environment.sh` 恢复 CPU 调度器
5. **生成报告**: `generate_report.py` 解析日志生成 Markdown 报告
6. **一键执行**: `build_and_run.sh`

## 精度对比

以 ONNX Runtime 输出作为标杆，对其他后端计算 **余弦相似度** 和 **平均绝对误差**。所有后端的输入数据通过 `fill_random_float`（固定种子 42）保证一致。

## 环境

- WSL2 + ADB 连接红米 K30 Pro（骁龙 865 / SM8250）
- ADB 路径: `/mnt/e/andorid/adb/adb.exe`
- 设备 ID: `b08dee23`
- Android NDK 交叉编译（arm64-v8a, android-29）

## 核心命令

```bash
# Android 编译
./scripts/build_android.sh [--debug]

# 锁频 + 清缓存（自动检测 root 权限）
./scripts/setup_test_environment.sh

# 推送并运行
./scripts/run_benchmark_android.sh [--debug]

# Profiling 集成
./scripts/profile_benchmark.sh

# 模型转换
./scripts/convert_models.sh

# 恢复环境
./scripts/restore_test_environment.sh
```

## 强制规则

1. 每个框架在部署前，一定要参考官方教程
2. 每个框架都有真实的模型转换和在设备上真实的测试数据
3. 完全部署好一个框架后再部署下一个，不要交叉部署
4. 停用的后端源代码完整保留，恢复见 `BACKEND_REENABLE_GUIDE.md`
5. 停用后端的旧完整配置备份在 `backup/all-backends` 分支

## 项目 Agent

- **[benchmark-agent](.claude/agents/benchmark-agent.md)** — 一句话自动执行完整 benchmark 测试。说"benchmark mnn mobilenetv2 --threads 4"即可自动完成编译、推送、锁频、运行、出报告全流程。

## 项目技能

本项目包含以下技能，存储在 `skills/` 目录下：

### 基础技能

- [model-pipeline](skills/model-pipeline/SKILL.md) — 模型下载、格式转换（MNN/ORT）、路径管理。需要准备测试模型时使用
- [android-device-ops](skills/android-device-ops/SKILL.md) — ADB 连接、文件推送/拉取、远程执行。需要与 Android 设备通信时使用
- [test-environment-control](skills/test-environment-control/SKILL.md) — CPU 锁频、缓存清理、环境恢复。需要在测试前保证环境一致性时使用
- [result-processor](skills/result-processor/SKILL.md) — 日志解析、精度对比、报告生成。需要分析测试结果时使用

### 流程技能

- [android-cross-compile](skills/android-cross-compile/SKILL.md) — NDK 交叉编译、Release/Debug 构建。需要编译 Android 二进制时使用
- [run-benchmark](skills/run-benchmark/SKILL.md) — 端侧推理基准测试完整流程。需要在 Android 设备上运行性能测试时使用
- [performance-profiling](skills/performance-profiling/SKILL.md) — 逐算子 profiling、火焰图、系统 trace。需要分析推理瓶颈时使用
- [integrate-framework](skills/integrate-framework/SKILL.md) — 添加新后端或恢复已停用后端的完整步骤。需要集成推理框架时使用