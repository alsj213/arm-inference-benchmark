# Benchmark Skills 设计文档

## 概述

为本 benchmark 项目创建 8 个 skill（4 基础 + 4 流程），覆盖 MNN 和 ONNX Runtime 两个活跃框架的完整工作流。

## 技能架构

```
┌─────────────────────────────────────────────────┐
│                  流程技能                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────┐ │
│  │交叉编译  │ │基准测试  │ │Profiling │ │框架集成│ │
│  │(流程)    │ │(流程)    │ │(流程)    │ │(流程) │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──┬──┘ │
│       │            │            │          │    │
│       ▼            ▼            ▼          ▼    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────┐ │
│  │模型流水线│ │设备运维  │ │环境控制  │ │结果  │ │
│  │(基础)    │ │(基础)    │ │(基础)    │ │处理  │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────┘ │
│                  基础技能                         │
└─────────────────────────────────────────────────┘
```

### 范围约束

当前仅覆盖 **MNN** 和 **ONNX Runtime** 两个活跃框架。ncnn、TNN、TFLite、QNN、TVM、llama.cpp 相关内容不包含，待后续扩展。

## 基础技能（4个）

### 1. model-pipeline — 模型流水线管理

**触发条件**：需要为基准测试准备模型，或为新框架转换模型格式

**MNN/ORT 范围**：
- ONNX 预训练模型下载（`download_pretrained.py`）
- ONNX → MNN 格式转换（`MNNConvert`）
- ORT 直接使用 .onnx（无需转换）
- 模型路径规范（`models/classification/*/`, `models/detection/*/`, `models/nlp/*/`）
- 已支持的 5 个模型：MobileNetV2、ResNet50、ShuffleNetV2、MobileViT-S、YOLOv8n
- BERT 模型待下载状态说明

### 2. android-device-ops — Android 设备运维

**触发条件**：需要推送文件到测试设备或远程执行命令

**MNN/ORT 范围**：
- ADB 连接检查
- 二进制文件推送（`benchmark_inference`、`single_op_benchmark`）
- 动态库推送（`libonnxruntime.so`）
- 模型文件推送（`models/` 目录）
- 远程执行命令
- 结果日志拉取

### 3. test-environment-control — 测试环境控制

**触发条件**：基准测试或 profiling 前需要保证环境一致性

**MNN/ORT 范围**：
- CPU 锁频（performance governor）
- 缓存清理
- 非 root 设备环境设置约束
- 环境恢复

### 4. result-processor — 测试结果处理

**触发条件**：基准测试或 profiling 完成后需要分析结果

**MNN/ORT 范围**：
- 日志解析（运行输出）
- 精度对比（以 ORT 为标杆，余弦相似度）
- Markdown 报告生成（`generate_report.py`）
- profiling 数据提取（MNN_PROFILING 输出、ORT Profiling 输出）
- 结果验证（`validate_results.py`）

## 流程技能（4个）

### 5. android-cross-compile — Android 交叉编译

**触发条件**：需要编译 benchmark 二进制到 Android 设备

**引用基础技能**：android-device-ops

**MNN/ORT 范围**：
- CMakeLists.txt option 控制（`BENCHMARK_MNN=ON`, `BENCHMARK_ORT=ON`）
- NDK 工具链配置（`cmake/android.toolchain.cmake`）
- MNN：git 子模块 + add_subdirectory 静态编译
- ORT：预编译 libonnxruntime.so + 头文件
- Release 构建（`build_android/`）
- Debug 构建（`build_android_debug/`）
- 主机侧工具编译（`build_host_tools.sh`）
- 构建验证

### 6. run-benchmark — 端侧基准测试

**触发条件**：需要在 Android 设备上运行推理性能测试

**引用基础技能**：model-pipeline, android-device-ops, test-environment-control, result-processor

**MNN/ORT 范围**：
- 完整流程编排
- 命令行参数说明
- `run_benchmark_android.sh` 使用
- 尾延迟指标（P50/P90/P99）
- 精度对比（ORT → MNN）

### 7. performance-profiling — 性能 Profiling

**触发条件**：需要分析推理延迟瓶颈或算子级别耗时

**引用基础技能**：android-device-ops, test-environment-control, result-processor

**MNN/ORT 范围**：
- Debug 构建要求
- MNN：`MNN_PROFILING=1` 环境变量逐算子 profiling
- ORT：`EnableProfiling()` API 逐算子 profiling
- simpleperf CPU 采样 + 火焰图
- atrace 系统 trace 采集
- perfetto 综合 trace 采集
- `profile_benchmark.sh` 集成入口

### 8. integrate-framework — 添加/恢复推理框架

**触发条件**：需要将新框架集成到 benchmark 项目中，或恢复已停用的后端

**引用基础技能**：model-pipeline, android-cross-compile

**MNN/ORT 范围**：
- 后端类模板（继承 BenchmarkBackend）
- CMake 集成步骤
- 模型转换和路径规范
- 精度对比验证（以 ORT 为标杆）
- 恢复已停用后端的步骤

## 技能目录结构

```
benchmark/
├── skills/
│   ├── model-pipeline/
│   │   └── SKILL.md
│   ├── android-device-ops/
│   │   └── SKILL.md
│   ├── test-environment-control/
│   │   └── SKILL.md
│   ├── result-processor/
│   │   └── SKILL.md
│   ├── android-cross-compile/
│   │   └── SKILL.md
│   ├── run-benchmark/
│   │   └── SKILL.md
│   ├── performance-profiling/
│   │   └── SKILL.md
│   └── integrate-framework/
│       └── SKILL.md
└── CLAUDE.md          # 引用所有 skill，注册触发条件
```

### 注册方式

skill 文件保存在 `skills/` 目录下，通过 `CLAUDE.md` 中的引用注册：

```markdown
## Skills

本项目包含以下技能，存储在 `skills/` 目录：

- [model-pipeline](skills/model-pipeline/SKILL.md) — ...
- ...
```

每个 skill 在 CLAUDE.md 中写明触发条件，供 Claude 在对话开始时判断是否需要加载。
不需要在 `.claude/settings.json` 中注册，项目级 skill 通过 CLAUDE.md 引用发现。

## 实现顺序

1. 基础技能先行（模型流水线 → 设备运维 → 环境控制 → 结果处理）
2. 流程技能在后（交叉编译 → 基准测试 → Profiling → 框架集成）
3. 每个 skill 按 writing-skills 的 TDD 流程（RED → GREEN → REFACTOR）