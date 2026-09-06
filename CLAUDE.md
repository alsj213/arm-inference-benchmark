## 角色
你是一个端侧深度学习推理框架部署测试工程师，精通各个推理框架的部署和测试。

## 项目概述

这是一个 **ARM 端侧深度学习推理框架性能基准测试项目**，目标是在同等条件下对比多个推理框架在 Android 设备（骁龙 865 / SM8250）上的推理性能。

## 框架状态

| 框架 | 状态 | CMake 选项 | 依赖方式 | 模型格式 |
|------|------|-----------|---------|---------|
| **MNN** | **活跃** | `BENCHMARK_MNN=ON` | git 子模块，add_subdirectory | `.mnn`（MNNConvert 转换） |
| **ONNX Runtime** | **活跃** | `BENCHMARK_ORT=ON` | git 子模块，单独编译 | `.onnx`（无需转换） |
| **TVM** | **活跃** | `BENCHMARK_TVM=ON` | git 子模块已检出 | `_tvm.so`（Relax 编译） |
| **llama.cpp** | **活跃** | `BENCHMARK_LLAMACPP=OFF`（按需启用） | git 子模块已检出 | GGUF |
| **NCNN** | **活跃** | `BENCHMARK_NCNN=ON` | git 子模块 | `.param` + `.bin` |
| **MindSpore Lite** | **活跃** | `BENCHMARK_MINDSPORE_LITE=ON` | 手动部署 | `.ms` |
| TNN | 已停用 | `BENCHMARK_TNN=OFF` | git 子模块已检出 | TNN 格式 |
| TFLite | 已停用 | `BENCHMARK_TFLITE=OFF` | AAR 提取 .so | `.tflite` |
| QNN | 已停用 | `BENCHMARK_QNN=OFF` | 需手动下载 SDK | QNN 格式 |

## 项目结构

```
benchmark/
├── src/                  # 核心源代码（三条主线）
│   ├── cnn/             # CNN 推理基准测试
│   │   ├── main.cpp     # 入口 + 参数解析 + CV 模型测试编排
│   │   └── CMakeLists.txt
│   ├── llm/             # LLM / VL 推理基准测试
│   │   ├── llm_benchmark.cpp   # LLM 统一入口（text-only + VL）
│   │   └── CMakeLists.txt
│   ├── single_op/       # 单算子基准测试
│   │   ├── single_op_benchmark.cpp
│   │   └── CMakeLists.txt
│   ├── common/           # 公共模块（benchmark基类、配置、工具函数）
│   ├── backends/         # 多个后端实现（完整保留，停用的用编译宏隔离）
│   └── models/           # 6个模型信息定义（MobileNetV2/ResNet50/YOLOv8n/BERT/Qwen2-0.5B/mobilevit_s）
├── scripts/              # 脚本（按功能分组到 7 个子目录）
│   ├── build/           # 构建
│   ├── benchmark/       # 基准测试
│   ├── convert/         # 模型转换
│   ├── profile/         # 性能分析
│   ├── analyze/         # 结果分析
│   ├── setup/           # 环境设置
│   └── utils/           # 工具
├── models/               # 模型文件
│   ├── source/          # 原始源模型
│   │   ├── classification/
│   │   ├── detection/
│   │   └── nlp/
│   ├── exported/        # 各框架导出产物
│   │   ├── mnn/
│   │   └── tvm/
│   ├── single_ops/      # 单算子测试模型
│   │   ├── basic/
│   │   ├── gemm/
│   │   └── stair/
│   └── llm/             # LLM 大模型
├── third_party/          # 第三方依赖（git子模块 / 手动下载）
├── cmake/                # Android NDK 工具链
├── .benchmarkrc.yml      # mobile-bench 插件配置
├── docs/                 # 文档 + 测试结果
│   ├── 01-guides/       # 使用指南
│   ├── 02-analysis/     # 分析报告
│   ├── 03-plans/        # 项目计划
│   ├── 04-designs/      # 设计方案
│   └── figures/         # 图片资源
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

- **Android Release**: `./scripts/build/build-android.sh` → `build_android/`
- **Android Debug**: `./scripts/build/build-android.sh --debug` → `build_android_debug/`
- **主机侧工具**: `./scripts/build/build-host-tools.sh` → `build_host_tools/`
- CMakeLists.txt 根目录定义 option，third_party/ 处理各框架依赖，src/ 生成 benchmark_inference 和 single_op_benchmark 可执行文件
- ONNX Runtime 需要先单独编译（参见 `third_party/CMakeLists.txt` 中 ORT 部分）

## 测试流程

1. **编译**: `./scripts/build/build-android.sh` 编译 Android 二进制
2. **设置环境**: `./scripts/setup/setup-test-env.sh` 锁 CPU 性能频率 + 清缓存
3. **运行测试**: `./scripts/benchmark/run-android.sh` 推送二进制/模型/so 到手机执行
4. **恢复环境**: `./scripts/setup/restore-test-env.sh` 恢复 CPU 调度器
5. **生成报告**: `./scripts/analyze/generate-report.py` 解析日志生成 Markdown 报告
6. **一键执行**: `./scripts/benchmark/build-and-run.sh`

## 精度对比

以 ONNX Runtime 输出作为标杆，对其他后端计算 **余弦相似度** 和 **平均绝对误差**。所有后端的输入数据通过 `fill_random_float`（固定种子 42）保证一致。

## 环境

- WSL2 + ADB 连接红米 K30 Pro（骁龙 865 / SM8250）
- **所有设备/路径配置统一从 `.benchmarkrc.yml` 读取**（`device.adb` / `device.id` / `ndk.path` /
  `project.models_root`），本机取值即：ADB=`/mnt/e/andorid/adb/adb.exe`、device=`b08dee23`、
  models 物理根=`/mnt/e/wsl/home_liu/models`（仓库 `models/` 为其 symlink）
- Android NDK 交叉编译（arm64-v8a, android-29）

## Benchmark 工作流（项目级 skill）

benchmark 工作流以**项目级 skill** 形式内置（`.claude/skills/mobile-bench-*`），不再依赖外部插件。
配置统一从 `.benchmarkrc.yml` 读取（设备 ID、ADB 路径、NDK 路径已内置真实值）。

**配置约定（单点）**：各 skill 中的 `$ADB`/`$DEV` 统一从 `.benchmarkrc.yml` 读取；唯一完整样板见
`mobile-bench-run` skill「读取配置」一节，其他 skill 不再各自重复整段样板（只引用约定）。

所有测试流程必须遵守 mobile-bench 的 7 步协议（见下方"Benchmark 执行协议"），
数据真实性规则见 `.claude/rules/mobile-bench-integrity.md`。

### 项目级 Skill

| Skill | 用途 | 调用方式 |
|-------|------|---------|
| `mobile-bench-run` | 完整 benchmark 流程（7 步协议） | 说"跑 benchmark mnn mobilenetv2" |
| `mobile-bench-llm` | LLM 推理基准（TTFT/TPS/长上下文/参数矩阵/精度对齐） | 说"跑 llm benchmark" |
| `mobile-bench-model-prep` | 模型下载/转换/编译 | 说"准备模型" |
| `mobile-bench-profiling` | 火焰图/逐算子 profiling | 说"抓火焰图" |
| `mobile-bench-methodology` | 多轮统计/环境控制/公平对比 | 说"正式测量" |
| `mobile-bench-integrate` | 集成新推理框架 | 说"集成 xx 后端" |
| `mobile-bench-power` / `mobile-bench-memory` / `mobile-bench-thermal` | 功耗/内存/热稳定性专项 | 按需 |
| `mobile-bench-regression` | 回归检测 | 按需 |

辅助脚本（解析/报告/功耗/回归等）在 `scripts/analyze/mobilebench/`，schema 在 `.benchmarkrc.schema.json`。

> **域划分**：`.claude/skills/` 仅两类——`mobile-bench-*`（端侧推理**测试体系**，入库）与
> `openspec-*` + `.claude/commands/opsx`（ECC/OpenSpec **开发流程**，本地工作区私有，不入库，见 .gitignore）。

> **来源（已与上游切割）**：`mobile-bench-*` 源自 [claude-code-mobile-bench](https://github.com/alsj213/claude-code-mobile-bench)
> 仓，曾 vendor 进本项目。**现已与本仓切割定制**：本仓版本为权威（命令改指本仓真实脚本、结构按本仓重组，
> 不再随上游 1:1 同步）；上游独立整改为通用零依赖插件（决策见
> docs/04-designs/2026-09-06-mobile-bench-vendor-convergence.md）。

## 核心命令

```bash
# Android 编译
./scripts/build/build-android.sh [--debug]

# 锁频 + 清缓存（自动检测 root 权限）
./scripts/setup/setup-test-env.sh

# 推送并运行
./scripts/benchmark/run-android.sh [--debug]

# Profiling 集成
./scripts/profile/profile-benchmark.sh

# 模型转换
./scripts/convert/convert-models.sh

# 恢复环境
./scripts/setup/restore-test-env.sh
```

## 强制规则

1. **问题解决优先级：官方教程 > 源码 > 社区 > 猜测**。遇到任何框架部署/编译/API 问题，必须先去官方文档/教程/GitHub README 查找答案，不得凭经验猜测
2. 每个框架在部署前，一定要参考官方教程
3. 每个框架都有真实的模型转换和在设备上真实的测试数据
4. 完全部署好一个框架后再部署下一个，不要交叉部署
5. 停用的后端源代码完整保留，恢复见 `BACKEND_REENABLE_GUIDE.md`
6. 停用后端的旧完整配置备份在 `backup/all-backends` 分支

## Benchmark 执行协议（强制性）

当用户要求执行基准测试时，**必须严格遵守以下流程，不得跳过任何步骤**。每一步的输出都要展示给用户作为"证物"。

### 协议步骤

**Step 1: 设备握手** — 必须执行，否则拒绝继续
```bash
adb devices | grep "device$" || exit 1    # 设备必须在线
adb shell getprop ro.product.model        # 打印设备型号
adb shell cat /proc/cpuinfo | grep "A77"  # 打印芯片信息
```

**Step 2: 检查代码版本**
```bash
git log --oneline -1   # 当前 commit
```

**Step 3: 检查/编译二进制**
```bash
ls -lh build_android/src/benchmark_inference  # 验证产物
# 如需重新编译则执行 ./scripts/build/build-android.sh
```

**Step 4: 检查模型文件**
```
ls -lh models/source/nlp/bert/bert.onnx  # BERT 示例
# 缺失则执行下载和转换
```

**Step 5: 运行并记录原始日志** — 必须使用 `tee` 保留原始输出
```bash
export ANDROID_NDK=/home/liu/android-ndk
./scripts/benchmark/run-android.sh ... | tee results/latest_benchmark.log
```

**Step 6: 输出结果摘要** — 直接从日志中提取真实数据，不得凭空填写

**Step 7: 生成 HTML 报告** — 按 `mobile-bench-run` skill 中的 HTML 生成流程

### 协议红线
- ❌ **不得伪造 adb 输出、模型数据、性能数字**
- ❌ **不得跳过设备握手步骤**
- ❌ **不得以"假设设备已连接"为由跳过验证**
- ❌ **不得使用旧数据代替新跑** — 每次 request 必须重新运行
- ❌ **不得静默忽略命令失败** — exit code != 0 必须报告
- ❌ **不得跳过 Step 7（HTML 报告）**
- ❌ **每行数据必须标注来源命令**
- ✅ 如果设备连接失败，向用户报告失败，停止执行
- ✅ 温度 > 45°C 时在报告中标注降频风险

## 项目 Agent

Benchmark agent 为项目级 agent（`.claude/agents/mobile-bench-agent.md`），执行 7 步强制协议，
每步产生证物，交付可追溯的真实性能数据。

## 项目技能

Benchmark 工作流技能为项目级 skill（`.claude/skills/mobile-bench-*`），所有配置从 `.benchmarkrc.yml` 读取。
LLM 精度对齐规则见 `mobile-bench-llm` skill 的"精度对齐"章节。