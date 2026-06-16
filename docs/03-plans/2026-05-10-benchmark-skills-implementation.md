# Benchmark Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task.

**Goal:** 创建 8 个项目级 skill（4 基础 + 4 流程），覆盖 MNN/ORT 完整工作流，保存到本项目 `skills/` 目录

**Architecture:** 基础技能被流程技能引用。按基础→流程的顺序实现。每个 skill 独立 SKILL.md 文件。

**保存位置:** `skills/<skill-name>/SKILL.md`

---

### Task 1: 创建 skills 目录结构

**Files:**
- Create: `skills/.gitkeep`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p /home/liu/project/newwork/benchmark/skills/{model-pipeline,android-device-ops,test-environment-control,result-processor,android-cross-compile,run-benchmark,performance-profiling,integrate-framework}
```

---

### Task 2: model-pipeline SKILL.md

**Files:**
- Create: `skills/model-pipeline/SKILL.md`

- [ ] **Step 1: 创建 SKILL.md**

```markdown
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
| MobileViT-S | classification | .mnn | .onnx | 就绪 |
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
  --MNNModel models/classification/mobilenetv2/mobilenetv2.mnn \
  --bizCode benchmark
```

### 3. ONNX Runtime

ORT 直接使用 `.onnx` 格式，无需转换。YOLOv8n 可使用优化后的 `.ort` 格式。

## 常见问题

- **MNNConvert 找不到**: 先运行 `build_host_tools.sh`，产物在 `tools/bin/MNNConvert`
- **模型路径不匹配**: 代码中通过 `ModelInfo` 结构体管理路径，保持目录结构一致
- **模型文件损坏**: 重新运行 `download_pretrained.py`
```

---

### Task 3: android-device-ops SKILL.md

**Files:**
- Create: `skills/android-device-ops/SKILL.md`

- [ ] **Step 1: 创建 SKILL.md**

```markdown
---
name: android-device-ops
description: Use when needing to interact with Android test device — check ADB connection, push binaries/libraries/models, execute remote commands, pull test results
---

# Android Device Ops

## Overview

管理 Android 测试设备的 ADB 通信操作。测试设备为红米 K30 Pro（骁龙 865），通过 WSL2 连接。

## 环境

- ADB: `/mnt/e/andorid/adb/adb.exe`
- 设备 ID: `b08dee23`
- 别名: `alias adb='/mnt/e/andorid/adb/adb.exe'`

## 快速参考

| 操作 | 命令 |
|------|------|
| 检查连接 | `adb devices` |
| 推送二进制 | `adb push build_android/src/benchmark_inference /data/local/tmp/` |
| 推送 ORT so | `adb push third_party/onnxruntime/build/Android/Release/libonnxruntime.so /data/local/tmp/` |
| 推送模型 | `adb push models /data/local/tmp/` |
| 远程执行 | `adb shell "cd /data/local/tmp && ./benchmark_inference --help"` |
| 拉取结果 | `adb pull /data/local/tmp/results.txt ./results/` |
| 设置权限 | `adb shell chmod +x /data/local/tmp/benchmark_inference` |

## 常用工作流

推送并运行：

```bash
adb shell mkdir -p /data/local/tmp/models
adb push build_android/src/benchmark_inference /data/local/tmp/
adb shell chmod +x /data/local/tmp/benchmark_inference
adb push third_party/onnxruntime/build/Android/Release/libonnxruntime.so /data/local/tmp/
adb push models/classification/mobilenetv2 /data/local/tmp/models/
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2"
```

## 常见问题

- **device not found**: 检查 `adb devices`，确认设备 ID `b08dee23` 在列表中
- **permission denied**: 推送后 `chmod +x`
- **ORT so 找不到**: 设置 `LD_LIBRARY_PATH=.`
- **空间不足**: `adb shell rm -rf /data/local/tmp/*`
```

---

### Task 4: test-environment-control SKILL.md

**Files:**
- Create: `skills/test-environment-control/SKILL.md`

- [ ] **Step 1: 创建 SKILL.md**

```markdown
---
name: test-environment-control
description: Use before running benchmarks or profiling to ensure consistent test conditions — lock CPU frequency, clear caches, restore environment after testing
---

# Test Environment Control

## Overview

在 Android 设备上保证测试环境一致性：锁 CPU 频率、关小核、清缓存，测试完成后恢复调度器。

## ⚠️ 注意

当前设备**未 root**，部分高级功能不可用。脚本会自动检测 root 状态并降级。

## 快速参考

| 操作 | 命令 |
|------|------|
| 锁频（root） | `./scripts/setup_test_environment.sh --root` |
| 锁频（非 root） | `./scripts/setup_test_environment.sh --no-root` |
| 恢复环境 | `./scripts/restore_test_environment.sh` |

## 工作流

```bash
# 脚本自动推送到设备执行
./scripts/setup_test_environment.sh --no-root

# 测试完成后恢复
./scripts/restore_test_environment.sh
```

脚本操作：
- CPU 设为 performance governor
- 清理缓存（`echo 3 > /proc/sys/vm/drop_caches`，需 root）
- 记录当前状态到日志
- 恢复时还原为 schedutil/interactive governor

## 常见问题

- **非 root 锁频效果有限**: 无 root 无法真正锁定频率，但 performance governor 仍有改善
- **测试前必须清缓存**: 防止上次结果影响本次推理
- **测试后必须恢复**: 避免 performance 模式导致发热/耗电
```

---

### Task 5: result-processor SKILL.md

**Files:**
- Create: `skills/result-processor/SKILL.md`

- [ ] **Step 1: 创建 SKILL.md**

```markdown
---
name: result-processor
description: Use after benchmarks or profiling complete — parse logs, compare precision with cosine similarity, generate Markdown reports, validate results
---

# Result Processor

## Overview

处理 benchmark 结果：解析日志、精度对比（以 ONNX Runtime 为标杆）、生成报告。

## 精度对比机制

- **标杆**: ONNX Runtime 输出
- **指标**: 余弦相似度（方向一致性，1.0=完全一致）、平均绝对误差（MAE）
- **输入一致**: `fill_random_float` 使用固定种子 42，保证所有后端输入相同

## 快速参考

| 操作 | 命令 |
|------|------|
| 生成报告 | `python scripts/generate_report.py -i <log> -o <output>` |
| 验证结果 | `python scripts/validate_results.py -i <results>` |
| 分析单算子 | `python scripts/analyze_single_op_results.py -i <csv>` |

## 输出指标

| 指标 | 说明 |
|------|------|
| P50 | 中位数延迟（ms），核心指标 |
| P90 | 90% 尾部延迟 |
| P99 | 99% 尾部延迟 |
| FPS | 每秒推理帧数 |
| 内存峰值 | 推理过程最大内存占用 |
| 余弦相似度 | 与 ORT 标杆的精度一致性 |

## 工作流

```bash
python scripts/generate_report.py \
  -i results/sm8250/benchmark_20260501.log \
  -o results/final_report.md

python scripts/validate_results.py -i results/final_report.md
```

## 常见问题

- **精度对比为 N/A**: 后端未实现 `infer_with_output()` 或 ORT 标杆输出不可用
- **数据不一致**: 检查是否使用固定种子 42
- **报告为空**: 检查日志文件路径是否正确
```

---

### Task 6: android-cross-compile SKILL.md

**Files:**
- Create: `skills/android-cross-compile/SKILL.md`

- [ ] **Step 1: 创建 SKILL.md**

```markdown
---
name: android-cross-compile
description: Use when compiling benchmark binary for Android ARM64 — configure NDK, build MNN statically, prepare ORT dynamic library, select Release/Debug mode
---

# Android Cross Compile

## Overview

将 benchmark 交叉编译到 Android ARM64。MNN 静态编译，ONNX Runtime 为预编译动态库。

## 构建类型

| 类型 | 脚本 | 输出目录 |
|------|------|---------|
| Release | `./scripts/build_android.sh` | `build_android/` |
| Debug | `./scripts/build_android.sh --debug` | `build_android_debug/` |

## 工作流

### 1. 安装依赖

```bash
./scripts/setup_deps.sh
```

### 2. 编译 ONNX Runtime（首次）

```bash
cd third_party/onnxruntime
./build.sh --config Release --android --arm64 --build_shared_lib
```

### 3. 编译 Benchmark

```bash
# Release
./scripts/build_android.sh

# Debug（profiling 需要）
./scripts/build_android.sh --debug
```

### 4. 主机侧工具

```bash
./scripts/build_host_tools.sh  # 产物 → tools/bin/MNNConvert
```

## CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `BENCHMARK_MNN` | ON | MNN 后端 |
| `BENCHMARK_ORT` | ON | ONNX Runtime 后端 |
| 其他 | OFF | 已停用 |

## 常见问题

- **ORT 头文件找不到**: 确保 ORT 子模块已初始化并预编译
- **MNN 编译慢**: 首次编译静态库较慢，后续增量编译很快
```

---

### Task 7: run-benchmark SKILL.md

**Files:**
- Create: `skills/run-benchmark/SKILL.md`

- [ ] **Step 1: 创建 SKILL.md**

```markdown
---
name: run-benchmark
description: Use when running inference performance tests on Android devices — full workflow from build to report
---

# Run Benchmark

## Overview

端侧推理性能基准测试完整流程。

**引用基础技能**: model-pipeline, android-device-ops, test-environment-control, result-processor

## 完整流程

```
编译 → 模型就绪 → 推送设备 → 锁频 → 执行 → 恢复环境 → 报告
```

## 一键运行

```bash
./scripts/build_and_run.sh
```

## 分步流程

### Step 1: 编译

```bash
./scripts/build_android.sh
```

### Step 2: 推送并运行

```bash
./scripts/run_benchmark_android.sh
```

脚本自动完成：推送二进制 → 推送 libonnxruntime.so → 推送模型 → 锁频 → 执行 → 拉取结果

### Step 3: 生成报告

```bash
python scripts/generate_report.py -i results/sm8250/*.log -o results/final_report.md
```

## 命令行参数

```
./benchmark_inference --model <name> --backend <type> --num_threads <n> --num_runs <n> --warmup_runs <n>
```

- `--model`: mobilenetv2, resnet50, shufflenet_v2, mobilevit_s, yolov8n
- `--backend`: mnn, ort

## 输出示例

```
[模型] MobileNetV2 | [后端] MNN
  P50:   8.70 ms
  P90:   9.12 ms
  P99:   9.45 ms
  FPS:   114.94
  余弦相似度: 1.000000 (vs ORT)
```

## 常见问题

- **结果波动大**: 检查是否已锁频
- **模型找不到**: `adb shell ls /data/local/tmp/models/`
- **so 找不到**: 设置 `LD_LIBRARY_PATH=.`
```

---

### Task 8: performance-profiling SKILL.md

**Files:**
- Create: `skills/performance-profiling/SKILL.md`

- [ ] **Step 1: 创建 SKILL.md**

```markdown
---
name: performance-profiling
description: Use when analyzing inference latency bottlenecks — operator-level profiling, CPU sampling, system tracing
---

# Performance Profiling

## Overview

分析推理延迟瓶颈，从算子级别到系统级别。

**引用基础技能**: android-device-ops, test-environment-control, result-processor

## 方法对比

| 方法 | 粒度 | 适用场景 |
|------|------|---------|
| MNN_PROFILING | 逐算子 | MNN 算子耗时分布 |
| ORT Profiling API | 逐算子 | ORT 算子耗时分布 |
| simpleperf | CPU 采样 | 热点函数定位、火焰图 |
| atrace | 系统 trace | 系统调用、线程调度 |
| perfetto | 综合 trace | CPU/GPU/内存全面分析 |

## 集成入口

```bash
./scripts/profile_benchmark.sh [--simpleperf] [--atrace] [--perfetto] [--mnn-profile] [--ort-profile]
```

## 工作流

### 前置条件

```bash
# Debug 构建（符号不剥离）
./scripts/build_android.sh --debug
```

### MNN 逐算子

```bash
adb shell "cd /data/local/tmp && MNN_PROFILING=1 ./benchmark_inference --model mobilenetv2 --backend mnn"
```

### simpleperf 火焰图

```bash
./scripts/simpleperf_profile.sh --app benchmark_inference --duration 30
# 产物: results/profiling/flamegraph.svg
```

### atrace / perfetto

```bash
./scripts/atrace_capture.sh --duration 10 --categories sched,freq
./scripts/perfetto_trace.sh --duration 15
```

## 常见问题

- **火焰图无符号**: 使用 Debug 构建
- **MNN_PROFILING 无输出**: 确认 MNN 编译时未禁用 profiling
- **perfetto 无法启动**: Android 10+ 内置支持，检查 `adb shell perfetto --version`
```

---

### Task 9: integrate-framework SKILL.md

**Files:**
- Create: `skills/integrate-framework/SKILL.md`

- [ ] **Step 1: 创建 SKILL.md**

```markdown
---
name: integrate-framework
description: Use when adding a new inference framework backend or re-enabling a disabled one — implement backend class, integrate CMake, convert models, validate precision
---

# Integrate Framework

## Overview

集成新推理框架或恢复已停用后端。恢复详细步骤见 `BACKEND_REENABLE_GUIDE.md`。

## 后端架构

所有后端继承 `BenchmarkBackend`（`src/common/benchmark.h`）：

```
BenchmarkBackend（抽象基类）
├── init(config)              ← 加载模型
├── infer(input)              ← 推理
├── infer_with_output(input, output)  ← 推理+输出（精度对比）
├── deinit()                  ← 释放资源
└── name()                    ← 后端名称
```

工厂函数 `create_backend(BackendType)` 由编译宏控制。

## 集成步骤

### 1. 创建后端文件

在 `src/backends/` 下创建 `xxx_backend.h/cpp`，实现 `BenchmarkBackend` 所有虚方法。

### 2. CMake 集成

根 `CMakeLists.txt` 添加 option：

```cmake
option(BENCHMARK_NEW_FW "Enable NewFW backend" OFF)
```

### 3. 模型转换

```bash
./scripts/convert_models.sh  # ONNX → 目标框架格式
```

### 4. 精度验证

以 ORT 为标杆，余弦相似度 > 0.99 为通过。

## 恢复已停用后端

```bash
# 1. 确保依赖就绪（third_party/）
# 2. CMake 中将 option 改为 ON
# 3. 重新编译
# 4. 转换模型 → 运行测试
```

## 关键约束

- 每个框架部署前参考官方教程
- 必须有真实模型转换和在设备上的测试数据
- 一次只部署一个框架，解决完再部署下一个
```

---

### Task 10: 更新 CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`（追加 Skills 章节）

- [ ] **Step 1: 在 CLAUDE.md 末尾追加 Skills 引用章节**

追加内容：

```markdown
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
```

---

## 依赖关系

```
基础技能（Task 2-5）可并行执行
     │
     ▼
流程技能（Task 6-9）依赖基础技能
     │
     ▼
CLAUDE.md 更新（Task 10）依赖全部完成
```

**推荐执行顺序:** Task 1 → (Task 2,3,4,5 并行) → (Task 6,7,8,9 并行) → Task 10