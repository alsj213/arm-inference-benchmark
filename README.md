# ARM 推理框架性能基准测试

[![Platform](https://img.shields.io/badge/platform-Android-lightgrey)](https://developer.android.com)
[![Architecture](https://img.shields.io/badge/architecture-ARM64-brightgreen)](https://developer.arm.com/architectures)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](https://github.com/alsj213/arm-inference-benchmark/actions)
[![Pages](https://img.shields.io/badge/docs-GitHub_Pages-green)](https://alsj213.github.io/arm-inference-benchmark/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

在 ARM 手机芯片上公平、可复现、可验证地对比多个端侧推理框架的性能。

---

## 为什么做这个项目

对比推理框架的性能很难：每个框架有自己的 benchmark 工具，预热策略不同，指标定义不统一，测出来的数据各说各话。

**本项目解决三个问题：**

- **一套 Harness，多个框架** — 相同输入、相同计时器、相同设备、相同协议
- **内置验证机制** — 每次测试结果都会用框架官方工具交叉验证，偏差 >5% 自动标记
- **完整审计追踪** — 每次运行都存入 SQLite，记录 commit hash、时间戳、设备温度

```bash
benchctl run cnn resnet50 -f mnn,ort         # 横向对比
benchctl history mnn resnet50                 # 纵向趋势
benchctl verify mnn resnet50                  # 原生工具验证
benchctl export mobilenetv2 -f mnn,ort --html # 生成报告
```

---

## 测试数据

### 整模型性能 (CNN / NLP)

> 红米 K30 Pro · 骁龙 865 (SM8250) · FP32 · 4 线程

| 模型 | MNN | ORT | TVM | MNN vs ORT |
|-------|-----|-----|-----|------------|
| **MobileNetV2** | 8.55 ms | 21.68 ms | — | **2.5x** 🏆 |
| **ResNet50** | 83.47 ms | 142.62 ms | 64.38 ms | 1.7x |
| **YOLOv8n** | 76.28 ms | 134.38 ms | 126.63 ms | 1.8x |
| **BERT** | 322.13 ms | 263.50 ms | — | 0.82x |

### LLM 性能

> Qwen2-0.5B · 4 线程

| 框架 | Prefill (tok/s) | Decode (tok/s) |
|-----------|-----------------|-----------------|
| **MNN LLM** | 267.09 | 60.84 |
| **llama.cpp** | 70.07 | 42.54 |

### 单算子测试

80+ 测试用例，覆盖 7 个类别：Conv1x1、DWConv、MatMul、LayerNorm、Softmax、GELU、非对齐 Conv。

*完整数据：[`results/single_op_benchmark_2026-06-09.md`](results/single_op_benchmark_2026-06-09.md)*

---

## 快速开始

### 环境要求

- **Android NDK** (r26+)
- **ADB** + Android 10+ 设备 (ARM64)
- **Python 3.10+**
- `pip install click pyyaml`

### 1. 克隆仓库

```bash
git clone --recurse-submodules https://github.com/alsj213/arm-inference-benchmark.git
cd arm-inference-benchmark
```

### 2. 编译

```bash
export ANDROID_NDK=/path/to/android-ndk
cmake -B build_android \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
    -DBENCHMARK_MNN=ON -DBENCHMARK_ORT=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build_android --target benchmark_inference -j$(nproc)
```

### 3. 运行

```bash
benchctl run cnn mobilenetv2 -f mnn          # 单框架
benchctl run cnn mobilenetv2 -f mnn,ort       # 横向对比
benchctl history mnn mobilenetv2              # 查看趋势
benchctl export mobilenetv2 -f mnn,ort --format html  # 导出报告
```

---

## 架构

```
┌─────────────────────────────────────────────┐
│  benchctl (Python CLI, PC 端)                │
│  run · history · verify · export · db        │
└──────────────┬──────────────────────────────┘
               │ ADB
┌──────────────▼──────────────────────────────┐
│  benchmark_inference (C++, 手机端)           │
│  JSON Lines → stdout                         │
│                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │ CNN     │ │ LLM     │ │ 单算子   │        │
│  │ P50/P99 │ │ TTFT/TPS│ │ 逐算子   │        │
│  └────┬────┘ └────┬────┘ └────┬────┘        │
│       └───────────┼───────────┘              │
│              ┌────▼────┐                     │
│              │ Harness  │   统一计时          │
│              └────┬────┘                     │
│    ┌──────────────┼──────────────┐           │
│    ▼              ▼              ▼           │
│  MNN    ONNX Runtime    TVM    llama.cpp     │
└──────────────────────────────────────────────┘
```

**核心设计：**

- **计时器与框架代码分离** — 计时在 Harness 层，不在框架 wrapper 里
- **JSON Lines 协议** — C++ 二进制每行输出一个 JSON 对象；benchctl 消费
- **SQLite 审计追踪** — 每次运行记录 commit hash、时间戳、设备温度
- **原生工具验证** — `benchctl verify` 调用框架自带工具交叉验证，标记偏差

---

## 支持的框架

| 框架 | 版本 | 状态 | 角色 |
|-----------|------|------|------|
| **MNN** | 3.6.1 | ✅ 活跃 | 主力测试 — CPU + LLM |
| **ONNX Runtime** | 1.28.0 | ✅ 活跃 | 精度标杆 |
| **TVM** | 0.15.0 | ⏸ 暂停 | Relay backend，需适配 v0.25 API |
| **llama.cpp** | b10121 | ✅ 活跃 | LLM 基准 |
| ~~NCNN~~ | — | 已移除 | — |
| ~~TFLite~~ | — | 已移除 | — |
| ~~TNN~~ | — | 已移除 | — |
| ~~QNN~~ | — | 已移除 | — |
| ~~MindSpore Lite~~ | — | 已移除 | — |

---

## 命令速查

```
benchctl
├── run      跑 benchmark (CNN/LLM/单算子)
│   benchctl run cnn resnet50 -f mnn,ort -t 4 -r 50
│
├── history  纵向趋势 + 回归检测
│   benchctl history mnn resnet50 --last 10
│
├── verify   原生工具交叉验证
│   benchctl verify mnn resnet50
│
├── export   导出报告 (JSON / HTML)
│   benchctl export resnet50 -f mnn,ort --format html
│
└── db       数据库管理
    benchctl db init | stats
```

---

## 测试方法

参考 [MLPerf Inference: Mobile](https://arxiv.org/abs/2012.02328)：

1. **固定随机种子** (42) — 所有框架输入一致
2. **预热** (10 轮) — 排除冷启动影响
3. **正式测试** (50+ 轮) — 单流串行推理
4. **指标** — P50 / P90 / P99 / FPS / 峰值内存
5. **精度** — 以 ONNX Runtime FP32 为基准计算余弦相似度

### 验证

每次测试结果都会用框架自己的 benchmark 工具交叉验证：

```
框架  │ Harness P50 │ 原生工具       │ 偏差    │ 结论
──────┼─────────────┼────────────────┼─────────┼────────
MNN   │ 8.58 ms     │ 5.39 ms (avg)  │ +59.0%  │ ⚠️ 待查
```

> MNN 原生 `benchmark.out` 测的是纯净 forward 时间，Harness 测了完整推理周期（memcpy → run → memcpy），偏差在预期范围内。

### 赛道定义

| 赛道 | 指标 | 模型 |
|-------|---------|--------|
| **CNN** | P50、P90、P99、FPS、内存 | MobileNetV2、ResNet50、YOLOv8n、BERT |
| **LLM** | TTFT、TPS、内存 | Qwen2-0.5B、Qwen3-4B |
| **单算子** | 逐算子延迟 | Conv1x1、MatMul、DWConv、LayerNorm…… |

---

## 项目结构

```
benchmark/
├── benchctl/          # Python CLI (PC 端)
│   ├── cli.py         # Click 命令
│   ├── db.py          # SQLite 数据库
│   ├── runner.py      # ADB 编排
│   ├── verify.py      # 原生工具验证
│   ├── report.py      # HTML/JSON 报告
│   └── tracks/        # 赛道逻辑 (cnn/llm/single_op)
├── src/               # C++ 代码 (手机端)
│   ├── cnn/           # CNN benchmark
│   ├── llm/           # LLM benchmark
│   ├── single_op/     # 单算子 benchmark
│   ├── common/        # Harness、配置、工具函数
│   ├── backends/      # 框架 wrapper (4 个后端)
│   └── models/        # 模型信息定义
├── third_party/       # Git 子模块 (MNN、ORT、TVM、llama.cpp)
├── scripts/           # Shell 脚本 (编译、转换、profiling)
├── docs/              # 文档 + 分析报告
├── results/           # 测试输出
└── .github/workflows/ # CI (编译 + lint + 回归)
```

---

## 开发

### CI/CD

每次 push 自动交叉编译 ARM64，验证 MNN / MNN+ORT / MNN+LLAMA 三路构建。

### 分支策略

- `main` — 稳定分支，CI 全绿
- `feat/*` — 功能分支，PR 合入
- `backup/*` — 历史备份

### 子模块版本

| 子模块 | Tag |
|-----------|-----|
| MNN | `3.6.1` |
| ONNX Runtime | `v1.28.0` |
| TVM | custom (v0.15.dev0) |
| llama.cpp | `b10121` |

---

## 常见问题

**Q: 为什么不用各框架自带的 benchmark 工具？**

A: 框架自带工具适合做*验证*（`benchctl verify`），但它们测量方法各不相同，不能直接对比。统一 Harness 保证同等条件、同等计时。

**Q: MNN 比 ORT 快 2.5x 真实吗？**

A: 在骁龙 865 + MobileNetV2 FP32 条件下确实如此。MNN 的 NCHW4c 内存布局 + NEON 手写汇编在小 CV 模型上有显著优势。但在 MatMul 为主的模型（如 BERT）上 ORT 反超。

**Q: 如何添加新框架？**

A: 在 `src/backends/` 实现 `BenchmarkBackend` 接口，添加 CMake option，写模型转换脚本。详见 `BACKEND_REENABLE_GUIDE.md`。

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
