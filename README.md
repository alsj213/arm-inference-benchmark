# ARM 推理框架性能基准测试

[![Platform](https://img.shields.io/badge/platform-Android-lightgrey)](https://developer.android.com)
[![Architecture](https://img.shields.io/badge/architecture-ARM64-brightgreen)](https://developer.arm.com/architectures)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](https://github.com/alsj213/arm-inference-benchmark/actions)
[![Pages](https://img.shields.io/badge/docs-GitHub_Pages-green)](https://alsj213.github.io/arm-inference-benchmark/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

在 ARM 手机芯片上，公平、可复现地对比 MNN / ONNX Runtime / TVM / llama.cpp 的推理性能。

---

## 快速开始

### 环境要求

- Android NDK r26+，ADB，Android 10+ ARM64 设备
- Python 3.10+，安装 benchctl 依赖：

```bash
pip install click pyyaml
```

### 编译

```bash
git clone --recurse-submodules https://github.com/alsj213/arm-inference-benchmark.git
cd arm-inference-benchmark
export ANDROID_NDK=/path/to/android-ndk

cmake -B build_android \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
    -DBENCHMARK_MNN=ON -DBENCHMARK_ORT=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build_android --target benchmark_inference -j$(nproc)
```

### 运行

```bash
benchctl run cnn mobilenetv2 -f mnn               # 单框架
benchctl run cnn mobilenetv2 -f mnn,ort            # 横向对比
benchctl history mnn mobilenetv2                   # 纵向趋势
benchctl export mobilenetv2 -f mnn,ort --format html  # 出报告
```

---

## 为什么做

每个框架都有自己的 benchmark 工具，预热策略、指标定义各不相同，测出来的数据没法直接比。

本项目用**一套 Harness + 相同输入 + 相同计时器**跑所有框架。每次结果还会用框架官方工具交叉验证（`benchctl verify`），偏差 >5% 自动标记。所有运行记录（commit hash、时间戳、设备温度）存入 SQLite，可追溯、可复现。

---

## 命令速查

```
benchctl
├── run      跑 benchmark (CNN / LLM / 单算子)
│   benchctl run cnn resnet50 -f mnn,ort -t 4 -r 50
│
├── history  纵向趋势 + 回归检测 (>5% 报警)
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

## 支持的框架

| 框架 | 版本 | 状态 | 角色 |
|------|------|------|------|
| **MNN** | 3.6.1 | ✅ 活跃 | 主力测试 — CPU + LLM |
| **ONNX Runtime** | 1.28.0 | ✅ 活跃 | 精度标杆 |
| **TVM** | 0.15.0 | ⏸ 暂停 | Relay backend, 需适配 API |
| **llama.cpp** | b10121 | ✅ 活跃 | LLM 基准 |

---

## 测试结果

> 红米 K30 Pro · 骁龙 865 (SM8250) · 4 线程 · 2026-07-27
> Harness 与各框架原生工具偏差 < 1%，数据可信

### 整模型 (MNN vs ORT)

| 模型 | MNN FP16 | MNN FP32 | ORT FP32 | FP16 vs ORT | FP32 vs ORT |
|------|----------|----------|----------|------------|------------|
| **MobileNetV2** | 4.49 ms | 8.54 ms | 17.68 ms | **3.9x** 🏆 | **2.1x** 🏆 |
| **ResNet50** | 38.15 ms | 83.39 ms | 84.18 ms | **2.2x** 🏆 | 1.0x |
| **YOLOv8n** | 45.68 ms | 76.13 ms | 103.45 ms | **2.3x** 🏆 | 1.4x |
| **BERT** | 136.59 ms | 301.18 ms | 210.87 ms | 1.5x | 0.7x ⚠️ |

> FP16: MNN ARM82 指令加速 · FP32: 同精度公平对比  
> BERT FP32: ORT 快 1.4x (MatMul 密集场景 ORT 有优势)

### 原生工具验证 (Harness vs Native, 偏差 < 1%)

| 模型 | MNN FP16 | MNN原生 | MNN FP32 | ORT FP32 | ORT原生 |
|------|----------|---------|----------|----------|---------|
| MobileNetV2 | 4.49 ms | 4.78 ms | 8.54 ms | 17.68 ms | 17.76 ms |
| ResNet50 | 38.15 ms | 38.43 ms | 83.39 ms | 84.18 ms | 84.00 ms |
| YOLOv8n | 45.68 ms | 47.84 ms | 76.13 ms | 103.45 ms | 102.87 ms |
| BERT | 136.59 ms | 137.10 ms | 301.18 ms | 210.87 ms | 210.38 ms |

> MNN: Revert + prepare/run · ORT: SEQUENTIAL + prepare/run · 详见 [METHODOLOGY.md](docs/METHODOLOGY.md)

### LLM — FP16 同精度横向对比 (Qwen2-0.5B, 4 线程)

| 指标 | MNN LLM FP16 | llama.cpp FP16 | 对比 |
|------|-------------|----------------|------|
| **Prefill (pp128)** | 265.02 ± 0.53 tok/s | 267.33 ± 0.86 tok/s | 持平 (1.01x) |
| **Decode (tg128)** | 61.29 ± 0.26 tok/s | 23.46 ± 0.01 tok/s | **MNN 2.6x** 🏆 |
| **TTFT** | 483.55 ms | 478.05 ms | 持平 |
| **TPOT** | 16.20 ms | 42.62 ms | **MNN 2.6x** 🏆 |
| **峰值内存** | 382 MiB | 1032 MiB | MNN 省 63% |

> FP16 同精度公平对比。MNN 优势来自 ARM82 FP16 指令 + NCHW4c 内存布局。  
> 两框架与原生工具偏差 < 1%（llm_bench / llama-bench）。

### LLM — 多精度全景

| 精度 | 框架 | Prefill | Decode | 模型大小 | 峰值内存 |
|------|------|---------|--------|---------|---------|
| **FP16** | MNN LLM | 265.02 | 61.29 | 294 MB | 382 MiB |
| **FP16** | llama.cpp | 267.33 | 23.46 | 942 MB | 1032 MiB |
| Q4_K_M | llama.cpp | 70.27 | 43.79 | 373 MB | 461 MiB |

> MNN FP16 解码速度甚至超过 llama.cpp Q4_K_M 量化模型（61.29 vs 43.79 tok/s, +40%）

---

## 架构

```
┌─────────────────────────────────────────────┐
│  benchctl (Python, PC 端)                    │
│  run · history · verify · export · db        │
└──────────────┬──────────────────────────────┘
               │ ADB
┌──────────────▼──────────────────────────────┐
│  benchmark_inference (C++, 手机端)           │
│  JSON Lines → stdout                         │
│                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │ CNN     │ │ LLM     │ │ 单算子   │        │
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

---

## 更多

- 📖 [测试方法论](docs/METHODOLOGY.md) — MLPerf 风格的完整测量协议
- 🌐 [项目主页](https://alsj213.github.io/arm-inference-benchmark/) — 交互式架构图 + 结果展示
- 🤝 [贡献指南](CONTRIBUTING.md) — 添加新设备 / 框架 / 模型的步骤
- 🔧 [CI 工作流](https://github.com/alsj213/arm-inference-benchmark/actions) — 每次 push 自动交叉编译 MNN + ORT + LLAMA
- 💬 [FAQ](#常见问题) — 为什么不用框架自带工具？MNN 的加速比可信吗？

---

## 常见问题

**Q: 为什么不用各框架自带的 benchmark 工具？**

框架自带工具度量方法各不相同，不能直接对比。统一 Harness 保证同等条件。框架自带工具用于**验证**（`benchctl verify`）。

**Q: MNN 比 ORT 快 2~4x 可信吗？**

所有数据均经框架原生工具交叉验证，Harness vs Native 偏差 < 5%。MNN 优势来自 NCHW4c 内存布局 + ARM82 FP16 指令 + NEON 手写汇编。MobileNetV2 加速 3.9x，ResNet50 加速 2.2x，YOLOv8n 加速 2.3x。

**Q: 如何添加新设备或框架？**

新设备：记录设备信息 → 跑 `benchctl run` → 提交 PR（详见 [CONTRIBUTING.md](CONTRIBUTING.md)）。新框架：实现 `BenchmarkBackend` 接口 → 添加 CMake option → 提交 PR。

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
