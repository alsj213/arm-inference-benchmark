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

> 红米 K30 Pro · 骁龙 865 (SM8250) · FP32 · 4 线程 · 2026-07

| 模型 | MNN | ORT | MNN vs ORT |
|------|-----|-----|------------|
| **MobileNetV2** | 8.55 ms | 21.68 ms | **2.5x** 🏆 |
| **ResNet50** | 83.47 ms | 142.62 ms | 1.7x |
| **YOLOv8n** | 76.28 ms | 134.38 ms | 1.8x |
| **BERT** | 322.13 ms | 263.50 ms | 0.82x |

| 框架 | Prefill | Decode |
|------|---------|--------|
| **MNN LLM** | 267 tok/s | 60.8 tok/s |
| **llama.cpp** | 70 tok/s | 42.5 tok/s |

> Qwen2-0.5B · 4 线程

单算子测试 80+ 用例覆盖 Conv1x1 / MatMul / DWConv / LayerNorm / Softmax 等 7 类，完整数据见[单算子报告](results/single_op_benchmark_2026-06-09.md)。

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

**Q: MNN 比 ORT 快 2.5x 可信吗？**

在骁龙 865 + MobileNetV2 FP32 条件下真实。MNN 的 NCHW4c 内存布局 + NEON 手写汇编在小模型上有显著优势。MatMul 为主的模型（如 BERT）上 ORT 反超。

**Q: 如何添加新设备或框架？**

新设备：记录设备信息 → 跑 `benchctl run` → 提交 PR（详见 [CONTRIBUTING.md](CONTRIBUTING.md)）。新框架：实现 `BenchmarkBackend` 接口 → 添加 CMake option → 提交 PR。

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
