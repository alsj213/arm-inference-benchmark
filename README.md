# ARM Inference Benchmark

[![Platform](https://img.shields.io/badge/platform-Android-lightgrey)](https://developer.android.com)
[![Architecture](https://img.shields.io/badge/architecture-ARM64-brightgreen)](https://developer.arm.com/architectures)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](https://github.com/alsj213/arm-inference-benchmark/actions)
[![Pages](https://img.shields.io/badge/docs-GitHub_Pages-green)](https://alsj213.github.io/arm-inference-benchmark/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Cross-framework ML inference benchmarking on ARM mobile devices.**
端侧深度学习推理框架性能基准测试 — fair, reproducible, verifiable.

---

## Why This Project

Comparing ML inference frameworks is hard: each has its own benchmark tool, different warm‑up policies, incompatible metric definitions. Results rarely agree with each other.

**ARM Inference Benchmark** solves this by:
- **One harness, many frameworks** — same input, same timer, same device, same protocol
- **Verification built in** — every result is cross‑checked against the framework's own native tool; deviations >5% are flagged
- **Full audit trail** — every run is stored in SQLite with git commit, timestamp, device temperature

```bash
benchctl run cnn resnet50 -f mnn,ort         # run benchmark
benchctl history mnn resnet50                 # vertical trend
benchctl verify mnn resnet50                  # native tool cross‑check
benchctl export mobilenetv2 -f mnn,ort --html # report
```

---

## Benchmarks

### Model Benchmarks (CNN / NLP)

> Device: Redmi K30 Pro · Snapdragon 865 (SM8250) · FP32 · 4 threads

| Model | MNN | ORT | TVM | MNN vs ORT |
|-------|-----|-----|-----|------------|
| **MobileNetV2** | 8.55 ms | 21.68 ms | — | **2.5x** 🏆 |
| **ResNet50** | 83.47 ms | 142.62 ms | 64.38 ms | 1.7x |
| **YOLOv8n** | 76.28 ms | 134.38 ms | 126.63 ms | 1.8x |
| **BERT** | 322.13 ms | 263.50 ms | — | 0.82x |

### LLM Benchmarks

> Qwen2-0.5B · 4 threads

| Framework | Prefill (tok/s) | Decode (tok/s) |
|-----------|-----------------|-----------------|
| **MNN LLM** | 267.09 | 60.84 |
| **llama.cpp** | 70.07 | 42.54 |

### Single‑Operator Benchmarks

80+ test cases across 7 categories: Conv1x1, DWConv, MatMul, LayerNorm, Softmax, GELU, Misaligned Conv.

*Full data: [`results/single_op_benchmark_2026-06-09.md`](results/single_op_benchmark_2026-06-09.md)*

---

## Quick Start

### Prerequisites

- **Android NDK** (r26+)
- **ADB** + Android 10+ device (ARM64)
- **Python 3.10+** (for benchctl)
- **Click** (`pip install click pyyaml`)

### 1. Clone & Init

```bash
git clone --recurse-submodules https://github.com/alsj213/arm-inference-benchmark.git
cd arm-inference-benchmark
```

### 2. Build

```bash
export ANDROID_NDK=/path/to/android-ndk
cmake -B build_android \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
    -DBENCHMARK_MNN=ON -DBENCHMARK_ORT=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build_android --target benchmark_inference -j$(nproc)
```

### 3. Run

```bash
# One‑shot
benchctl run cnn mobilenetv2 -f mnn

# Compare frameworks
benchctl run cnn mobilenetv2 -f mnn,ort

# See history
benchctl history mnn mobilenetv2

# Export report
benchctl export mobilenetv2 -f mnn,ort --format html
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  benchctl (Python CLI, host‑side)            │
│  run · history · verify · export · db        │
└──────────────┬──────────────────────────────┘
               │ ADB
┌──────────────▼──────────────────────────────┐
│  benchmark_inference (C++, device‑side)      │
│  JSON Lines → stdout                         │
│                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │ CNN     │ │ LLM     │ │ SingleOp│        │
│  │ P50/P99 │ │ TTFT/TPS│ │ per‑op  │        │
│  └────┬────┘ └────┬────┘ └────┬────┘        │
│       └───────────┼───────────┘              │
│              ┌────▼────┐                     │
│              │ Harness  │   unified timer     │
│              └────┬────┘                     │
│    ┌──────────────┼──────────────┐           │
│    ▼              ▼              ▼           │
│  MNN    ONNX Runtime    TVM    llama.cpp     │
└──────────────────────────────────────────────┘
```

**Key design decisions:**

- **LoadGen‑style separation** — the timer lives in the harness, not in framework code
- **JSON Lines protocol** — C++ binaries output one JSON object per line; benchctl consumes them
- **SQLite audit trail** — every run is recorded with commit hash, timestamp, device temp
- **Native tool verification** — `benchctl verify` runs the framework's own benchmark tool and flags deviations >5%

---

## Supported Frameworks

| Framework | Status | Role |
|-----------|--------|------|
| **MNN** (3.6.1) | ✅ Active | 主力测试 — CPU + LLM |
| **ONNX Runtime** (1.28.0) | ✅ Active | Accuracy baseline |
| **TVM** (0.15.0) | ⏸ Disabled | Relay backend; v0.25 API migration needed |
| **llama.cpp** (b10121) | ✅ Active | LLM benchmark baseline |
| ~~NCNN~~ | Removed | — |
| ~~TFLite~~ | Removed | — |
| ~~TNN~~ | Removed | — |
| ~~QNN~~ | Removed | — |
| ~~MindSpore Lite~~ | Removed | — |

---

## Commands

```
benchctl
├── run      CNN/LLM/单算子 benchmark
│   benchctl run cnn resnet50 -f mnn,ort -t 4 -r 50
│
├── history  纵向性能趋势 + 回归检测
│   benchctl history mnn resnet50 --last 10
│
├── verify   原生工具交叉验证
│   benchctl verify mnn resnet50
│
├── export   报告导出 (JSON / HTML)
│   benchctl export resnet50 -f mnn,ort --format html
│
└── db       数据库管理
    benchctl db init | stats
```

---

## Methodology

### Measurement Protocol

Inspired by [MLPerf Inference: Mobile](https://arxiv.org/abs/2012.02328):

1. **Fixed random seed** (42) — all frameworks see identical input
2. **Warmup** (10 iterations) — exclude cold‑start effects
3. **Measurement** (50+ iterations) — single‑stream, synchronous inference
4. **Metrics** — P50 / P90 / P99 latency, throughput (FPS), peak memory
5. **Accuracy** — cosine similarity vs ONNX Runtime FP32 reference

### Verification

Every result is validated against the framework's own tool:

```
Framework │ Harness P50 │ Native Tool    │ Deviation │ Verdict
──────────┼─────────────┼────────────────┼───────────┼────────
MNN       │ 8.58 ms     │ 5.39 ms (avg)  │ +59.0%    │ ⚠️ check
```

*Note: MNN's native `benchmark.out` measures raw forward time while the harness measures full inference cycle (memcpy → run → memcpy). The deviation is expected and documented.*

### Track Definitions

| Track | Metrics | Models |
|-------|---------|--------|
| **CNN** | P50, P90, P99, FPS, Memory | MobileNetV2, ResNet50, YOLOv8n, BERT |
| **LLM** | TTFT, TPS, Memory | Qwen2‑0.5B, Qwen3‑4B |
| **SingleOp** | Per‑op latency (mean) | Conv1x1, MatMul, DWConv, LayerNorm, … |

---

## Project Structure

```
benchmark/
├── benchctl/          # Python CLI (host‑side orchestrator)
│   ├── cli.py         # Click CLI commands
│   ├── db.py          # SQLite database
│   ├── runner.py      # ADB device orchestration
│   ├── verify.py      # Native tool verification
│   ├── report.py      # HTML/JSON report generation
│   └── tracks/        # Track‑specific logic (cnn/llm/single_op)
├── src/               # C++ device‑side code
│   ├── cnn/           # CNN benchmark binary
│   ├── llm/           # LLM benchmark binary
│   ├── single_op/     # Single‑operator benchmark
│   ├── common/        # Harness, config, utilities
│   ├── backends/      # Framework wrappers (5 backends)
│   └── models/        # Model info definitions
├── third_party/       # Git submodules (MNN, ORT, TVM, llama.cpp)
├── scripts/           # Shell scripts (build, convert, profile)
├── models/            # Model files (symlink to external storage)
├── docs/              # Documentation + analysis reports
├── results/           # Benchmark output
└── templates/         # HTML report templates
```

---

## Development

### CI/CD

GitHub Actions runs cross‑compilation on every push to verify the build stays green.

### Branch Strategy

- `main` — stable, all tests pass
- `feat/*` — feature branches, merged via PR
- `backup/*` — historical backups

### Submodule Tags

| Submodule | Tag |
|-----------|-----|
| MNN | `3.6.1` |
| ONNX Runtime | `v1.28.0` |
| TVM | custom (v0.15.dev0) |
| llama.cpp | `b10121` |

---

## FAQ

**Q: Why not use each framework's own benchmark tool?**
A: Framework‑native tools are great for *verification* (see `benchctl verify`), but they use different measurement protocols — you can't compare them directly. Our unified harness ensures apples‑to‑apples comparison.

**Q: Is MNN's 2.5x speedup over ORT real?**
A: Yes, for MobileNetV2 FP32 on Snapdragon 865. MNN's NCHW4c layout + hand‑written NEON assembly give it a significant edge on small CV models. However, ORT pulls ahead on MatMul‑heavy models like BERT.

**Q: How do I add a new framework?**
A: Implement the `BenchmarkBackend` interface in `src/backends/`, add the CMake option, and create a model converter script if needed. See `BACKEND_REENABLE_GUIDE.md`.

---

## License

MIT License — see [LICENSE](LICENSE)
