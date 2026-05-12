---
name: performance-profiling
description: Use when analyzing inference latency bottlenecks — operator-level profiling with MNN/ORT built-in tools, CPU sampling with simpleperf, system tracing with atrace/perfetto
---

# Performance Profiling

## Overview

分析推理延迟瓶颈，从算子级别到系统级别。引用基础技能：android-device-ops、test-environment-control、result-processor。

## 方法对比

| 方法 | 粒度 | 适用场景 |
|------|------|---------|
| MNN_PROFILING | 逐算子 | MNN 各算子耗时分布 |
| ORT Profiling API | 逐算子 | ORT 各算子耗时分布 |
| simpleperf | CPU 采样 | 热点函数定位，火焰图 |
| atrace | 系统 trace | 系统调用、线程调度分析 |
| perfetto | 综合 trace | CPU/GPU/内存全面分析 |

## 集成入口

```bash
./scripts/profile_benchmark.sh [options]
```

选项：
| 选项 | 说明 | 默认 |
|------|------|------|
| `--backend <name>` | 后端选择 | mnn |
| `--model <name>` | 模型选择 | mobilenetv2 |
| `--runs <num>` | benchmark 运行次数 | 100 |
| `--threads <num>` | 线程数 | 4 |
| `--profile <tools>` | 工具集（逗号分隔） | simpleperf |
| `--build-type <type>` | 构建类型（release/debug） | release |
| `--simpleperf-duration <sec>` | simpleperf 采样时长 | 10 |
| `--atrace-duration <sec>` | atrace 捕获时长 | 5 |
| `--perfetto-duration <sec>` | perfetto 捕获时长 | 5 |
| `--perfetto-preset <preset>` | perfetto 预设（cpu/memory/full） | cpu |

工具集可选值：`simpleperf`, `atrace`, `perfetto`, `framework`, `all`

## 工作流

### 前置条件

```bash
# profiling 分析建议使用 Debug 版本以获得完整符号信息
./scripts/build_android.sh --debug
```

### MNN 逐算子 Profiling

```bash
adb shell "cd /data/local/tmp/benchmark && MNN_PROFILING=1 LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2 --backend mnn"
```

输出包含每个 MNN 算子的耗时和占比。

### ORT 逐算子 Profiling

通过 `--profiling` 参数启用：
```bash
adb shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2 --backend ort --profiling ort_profile.json"
```

输出 JSON 文件包含各算子耗时。

### simpleperf 火焰图

```bash
./scripts/simpleperf_profile.sh --backend mnn --model mobilenetv2 --duration 10
```

产物在 `results/profiling/<timestamp>_<model>_<backend>/simpleperf/flamegraph.svg`。

### atrace / perfetto

```bash
./scripts/atrace_capture.sh --duration 10 --categories sched,freq
./scripts/perfetto_trace.sh --duration 15
```

## 常见问题

- **火焰图没有符号**: 使用 Debug 构建重新编译（`build_android.sh --debug`）
- **MNN_PROFILING 无输出**: 确认使用 MNN backend，MNN 编译时未禁用 profiling
- **perfetto 无法启动**: Android 10+ 内置支持，检查 `adb shell perfetto --version`
- **profile_benchmark.sh 默认 release**: profiling 分析建议改为 `--build-type debug`