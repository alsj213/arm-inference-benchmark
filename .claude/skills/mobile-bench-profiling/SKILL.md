---
name: mobile-bench-profiling
description: Use when analyzing inference latency bottlenecks — operator-level profiling, CPU sampling with simpleperf, system tracing with atrace/perfetto, and flame graph generation
---

# Mobile Bench Profiling

## Overview

分析推理延迟瓶颈的方法集，从算子级别到系统级别。

配置通过项目根目录的 `.benchmarkrc.yml` 读取。

## 方法对比

| 方法 | 粒度 | 适用场景 | 前置条件 |
|------|------|---------|---------|
| MNN_PROFILING | 逐算子 | MNN 各算子耗时分布 | MNN 编译时未禁用 profiling |
| ORT Profiling API | 逐算子 | ORT 各算子耗时分布 | — |
| TVM Debug Runtime | 逐算子 | TVM Relax VM 各算子耗时 | TVM 编译时启用 debug runtime |
| NCNN_PROFILING | 逐算子 | ncnn 各层耗时分布 | ncnn 编译时未禁用 profiling |
| simpleperf | CPU 采样 | 热点函数定位，火焰图 | Debug 构建获得完整符号 |
| atrace | 系统 trace | 系统调用、线程调度分析 | Android 10+ |
| perfetto | 综合 trace | CPU/GPU/内存全面分析 | Android 10+ 内置支持 |

## 集成入口

```bash
./scripts/profile/profile-benchmark.sh [options]
```

| 选项 | 说明 | 默认 |
|------|------|------|
| `--backend <name>` | 后端选择 (mnn, ort, ncnn, tvm, mslite, llamacpp) | mnn |
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

## 前置条件

```bash
# profiling 分析建议使用 Debug 版本以获得完整符号信息
./scripts/build/build-android.sh --debug
```

## MNN 逐算子 Profiling

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")

$ADB shell "cd /data/local/tmp/benchmark && MNN_PROFILING=1 LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2 --backend mnn"
```

输出包含每个 MNN 算子的耗时和占比。

## ORT 逐算子 Profiling

通过 `--profiling` 参数启用：

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")

$ADB shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2 --backend ort --profiling ort_profile.json"
```

输出 JSON 文件包含各算子耗时。

## TVM 逐算子 Profiling

### TVM Debug Runtime

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")

# TVM 编译时使用 debug runtime 获得逐算子耗时
$ADB shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2 --backend tvm"
```

TVM Relax VM 输出包含每个 VM 指令的耗时和调用次数。

### TVM RPC Profiling

通过 TVM RPC 进行远程 profiling：

```bash
# 在主机端启动 RPC tracker
python3 -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190

# 手机端启动 RPC server
$ADB reverse tcp:9190 tcp:9190
$ADB shell "cd /data/local/tmp/benchmark && python3 tvm_rpc_server.py --tracker 127.0.0.1:9190"
```

## ncnn 逐算子 Profiling

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")

$ADB shell "cd /data/local/tmp/benchmark && NCNN_PROFILING=1 LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2 --backend ncnn"
```

## simpleperf 火焰图

```bash
./scripts/profile/simpleperf-profile.sh --backend mnn --model mobilenetv2 --duration 10
```

产物在 `results/profiling/<timestamp>_<model>_<backend>/simpleperf/flamegraph.svg`。

### 手动生成火焰图

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")

# 1. simpleperf 采样
$ADB shell simpleperf record -o /data/local/tmp/perf.data -e cpu-cycles:u -f 4000 --duration 10 \
  --app benchmark_inference

# 2. 拉取数据
$ADB pull /data/local/tmp/perf.data .

# 3. 生成火焰图
simpleperf report -i perf.data --full-call-graph | \
  stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

产物在 `results/profiling/<timestamp>_<model>_<backend>/simpleperf/flamegraph.svg`。

### LLM / VL 火焰图（进程退出型模型）

CNN 模型使用 `--app` 方式 attach 到常驻进程。LLM benchmark 执行完即退出，需要用**命令行模式**：

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")

# 前置：推送 Debug 构建（获取完整函数符号）
$ADB push build_android_debug/third_party/MNN/libMNN.so /data/local/tmp/benchmark/
$ADB push build_android_debug/src/llm/llm_benchmark /data/local/tmp/benchmark/

# 1. simpleperf 记录（--call-graph fp 记录调用链）
$ADB shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=. \
  ./simpleperf record -o /data/local/tmp/perf.data \
  -e cpu-cycles -f 4000 --call-graph fp --duration 20 \
  -- ./llm_benchmark --backend mnn_llm \
    --image test.raw --image-size 420 420 \
    --prompt 'describe' --max-tokens 32"

# 2. 拉取数据
$ADB pull /data/local/tmp/perf.data .

# 3. 生成火焰图
simpleperf report -i perf.data --full-call-graph | \
  stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

说明：
- LLM/VL 推理总耗时 ~17s，`--duration 20` 确保覆盖 vision + prefill + decode 全阶段
- `--call-graph fp` 使用帧指针记录调用链（无法用 `--app` attach，必须用命令行模式）
- Debug 构建带符号，火焰图中显示函数名而非裸地址

## atrace

```bash
./scripts/profile/atrace-capture.sh --duration 10 --categories sched,freq
```

## perfetto

```bash
./scripts/profile/perfetto-trace.sh --duration 15
```

Android 10+ 内置支持 perfetto。

## 常见问题

- **火焰图没有符号**: 使用 Debug 构建重新编译（`./scripts/build/build-android.sh --debug`）
- **MNN_PROFILING 无输出**: 确认使用 MNN backend，MNN 编译时未禁用 profiling
- **perfetto 无法启动**: Android 10+ 内置支持，检查 `adb shell perfetto --version`
- **profile-benchmark.sh 默认 release**: profiling 分析建议改为 `--build-type debug`

## 关联 Skill

- [mobile-bench-run](../mobile-bench-run/SKILL.md) — 完整基准测试流程
- [mobile-bench-model-prep](../mobile-bench-model-prep/SKILL.md) — 模型准备与编译
- [mobile-bench-integrate](../mobile-bench-integrate/SKILL.md) — 集成新推理框架后端
