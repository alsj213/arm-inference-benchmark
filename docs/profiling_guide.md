# CPU 性能分析指南

本指南介绍如何使用 simpleperf、atrace、perfetto 和框架内置 profiling 工具来分析推理性能瓶颈。

---

## 1. 概述

### 为什么需要 CPU Profiling？

Benchmark 告诉你"多慢"，Profiling 告诉你"为什么慢"。

| 工具 | 定位 | 输出 | 查看方式 |
|------|------|------|----------|
| **simpleperf** | 函数级 CPU 热点 | 火焰图、热点报告 | 终端/浏览器 |
| **atrace** | 系统级调度 trace | 线程调度、CPU 频率 | chrome://tracing |
| **perfetto** | 综合 trace | CPU/内存/GPU 全景 | ui.perfetto.dev |
| **框架 profiling** | 逐算子耗时 | 各层执行时间 | 终端/JSON |

### 选择决策树

```
想要什么？
├── 知道 CPU 时间花在哪个函数 → simpleperf
├── 看线程在哪个核运行、是否降频 → atrace / perfetto
├── 看推理各算子耗时 → 框架内置 profiling
└── 全面分析 → profile_benchmark.sh --profile all
```

---

## 2. 环境准备

### 2.1 NDK 安装

simpleperf 需要 Android NDK：

```bash
# 下载 NDK r25c
wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip
unzip android-ndk-r25c-linux.zip

# 配置环境变量
export ANDROID_NDK=/path/to/android-ndk-r25c
```

### 2.2 ADB 配置（WSL2）

```bash
# WSL2 环境
export PATH=$PATH:/mnt/e/andorid/adb/
alias adb='/mnt/e/andorid/adb/adb.exe'

# 验证连接
adb devices
```

### 2.3 工具检查

```bash
# 检查 simpleperf（NDK r25+ 路径）
ls $ANDROID_NDK/simpleperf/bin/android/arm64/simpleperf

# 检查 perfetto（设备上）
adb shell which perfetto

# 检查 atrace（设备上）
adb shell which atrace

# 检查 FlameGraph（用于生成火焰图）
ls tools/FlameGraph/flamegraph.pl
# 如果没有，克隆：
# git clone https://github.com/brendangregg/FlameGraph.git tools/FlameGraph
```

---

## 3. simpleperf 详解

### 3.1 原理

simpleperf 使用 Linux `perf_event_open` 系统调用进行 CPU 采样，记录函数调用栈和执行时间。

### 3.2 快速开始

```bash
# 方式一：使用集成脚本（推荐，自动生成火焰图）
./scripts/simpleperf_profile.sh --backend mnn --model mobilenetv2

# 方式二：使用 profile_benchmark.sh
./scripts/profile_benchmark.sh --backend mnn --model mobilenetv2 --profile simpleperf

# 方式三：手动执行
adb shell "cd /data/local/tmp/benchmark && \
    LD_LIBRARY_PATH=. nohup ./benchmark_inference \
    --backend mnn --model mobilenetv2 --runs 1000 > /dev/null 2>&1 &"

PID=$(adb shell pidof benchmark_inference)
adb shell "/data/local/tmp/simpleperf record \
    -p $PID -e cpu-cycles -f 4000 --call-graph fp --duration 10 -o perf.data"
adb pull /data/local/tmp/benchmark/profiling/perf.data .
```

**自动生成火焰图：**
脚本会自动检测 FlameGraph 工具，如果存在则生成 `flamegraph.svg` 文件。
```bash
# 确保 FlameGraph 已克隆
git clone https://github.com/brendangregg/FlameGraph.git tools/FlameGraph
```

### 3.3 参数详解

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `-e` | 采样事件 | `cpu-cycles`（默认）、`cache-misses`、`branch-misses` |
| `-f` | 采样频率 Hz | 4000（平衡精度和开销） |
| `--call-graph` | 调用图模式 | `fp`（快速）、`dwarf`（更准确） |
| `--duration` | 采样时长秒 | 10-30 |

### 3.4 采样事件选择

| 事件 | 用途 |
|------|------|
| `cpu-cycles` | CPU 周期，定位热点函数 |
| `cache-misses` | Cache 未命中，分析内存访问 |
| `branch-misses` | 分支预测失败，分析分支逻辑 |
| `instructions` | 指令数，计算 IPC |

### 3.5 生成报告

```bash
# 函数热点报告
simpleperf report -i perf.data --sort dso,symbol -n

# DSO（共享库）报告
simpleperf report -i perf.data --sort dso -n

# 带调用链的报告
simpleperf report -i perf.data --show-callchain
```

### 3.6 火焰图生成

```bash
# 1. 导出为 FlameGraph 格式
simpleperf report-sample --show-callchain > out.perf

# 2. 使用 FlameGraph 工具
git clone https://github.com/brendangregg/FlameGraph.git
FlameGraph/stackcollapse-perf.pl out.perf > out.folded
FlameGraph/flamegraph.pl out.folded > flamegraph.svg

# 3. 用浏览器打开 flamegraph.svg
```

### 3.7 结果解读

**热点报告示例：**
```
Overhead  Shared Object       Symbol
  45.23%  libMNN.so           MNN::ConvolutionDepthwise::execute()
  12.45%  libMNN.so           MNN::WinogradTransform::transform()
   8.67%  libc.so             memcpy
   5.32%  libMNN.so           MNN::MatrixMul::compute()
```

**火焰图阅读：**
- X 轴：采样占比（越宽 = 耗时越多）
- Y 轴：调用栈深度（越深 = 调用链越长）
- 颜色：随机分配，无特殊含义

---

## 4. atrace 详解

### 4.1 原理

atrace 是 Android 对 Linux ftrace 的封装，可以追踪系统级事件（调度、频率、IPC 等）。

### 4.2 快速开始

```bash
# 方式一：使用集成脚本
./scripts/atrace_capture.sh --backend mnn --model mobilenetv2

# 方式二：使用 profile_benchmark.sh
./scripts/profile_benchmark.sh --backend mnn --model mobilenetv2 --profile atrace

# 方式三：手动执行（同步模式）
adb shell "atrace -t 5 -b 32768 sched freq idle > /data/local/tmp/trace.txt"
adb pull /data/local/tmp/trace.txt .
```

### 4.3 常用类别

| 类别 | 说明 | 推理场景 |
|------|------|----------|
| `sched` | CPU 调度 | 线程在哪个核运行 |
| `freq` | CPU 频率 | 是否降频 |
| `idle` | CPU idle | CPU 利用率 |
| `binder_driver` | Binder IPC | 进程间通信 |
| `gfx` | 图形 | GPU 相关 |
| `view` | View 系统 | UI 渲染 |

### 4.4 查看 trace

**方式一：Perfetto UI（推荐）**
1. 打开 https://ui.perfetto.dev
2. 点击 "Open trace file"
3. 选择 `trace.txt` 文件

**方式二：chrome://tracing**
1. 打开 Chrome 浏览器
2. 地址栏输入 `chrome://tracing`
3. 点击 "Load" 按钮
4. 选择 `trace.txt` 文件

**注意：** atrace 输出的是 ftrace 格式，Perfetto UI 支持更好的兼容性。如果 chrome://tracing 无法打开，请使用 Perfetto UI。

**快捷键：**
- `W/S`：放大/缩小
- `A/D`：左/右移动
- `1/2/3/4`：切换视图模式

### 4.5 结果解读

**CPU 调度分析：**
- 长条：线程在某个 CPU 核上运行
- 空白：线程等待或被抢占
- 颜色：不同线程

**CPU 频率分析：**
- 高频：性能模式
- 低频：省电模式或降频

---

## 5. perfetto 详解

### 5.1 原理

perfetto 是 atrace 的继任者，提供更强大的 trace 能力和更好的可视化。

### 5.2 快速开始

```bash
# 方式一：使用集成脚本（推荐）
./scripts/perfetto_trace.sh --backend mnn --model mobilenetv2

# 方式二：使用 profile_benchmark.sh
./scripts/profile_benchmark.sh --backend mnn --model mobilenetv2 --profile perfetto

# 方式三：手动执行
# 生成配置文件
cat > config.pbtx << EOF
buffers: { size_kb: 32768 fill_policy: RING_BUFFER }
duration_ms: 5000
data_sources: { config { name: "linux.ftrace" ftrace_config {
    ftrace_events: "sched/sched_switch"
    ftrace_events: "power/cpu_frequency"
    atrace_categories: "sched"
    atrace_categories: "freq"
    atrace_apps: "*"
}}}
data_sources: { config { name: "linux.process_stats"
    process_stats_config { scan_all_processes_on_start: true }
}}
EOF

# 推送配置并执行（注意：输出必须在 /data/misc/perfetto-traces/ 目录）
adb push config.pbtx /data/local/tmp/benchmark/profiling/
adb shell "cat /data/local/tmp/benchmark/profiling/config.pbtx | \
    perfetto --txt -c - -o /data/misc/perfetto-traces/trace.perfetto-trace"
adb shell "cp /data/misc/perfetto-traces/trace.perfetto-trace /data/local/tmp/benchmark/profiling/"
adb pull /data/local/tmp/benchmark/profiling/trace.perfetto-trace .
```

### 5.3 预设配置

| 预设 | 内容 | 用途 |
|------|------|------|
| `cpu` | 调度 + 频率 + 进程 | 推理性能分析 |
| `memory` | 内存分配 + LMK | 内存分析 |
| `full` | CPU + 内存 + GPU + Binder | 全面分析 |

### 5.4 查看 trace

1. 打开 https://ui.perfetto.dev
2. 点击 "Open trace file"
3. 选择 `trace.perfetto-trace` 文件

**功能：**
- SQL 查询：可以查询 trace 数据
- 时间线：可视化事件时间线
- 统计：自动计算统计信息

---

## 6. 框架内置 Profiling

### 6.1 MNN

```bash
# 使用环境变量开启 profiling
adb shell "cd /data/local/tmp/benchmark && \
    MNN_OPENCL_PROFILE=1 \
    LD_LIBRARY_PATH=. ./benchmark_inference \
    --backend mnn --model mobilenetv2 --runs 10"
```

输出会显示每个算子的执行时间。

### 6.2 ONNX Runtime

ORT profiling 需要在代码中启用：

```cpp
SessionOptions session_options;
session_options.EnableProfiling("profile_output.json");
```

生成的 JSON 文件可用 `chrome://tracing` 打开。

### 6.3 ncnn

```bash
# 编译时开启 NCNN_PROFILE
cmake -DNCNN_PROFILE=ON ..

# 运行时自动输出各层耗时
adb shell "cd /data/local/tmp/benchmark && \
    NCNN_PROFILE=1 \
    LD_LIBRARY_PATH=. ./benchmark_inference \
    --backend ncnn --model mobilenetv2 --runs 10"
```

### 6.4 TFLite

```bash
# 使用 benchmark_model 工具
adb shell "benchmark_model \
    --graph=/data/local/tmp/benchmark/models/classification/mobilenetv2/mobilenetv2.tflite \
    --enable_op_profiling=true \
    --num_threads=4"
```

---

## 7. 集成使用

### 7.1 一键 Profiling

```bash
# 所有工具同时采集
./scripts/profile_benchmark.sh \
    --backend mnn \
    --model mobilenetv2 \
    --threads 4 \
    --runs 100 \
    --profile all

# 只采集 simpleperf
./scripts/profile_benchmark.sh \
    --backend mnn \
    --model mobilenetv2 \
    --profile simpleperf
```

### 7.2 批量 Profiling

```bash
# 对所有 backend 做 profiling
for backend in mnn onnxrt ncnn tflite; do
    ./scripts/profile_benchmark.sh \
        --backend $backend \
        --model mobilenetv2 \
        --profile simpleperf \
        --simpleperf-duration 10
done
```

### 7.3 结果目录结构

目录结构根据 `--profile` 参数动态生成，只创建启用工具的目录：

```
results/profiling/
└── 20260426_143000_mobilenetv2_mnn/
    ├── device_info.txt           # CPU、温度等详细信息
    ├── benchmark_output.txt      # benchmark 原始输出
    ├── thermal_pre.txt           # 测试前温度
    ├── thermal_post.txt          # 测试后温度
    ├── summary.md                # 汇总报告（仅 profile_benchmark.sh）
    ├── simpleperf/               # --profile simpleperf
    │   ├── perf.data             # 原始采样数据
    │   ├── report_functions.txt  # 函数热点报告
    │   ├── report_dso.txt        # 共享库报告
    │   ├── report_callchain.txt  # 调用链报告
    │   ├── out.folded            # 火焰图折叠数据
    │   └── flamegraph.svg        # 火焰图（自动生成）
    ├── atrace/                   # --profile atrace
    │   └── trace.txt             # 系统 trace（ftrace 格式）
    ├── perfetto/                 # --profile perfetto
    │   ├── trace.perfetto-trace  # perfetto trace
    │   └── config.pbtx           # 配置文件
    └── framework/                # --profile framework
        ├── mnn_profile.txt       # MNN 逐算子耗时
        └── ort_profile.json      # ORT profiling trace
```

**示例：**
```bash
# 只生成 simpleperf 目录
./scripts/profile_benchmark.sh --backend mnn --model mobilenetv2 --profile simpleperf

# 生成 simpleperf + perfetto 目录
./scripts/profile_benchmark.sh --backend mnn --model mobilenetv2 --profile simpleperf,perfetto
```

---

## 8. 常见问题 (FAQ)

### Q1: simpleperf 报 "not supported"

**原因：** 内核未启用 `perf_event_open` 支持

**解决：**
```bash
# 检查内核配置
adb shell "zcat /proc/config.gz | grep CONFIG_PERF_EVENTS"

# 如果没有输出，需要刷入支持 perf 的内核
```

### Q2: 非 root 设备如何使用 simpleperf

**解决：**
```bash
# 从 NDK 推送 simpleperf（支持非 root）
# NDK r25+ 路径
adb push $ANDROID_NDK/simpleperf/bin/android/arm64/simpleperf /data/local/tmp/
adb shell chmod 755 /data/local/tmp/simpleperf

# 使用 -g 参数（需要 root）或 --no-callchain-jit（非 root）
adb shell "/data/local/tmp/simpleperf record -p PID --no-callchain-jit ..."
```

### Q3: 采样数据不完整

**原因：** 采样频率太高或 buffer 太小

**解决：**
```bash
# 降低采样频率
simpleperf record -f 1000 ...  # 从 4000 降到 1000

# 增加 buffer
simpleperf record -m 16384 ...  # 增加 buffer 到 16MB
```

### Q4: WSL2 下 adb 连接问题

**解决：**
```bash
# 重启 adb server
/mnt/e/andorid/adb/adb.exe kill-server
/mnt/e/andorid/adb/adb.exe start-server

# 检查设备
/mnt/e/andorid/adb/adb.exe devices
```

### Q5: perfetto 报 "permission denied"

**原因：** 非 root 设备权限不足

**解决：**
```bash
# 使用 shell 用户权限
adb shell "perfetto --txt -c config.pbtx -o trace.perfetto-trace --background"

# 或者使用 atrace 替代
adb shell "atrace --async_start -c sched,freq,idle"
```

---

## 9. 最佳实践

### 9.1 测试前准备

1. **关闭后台应用**：减少干扰
2. **开启飞行模式**：避免网络中断
3. **冷却手机**：避免降频影响结果
4. **固定 CPU 频率**（可选）：
   ```bash
   adb shell "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
   ```

### 9.2 采样参数选择

| 场景 | 采样频率 | 时长 | 事件 |
|------|----------|------|------|
| 快速定位热点 | 1000 Hz | 5s | cpu-cycles |
| 详细分析 | 4000 Hz | 10-30s | cpu-cycles |
| Cache 分析 | 1000 Hz | 10s | cache-misses |
| 分支分析 | 1000 Hz | 10s | branch-misses |

### 9.3 结果验证

1. **多次采样**：至少 3 次，确认结果一致
2. **对比分析**：不同 backend 使用相同参数
3. **温度监控**：确保测试期间温度稳定

---

## 10. 参考资料

- [simpleperf 官方文档](https://developer.android.com/ndk/guides/simpleperf)
- [Perfetto 官方文档](https://perfetto.dev/docs/)
- [Android 性能分析最佳实践](https://developer.android.com/topic/performance)
- [FlameGraph 工具](https://github.com/brendangregg/FlameGraph)
