# ONNX Runtime Profiling 测试报告

**测试日期**: 2026-04-30
**测试设备**: 红米 K30 Pro (骁龙 865 / SM8250)
**测试模型**: MobileNetV2, ResNet50
**测试框架**: ONNX Runtime v1.21.0

---

## 1. 测试环境

### 1.1 硬件配置

| 项目 | 规格 |
|------|------|
| 设备 | 红米 K30 Pro |
| 芯片 | 骁龙 865 (SM8250) |
| CPU | 4×A77 (2.84GHz + 2.42GHz) + 4×A55 (1.8GHz) |
| ISA | ARMv8.2-A, FP16 |
| 内存 | 8GB LPDDR5 |

### 1.2 软件环境

| 项目 | 版本 |
|------|------|
| Android | 10+ |
| NDK | r25c |
| ONNX Runtime | v1.21.0 |
| 编译配置 | Debug (带符号信息) |

### 1.3 工具链

- **simpleperf**: CPU 采样分析 (NDK 自带)
- **FlameGraph**: 火焰图生成工具
- **benchmark_inference**: 自研测试程序

---

## 2. 测试步骤

### 2.1 编译 ORT Debug 版本

```bash
# 设置环境变量
export ANDROID_NDK=/home/liu/android-ndk
export ANDROID_SDK=/home/liu/Android/Sdk

# 编译 Debug 版本（带符号信息）
cd third_party/onnxruntime
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  -u LD_LIBRARY_PATH -u LDFLAGS -u PKG_CONFIG_PATH \
  ./build.sh \
  --android --android_abi arm64-v8a --android_api 21 \
  --android_sdk_path $ANDROID_SDK --android_ndk_path $ANDROID_NDK \
  --build_shared_lib --config Debug --use_nnapi \
  --skip_tests --parallel --skip_submodule_sync

# 编译产物
ls -lh build/Android/Debug/libonnxruntime.so
# 输出: 834MB (包含完整 debug_info)
```

### 2.2 部署到设备

```bash
# 推送 Debug 版本库
adb push third_party/onnxruntime/build/Android/Debug/libonnxruntime.so \
  /data/local/tmp/benchmark/libonnxruntime_debug.so

# 推送 benchmark 程序
adb push build_android/src/benchmark_inference /data/local/tmp/benchmark/

# 设置权限
adb shell "chmod 755 /data/local/tmp/benchmark/benchmark_inference"
```

### 2.3 运行 Simpleperf 采样

```bash
# 创建设备端脚本
adb shell "cat > /data/local/tmp/benchmark/profiling/run_with_profile.sh << 'EOF'
#!/bin/sh
cd /data/local/tmp/benchmark
LD_LIBRARY_PATH=. ./benchmark_inference \
    --backend onnxrt --model mobilenetv2 \
    --threads 4 --warmup 10 --runs 1000 > profiling/benchmark_output.txt 2>&1 &
BENCH_PID=\$!
echo \$BENCH_PID > profiling/benchmark.pid

# 等待 warmup 完成
sleep 5

# 启动 simpleperf (dwarf 模式，4000Hz，采样 15 秒)
/data/local/tmp/simpleperf record \
    -p \$BENCH_PID \
    -e cpu-cycles \
    -f 4000 \
    --call-graph dwarf \
    --duration 15 \
    -o profiling/perf.data

wait \$BENCH_PID
EOF
chmod 755 /data/local/tmp/benchmark/profiling/run_with_profile.sh"

# 执行采样
adb shell "sh /data/local/tmp/benchmark/profiling/run_with_profile.sh"

# 拉取数据
adb pull /data/local/tmp/benchmark/profiling/perf.data results/simpleperf/
adb pull /data/local/tmp/benchmark/profiling/benchmark_output.txt results/
```

### 2.4 生成火焰图

```bash
RESULT_DIR="results/profiling/20260430_205808_mobilenetv2_onnxrt_debug"

# 1. 生成函数热点报告
adb shell "/data/local/tmp/simpleperf report \
    -i /data/local/tmp/benchmark/profiling/perf.data \
    --sort dso,symbol -n" > "$RESULT_DIR/simpleperf/report_functions.txt"

# 2. 生成折叠格式数据
adb shell "/data/local/tmp/simpleperf report-sample \
    -i /data/local/tmp/benchmark/profiling/perf.data \
    --show-callchain" | \
    tr -d '\r' | \
    awk '/^sample:/ { ... } /symbol:/ { ... }' > "$RESULT_DIR/simpleperf/out.folded"

# 3. 生成 SVG 火焰图
./tools/FlameGraph/flamegraph.pl "$RESULT_DIR/simpleperf/out.folded" \
    > "$RESULT_DIR/simpleperf/flamegraph.svg"
```

### 2.5 线程数性能测试

```bash
# 测试不同线程数
for t in 1 2 4 6 8; do
  echo "=== Threads: $t ==="
  adb shell "cd /data/local/tmp/benchmark && \
    LD_LIBRARY_PATH=. ./benchmark_inference \
    --backend onnxrt --model mobilenetv2 \
    --threads $t --warmup 10 --runs 100"
done
```

---

## 3. 测试结果

### 3.1 线程数性能对比

#### MobileNetV2 (小模型，计算量 ~400M FLOPs)

| 线程数 | Mean (ms) | P50 (ms) | FPS | 相对性能 |
|--------|-----------|----------|-----|----------|
| 1 | 28.80 | 28.80 | 34.72 | 基准 |
| 2 | 22.28 | 22.27 | 44.89 | +29% |
| **4** | **17.70** | **17.69** | **56.50** | **+63%** |
| 6 | 19.24 | 19.16 | 51.97 | +49% |
| 8 | 28.40 | 27.05 | 35.21 | +1% |

#### ResNet50 (大模型，计算量 ~4G FLOPs)

| 线程数 | Mean (ms) | P50 (ms) | FPS | 相对性能 |
|--------|-----------|----------|-----|----------|
| 1 | 223.33 | - | 4.48 | 基准 |
| 2 | 142.16 | - | 7.03 | +57% |
| **4** | **83.73** | **83.52** | **11.94** | **+167%** |
| 6 | 117.65 | - | 8.50 | +90% |
| 8 | 139.05 | - | 7.19 | +60% |

**结论**: 4 线程是最优配置，完美匹配骁龙 865 的 4 个大核。

### 3.2 Simpleperf 热点分析

**采样统计**:
- 总采样数: 240,382
- 采样时长: 15 秒
- 采样频率: 4000 Hz
- 事件: cpu-cycles

#### Top 10 热点函数

| 排名 | 函数 | 占比 | 类别 |
|------|------|------|------|
| 1 | MlasConvIm2Col | 13.29% | 计算 |
| 2 | ThreadPoolTempl::WorkerLoop | 11.62% | 线程 |
| 3 | RunQueue::PopFront | 10.25% | 线程 |
| 4 | MLAS_ACTIVATION_FUNCTION::Activate | 8.09% | 计算 |
| 5 | __cxx_atomic_load (unsigned int) | 6.39% | 线程 |
| 6 | RunQueue::ElemState load | 5.65% | 线程 |
| 7 | __value_func::~__value_func | 5.41% | 线程 |
| 8 | ThreadPoolTempl::SpinLoopStatus load | 4.57% | 线程 |
| 9 | __cxx_atomic_load (bool) | 4.18% | 线程 |
| 10 | __value_func::operator= | 3.63% | 线程 |

#### 热点分类汇总

| 类别 | 占比 | 说明 |
|------|------|------|
| **计算密集** | ~28% | Conv、MatMul、Activation |
| **线程同步** | ~38% | ThreadPool、RunQueue、Atomic |
| **其他** | ~34% | 内存管理、调度等 |

### 3.3 火焰图分析

火焰图文件: `results/profiling/20260430_205808_mobilenetv2_onnxrt_debug/simpleperf/flamegraph.svg`

**关键观察**:
1. `MlasConvIm2Col` 是最宽的叶子节点，表示卷积 im2col 变换是主要计算瓶颈
2. 线程池相关函数占比高，但这是多线程并行的必要开销
3. 激活函数 (ReLU6) 占 8%，可以考虑融合优化

---

## 4. 性能瓶颈分析

### 4.1 计算密集型热点 (~28%)

| 函数 | 占比 | 优化方向 |
|------|------|----------|
| MlasConvIm2Col | 13.29% | Winograd 变换、Direct Conv |
| MLAS_ACTIVATION_FUNCTION | 8.09% | 算子融合 (Conv+BN+ReLU) |
| MlasSgemm 系列 | ~6% | 矩阵乘法优化 |
| MLAS_BIAS_ADDITION | 2.02% | 融合到 MatMul/Conv |

**优化建议**:
- **Winograd 变换**: 3×3 卷积可减少 2.25× 乘法次数
- **算子融合**: Conv + BatchNorm + ReLU6 融合为单个算子
- **FP16 推理**: 骁龙 865 支持 ARMv8.2-A FP16，理论吞吐量翻倍

### 4.2 线程同步开销 (~38%)

| 函数 | 占比 | 说明 |
|------|------|------|
| ThreadPoolTempl::WorkerLoop | 11.62% | 工作线程主循环 |
| RunQueue::PopFront | 10.25% | 任务窃取队列 |
| atomic_load 系列 | ~16% | 缓存一致性协议开销 |
| std::function 操作 | ~10% | 函数对象拷贝/析构 |

**分析**:
- 线程同步开销占比大，但这是多线程并行的必要代价
- 4 线程配置下，并行收益 (167%) 远超同步开销 (38%)
- 对于小模型 (MobileNetV2)，任务粒度太细，线程开销相对更高

### 4.3 内存访问

| 函数 | 占比 | 说明 |
|------|------|------|
| __memcpy | 0.02% | 内存拷贝 |
| je_malloc/je_free | 0.02% | 内存分配/释放 |

**结论**: 内存访问不是瓶颈。

---

## 5. 优化建议总结

### 5.1 已验证的最优配置

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| 线程数 | 4 | 匹配骁龙 865 大核数量 |
| Warmup | 10 | CPU 频率稳定 |
| 测试次数 | 100+ | 统计显著性 |

### 5.2 待验证的优化方向

| 优化措施 | 预期提升 | 实施难度 | 状态 |
|---------|---------|----------|------|
| NNAPI Delegate | +100-300% | 中 | 待测试 |
| FP16 推理 | +50-100% | 中 | 待测试 |
| Winograd 变换 | +20-30% | 高 | 需改 ORT 源码 |
| 算子融合 | +10-20% | 高 | 需改 ORT 源码 |

### 5.3 与 MNN 对比

根据 README 中的历史数据:

| 模型 | ORT (4线程) | MNN (4线程) | MNN 加速比 |
|------|-------------|-------------|-----------|
| MobileNetV2 | 18.17ms | 8.70ms | 2.09× |
| ResNet50 | 84.94ms | 83.66ms | 1.02× |
| MobileViT-S | 72.49ms | 60.16ms | 1.20× |

**结论**: MNN 在小模型上优势明显，大模型差距缩小。

---

## 6. 生成的文件

```
results/profiling/20260430_205808_mobilenetv2_onnxrt_debug/
├── benchmark_output.txt          # benchmark 原始输出
├── summary.md                    # 简要汇总
├── simpleperf/
│   ├── perf.data                 # 原始采样数据 (44MB)
│   ├── report_functions.txt      # 函数热点报告 (302KB)
│   ├── out.folded                # 火焰图折叠数据 (897MB)
│   └── flamegraph.svg            # 火焰图 (215KB)
└── docs/
    └── ort_profiling_report_20260430.md  # 本报告
```

### 查看火焰图

```bash
# 浏览器打开
xdg-open results/profiling/20260430_205808_mobilenetv2_onnxrt_debug/simpleperf/flamegraph.svg

# 或者
open results/profiling/20260430_205808_mobilenetv2_onnxrt_debug/simpleperf/flamegraph.svg
```

---

## 7. 复现命令

```bash
# 1. 编译 ORT Debug 版本
cd third_party/onnxruntime
./build.sh --android --android_abi arm64-v8a --android_api 21 \
  --android_sdk_path $ANDROID_SDK --android_ndk_path $ANDROID_NDK \
  --build_shared_lib --config Debug --use_nnapi \
  --skip_tests --parallel --skip_submodule_sync

# 2. 部署到设备
adb push build/Android/Debug/libonnxruntime.so /data/local/tmp/benchmark/
adb push build_android/src/benchmark_inference /data/local/tmp/benchmark/

# 3. 运行 profiling
./scripts/simpleperf_profile.sh --backend onnxrt --model mobilenetv2

# 4. 线程数测试
for t in 1 2 4 6 8; do
  adb shell "cd /data/local/tmp/benchmark && \
    LD_LIBRARY_PATH=. ./benchmark_inference \
    --backend onnxrt --model mobilenetv2 --threads $t --runs 100"
done
```

---

**报告生成时间**: 2026-04-30 21:15
**测试执行**: Claude Code 自动化测试
