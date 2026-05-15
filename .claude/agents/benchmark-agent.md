---
name: benchmark-agent
description: 基准测试自动化流程。当用户说"跑 benchmark"或"测试某模型"时，按此流程执行。
---

# Benchmark 执行流程

本文件不是 subagent，而是定义了一个**强制流程**。当用户要求执行基准测试时，必须按以下步骤执行，每步输出证物。

## 执行前验证

在运行任何 benchmark 前，必须先执行以下验证。**任一步失败则停止并向用户报告**：

```bash
# 1. 设备在线
/mnt/e/andorid/adb/adb.exe devices | grep -q "b08dee23" && echo "OK" || echo "FAIL"

# 2. 设备型号
/mnt/e/andorid/adb/adb.exe shell getprop ro.product.model

# 3. CPU 确认  
/mnt/e/andorid/adb/adb.exe shell cat /proc/cpuinfo | grep "A77" | head -1

# 4. NDK 就绪
ls /home/liu/android-ndk/build/cmake/android.toolchain.cmake && echo "OK" || echo "FAIL"
```

## 执行流程

### 1. 编译（如需）

```bash
export ANDROID_NDK=/home/liu/android-ndk
./scripts/build_android.sh
```

成功后记录：
```bash
ls -lh build_android/src/benchmark_inference
md5sum build_android/src/benchmark_inference
```

### 2. 模型检查

```bash
# 目标模型文件存在？
ls models/nlp/bert/bert.onnx 2>/dev/null || python scripts/download_pretrained.py
ls models/nlp/bert/bert_MNN.mnn 2>/dev/null || { ./scripts/build_host_tools.sh && ./scripts/convert_models.sh; }
```

### 3. 运行（保存原始日志）

```bash
export ANDROID_NDK=/home/liu/android-ndk
./scripts/run_benchmark_android.sh \
  --build-type release \
  --backend <backend> \
  --model <model> \
  --threads <N> \
  --runs <N> \
  2>&1 | tee results/benchmark_$(date +%Y%m%d_%H%M%S).log
```

### 4. 输出摘要

从 tee 保存的原始日志中，用 grep 提取真实数据：

```bash
grep -E "P50|P90|Init|Throughput|Cosine|FPS" results/*.log
```

然后以表格形式呈现给用户。

## 数据真实性红线

- **禁止**编造性能数字。所有数字必须从真实 adb shell 输出中提取
- **禁止**跳过设备握手。不连接设备不执行
- **禁止**修改 benchmark_inference 源码来"加速"结果
- **允许**重复运行测试以验证数据稳定性
- **允许**在报告中注明测试条件（设备温度、是否锁频等）
