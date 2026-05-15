---
name: run-benchmark
description: Use when running inference performance tests on Android devices — full workflow from build to report, including environment setup and result processing
---

# Run Benchmark

## Overview

端侧推理性能基准测试完整流程。引用基础技能：model-pipeline、android-device-ops、test-environment-control、result-processor。

## 完整流程

```text
设备握手 → 代码版本 → 二进制验证 → 模型检查 → 锁频 → 执行(tee日志) → 恢复环境 → 数据提取 → 报告
```

## 数据真实性要求

- **必须** `tee` 保存原始日志，不得仅靠 stdout
- **必须** 记录二进制 md5 和模型文件大小
- **必须** 在报告中标注每行数据的来源命令
- **必须** 检查设备温度，> 45°C 时标注降频风险

## 一键构建并运行

```bash
./scripts/build_and_run.sh --backend all --model all --threads 1
```

常用选项：
| 选项 | 说明 | 默认 |
|------|------|------|
| `--backend <mnn\|onnxrt\|ort\|all>` | 后端选择 | all |
| `--model <name\|all>` | 模型选择 | all |
| `--precision <fp32\|fp16\|int8>` | 精度 | fp32 |
| `--threads <n>` | 线程数 | 1 |
| `--warmup <n>` | 预热次数 | 10 |
| `--runs <n>` | 测试次数 | 100 |
| `--gpu` | 使用 GPU | 否 |

## 分步流程

### Step 1: 编译

```bash
./scripts/build_android.sh
```

### Step 2: 推送并运行

```bash
./scripts/run_benchmark_android.sh [options] [benchmark_args]
```

脚本自动完成：
1. 调用 `setup_test_environment.sh` 锁频（自动检测 root）
2. 推送 `benchmark_inference` 到 `/data/local/tmp/benchmark/`
3. 推送 `libonnxruntime.so` 和 `libMNN.so` 动态库
4. 推送 `models/classification` 模型文件
5. 执行测试（`LD_LIBRARY_PATH=/data/local/tmp/benchmark`）
6. 调用 `restore_test_environment.sh` 恢复环境

run_benchmark_android.sh 选项：
| 选项 | 说明 |
|------|------|
| `--build-type <release\|debug>` | 构建类型 |
| `--no-hardware-control` | 跳过锁频和恢复 |
| `--hardware-check` | 仅检查设备状态 |
| `--generate-report` | 测试后生成报告 |
| `--results-dir <path>` | 结果目录 |

### Step 3: 生成报告

```bash
./scripts/run_benchmark_android.sh --generate-report --results-dir results/sm8250
```

等价于手动：
```bash
python3 scripts/generate_report.py results/sm8250
```

## 命令行参数 (benchmark_inference)

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--model` | 模型名 | mobilenetv2, resnet50, yolov8n, bert |
| `--backend` | 后端类型 | mnn, onnxrt, ort |
| `--precision` | 推理精度 | fp32, fp16, int8 |
| `--threads` | 线程数 | 1, 2, 4 |
| `--runs` | 运行次数 | 100 |
| `--warmup` | 预热次数 | 10 |
| `--gpu` | 使用 GPU（可选） | 无参数 |
| `--profiling <file>` | 启用逐算子 profiling | profiling.json |

## 输出示例

```
--- Performance Results ---
  Init time:  7.23 ms
  P50:    8.70 ms
  P90:    9.12 ms
  P99:    9.45 ms
  Throughput: 114.94 FPS
  Peak mem:   24576 KB
```

精度对比信息（MNN vs ORT 标杆）在日志开头部分。

## 常见问题

- **结果波动大**: 检查是否已执行 `setup_test_environment.sh`（run_benchmark_android.sh 默认自动执行）
- **模型找不到**: `adb shell ls /data/local/tmp/benchmark/models/`
- **libonnxruntime.so 找不到**: 脚本自动推送至同目录，`LD_LIBRARY_PATH` 已设置
- **build_and_run.sh 使用 ANDROID_NDK**: 确保该环境变量已设置