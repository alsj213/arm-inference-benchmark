# Phase 1: CPU 全量 Profiling

## 测试环境
- 设备: 红米 K30S / M2007J3SC (骁龙 865 SM8250)
- CPU: 4x Cortex-A77 (0x805) + 4x Cortex-A55 (0xd0d) = 8 核
- 测试日期: 2026-06-06
- 分支: feat/benchmark-full-plan
- Commit: e2f9c54 feat: Phase 0 环境就绪
- Warmup: 50 iterations, Test: 100 iterations
- 频率: schedutil (未锁频，设备未 root)
- Precision: FP32
- Threads: 1 (单线程)

## MNN vs ORT 全模型对比

| 模型 | MNN Mean(ms) | ORT Mean(ms) | MNN/ORT 加速比 | 胜出 |
|------|-------------|-------------|---------------|------|
| MobileNetV2 | 18.49 | 29.17 | **1.58x** | MNN |
| ResNet50 | 143.88 | 221.56 | **1.54x** | MNN |
| YOLOv8n | 174.00 | 304.19 | **1.75x** | MNN |
| BERT | 683.45 | 598.80 | **0.88x** | ORT |

### 详细指标

| 模型 | 后端 | Mean(ms) | P50(ms) | P99(ms) | Std(ms) | FPS | Init(ms) | 精度 |
|------|------|----------|---------|---------|---------|-----|----------|------|
| mobilenetv2 | MNN | 18.49 | 18.49 | 18.70 | 0.08 | 54.1 | 26.45 | PASSED |
| mobilenetv2 | ORT | 29.17 | 29.14 | 31.25 | 0.27 | 34.3 | 29.31 | PASSED |
| resnet50 | MNN | 143.88 | 143.87 | 146.02 | 0.38 | 7.0 | 693.27 | PASSED |
| resnet50 | ORT | 221.56 | 221.58 | 222.20 | 0.35 | 4.5 | 388.03 | PASSED |
| bert | MNN | 683.45 | 683.44 | 689.30 | 0.99 | 1.5 | 1279.11 | PASSED |
| bert | ORT | 598.80 | 598.85 | 600.14 | 0.54 | 1.7 | 641.74 | PASSED |
| yolov8n | MNN | 174.00 | 174.02 | 174.74 | 0.32 | 5.8 | 155.46 | PASSED |
| yolov8n | ORT | 304.19 | 304.09 | 306.34 | 0.72 | 3.3 | 27.07 | PASSED |

## simpleperf PMU 分析 (MobileNetV2 MNN)

### PMU 计数器
| 事件 | 计数 | 速率 |
|------|------|------|
| cpu-cycles | 9,986,404,311 | 2.73 GHz |
| instructions | 21,474,232,393 | 5.85 G/sec |
| cache-misses | 97,976,042 | 26.68 M/sec |
| branch-misses | 2,773,935 | 757.6 K/sec |

> 注意: 4 个硬件事件数超过可用 PMU 计数器，存在多路复用 (multiplexing)，事件计数可能偏低。建议 root 后使用 `--use-devfreq-counters`。

### 热点函数 (Top 10 by CPU cycles overhead)
| Overhead | Symbol | Shared Object |
|----------|--------|---------------|
| 64.00% | LoopL2 | libMNN.so |
| 5.19% | L16LoopW | libMNN.so |
| 4.97% | __memcpy | libc.so |
| 2.99% | LoopE12L4 | libMNN.so |
| 2.54% | E1LoopL | libMNN.so |
| 2.34% | StoreLH8 | libMNN.so |
| 2.20% | L8LoopW | libMNN.so |
| 1.96% | PostTreatLH8 | libMNN.so |
| 1.71% | L1LoopW | libMNN.so |
| 1.05% | L4LoopW | libMNN.so |

**关键发现**: MobileNetV2 推理 64% 的时间消耗在 MNN 的 `LoopL2` 函数（depthwise convolution 内层循环），kptr_restrict 限制了内核符号解析。

### 火焰图采样
- 采样数: 19,871 (0 丢失)
- 采样事件: cpu-cycles
- 调用图: fp (frame pointer)
- 数据文件: `perf_mbv2.data` (2.0 MB)
- 完整报告: `mobilenetv2_simpleperf_report.txt` (523 行)

## 缺失模型
| 模型 | 状态 | 原因 |
|------|------|------|
| MobileViT-S | TODO 跳过 | 模型目录不存在，需单独下载 |
| Qwen2-0.5B | TODO 跳过 | 模型目录不存在，LLM 模型较大 |

## 关键结论

1. **MNN 在 CNN 模型上显著优于 ORT**: MobileNetV2 (1.58x)、ResNet50 (1.54x)、YOLOv8n (1.75x)
2. **ORT 在 BERT (Transformer) 上反超 MNN**: 0.88x 加速比，说明 MNN 的 Transformer 算子优化有提升空间
3. **所有模型精度验证通过**: 余弦相似度 ~1.0，均值绝对误差 < 0.00001
4. **StdDev 极低**: 所有模型 Std < 1ms（MobileNetV2 仅 0.08ms），性能非常稳定
5. **MNN 热点函数**: depthwise conv 内层循环 `LoopL2` 占 64% CPU 时间，是 Phase 2 算子级优化的重点

## 产物清单
```
results/phase1_cpu_profiling/
├── README.md                              # 本文件
├── all_models_summary.md                  # 摘要对比表
├── mobilenetv2_mnn_cpu_profile.log        # MobileNetV2 MNN 原始日志
├── resnet50_mnn_cpu_profile.log           # ResNet50 MNN 原始日志
├── bert_mnn_cpu_profile.log               # BERT MNN 原始日志
├── yolov8n_mnn_cpu_profile.log            # YOLOv8n MNN 原始日志
├── mobilenetv2_ort_cpu_profile.log        # MobileNetV2 ORT 原始日志
├── resnet50_ort_cpu_profile.log           # ResNet50 ORT 原始日志
├── bert_ort_cpu_profile.log               # BERT ORT 原始日志
├── yolov8n_ort_cpu_profile.log            # YOLOv8n ORT 原始日志
├── mobilenetv2_simpleperf_stat.log        # PMU 计数器原始日志
├── perf_mbv2.data                         # 火焰图采样数据 (2.0 MB)
└── mobilenetv2_simpleperf_report.txt      # 火焰图完整报告 (523 行)
```

## 状态: DONE_WITH_CONCERNS

- 4/4 模型 MNN Profling 完成
- 4/4 模型 ORT 基线完成
- simpleperf PMU + 火焰图完成
- MobileViT-S 和 Qwen2-0.5B 因缺失跳过（TODO）
- 设备未 root，simpleperf kernel 符号受限，PMU 计数器有复用偏差
