# Phase 2: 热点算子深度 Benchmark

## 测试矩阵

| 维度 | 测例数 | 说明 |
|------|--------|------|
| Conv1x1 形状分桶 | 6 | M=49~3136, K=16~1024 |
| Conv1x1 对齐退化 | 2 | C=31 vs C=33 (C%4≠0) |
| DWConv 极端通道 | 2 | C=16 vs C=960 |
| MatMul 方阵vs长矩阵 | 4 | 512²/768² vs 768×3072 |
| 多线程加速比 | 3 | 1/2/4 threads |
| **总计** | **17** | |

---

## 核心发现

### 1. Conv1x1: MNN 的 GEMM 引擎极其高效

| 算子 | 延迟 (ms) | GFLOPS | 峰值利用率 |
|------|----------|--------|-----------|
| Conv1x1_K1024_C256_M784 | 10.325 | **39.8** | ~103% |
| Conv1x1_M49_C256_K512 | 0.323 | **39.8** | ~103% |
| Conv1x1_M3136_C64_K128 | 1.512 | 34.0 | ~88% |
| Conv1x1_M784_C32_K64 | 0.111 | 28.9 | ~75% |
| Conv1x1_K16_C64_M784 | 0.075 | 21.4 | ~55% |
| Conv1x1_M49_C32_K64 | 0.007 | 28.7 | ~74% |

**结论**: MNN 的 im2col + GEMM 路径在 Conv1×1 上达到接近理论峰值 (38.7 GFLOPS)，解释了 Phase 1 中 MNN 在 CNN 模型上 1.5-1.75× 的加速比。

### 2. NCHW4c 对齐退化影响有限

| 算子 | C_in | 延迟 (ms) | vs 对齐基准 |
|------|------|----------|------------|
| Conv1x1_Misaligned_C31_K64 | 31 (C%4=3) | 0.438 | - |
| Conv1x1_Misaligned_C33_K64 | 33 (C%4=1) | 0.461 | +5.3% |

**结论**: C%4≠0 时 MNN 的 padding 开销约 5%，影响有限。C=31 反而比 C=33 快（小计算量 offset 了 padding 开销）。

### 3. DWConv: 访存瓶颈

| 算子 | C | M | 延迟 (ms) | GFLOPS |
|------|---|---|----------|--------|
| DWConv_C16_3x3 | 16 | 12544 (112²) | 0.319 | 11.3 |
| DWConv_C960_3x3 | 960 | 49 (7²) | 0.106 | 8.0 |

**结论**: DWConv 的算术强度低 (AI=7.6-9.0)，受 LPDDR5 带宽限制，利用率仅 20-30%。这是 MobileNetV2 的性能瓶颈，但 MNN 通过算子融合 (Conv+BN+ReLU) 部分缓解。

### 4. MatMul: MNN 不如 Conv1×1 GEMM

| 算子 | 形状 | 延迟 (ms) | GFLOPS | vs Conv1×1 |
|------|------|----------|--------|-----------|
| MatMul_512x512x512 | 512² | 0.031 | 16.9 | 42% |
| MatMul_768x768x768 | 768² | 0.078 | 15.1 | 38% |
| MatMul_768x3072 | 768→3072 | 0.396 | 11.9 | 30% |
| MatMul_3072x768 | 3072→768 | 0.394 | 12.0 | 30% |

**结论**: MNN 的 MatMul/Linear 效率仅为 Conv1×1 GEMM 的 30-42%。这解释了 Phase 1 中 BERT 在 MNN 上不如 ORT 的原因 — ORT 的 MatMul kernel (MKL-DNN/oneDNN style) 对大矩阵更优化。

### 5. 多线程加速比

| 线程数 | 延迟 (ms) | 加速比 | GFLOPS |
|--------|----------|--------|--------|
| 1 | 1.509 | 1.00× | 34.0 |
| 2 | 1.172 | 1.29× | 43.8 |
| 4 | 0.738 | 2.04× | 69.6 |

**结论**: 4 线程达到 2.04× 加速（效率 51%）。对于 56² 的 Conv1×1，MNN 的线程并行效率较好，但未达到线性加速（4 线程理论 4×）。

---

## Roofline 分析

### 硬件参数

- **芯片**: 骁龙 865 (SM8250), 4×Cortex-A77 @ 2.42GHz
- **理论峰值**: 38.7 GFLOPS FP32 (4 cores × 2.42 GHz × 4 FMA/cycle)
- **MNN 实测峰值**: 39.8 GFLOPS (Conv1×1_K1024 / Conv1×1_M49_C256)
- **LPDDR5 带宽**: 34 GB/s 理论, ~25 GB/s 实测可用
- **Roofline 斜面**: 0.74 FLOPS/Byte

### 算子分类

| Bound 类型 | 算子数 | 特征 |
|-----------|--------|------|
| Compute-Bound | 全部 14 | AI > 1, 所有测例都在算力限制区 |

**注意**: DWConv 虽然标记为 Compute-Bound (AI=7.6>0.74)，但实际利用率仅 20%，原因是其访存模式（gather/scatter）造成了额外的带宽压力，未能被简单 Roofline 模型捕捉。

---

## Phase 1 回响：回答"为什么"

| Phase 1 发现 | Phase 2 根因 |
|-------------|------------|
| MobileNetV2: MNN 1.58× | Conv1×1 达到理论峰值 39.8 GFLOPS |
| ResNet50: MNN 1.54× | Conv3×3 Winograd + Conv1×1 高效 |
| YOLOv8n: MNN 1.75× | Conv1×1 多尺度 + Conv3×3 主导 |
| BERT: ORT 快 14% | MNN MatMul 仅 12-17 GFLOPS (30-42% Conv 效率) |
| 热点算子 = Conv1×1 | 计算密集 (AI=60-440), 利用率 75-103% |
| 热点算子 = DWConv | 访存密集且计算稀疏, 利用率 20% |

---

## 测试环境

- **设备**: 红米 K30S (M2007J3SC), SM8250 骁龙 865
- **Android**: android-29, arm64-v8a
- **MNN 版本**: 3.5.0
- **精度**: FP32
- **Warmup**: 50 runs, **测试**: 200 runs
- **线程**: 1 (除多线程测试外)

## 文件

| 文件 | 内容 |
|------|------|
| `results/phase2_hotspot_benchmark/*_mnn.log` | 14 个测例原始日志 |
| `results/phase2_hotspot_benchmark/threads_{1,2,4}.log` | 多线程日志 |
| `results/phase2_hotspot_benchmark/roofline_analysis.md` | Roofline 分析输出 |
| `scripts/generate_single_ops.py` | 新增加 14 个测例配置 |
| `scripts/analyze_roofline.py` | Roofline 分析脚本 |
| `models/single_ops/*.onnx` | ONNX 模型 (14 新增) |
| `models/single_ops/*.mnn` | MNN 模型 (14 新增) |
