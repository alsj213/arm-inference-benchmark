
# Phase 2 Roofline 分析结果

## 硬件参数

- **峰值算力**: 25 GFLOPS (FP32, MNN NEON 优化实测)
- **内存带宽**: 34 GB/s (LPDDR5)
- **Roofline 斜面**: 0.74 FLOPS/Byte
- **芯片**: 骁龙 865 (SM8250), 4×A77 @ 2.42GHz

## 算子 Roofline 分析

| 算子 | 算术强度 | GFLOPS | Bound 类型 | 利用率(%) |
|------|---------|--------|-----------|----------|
| Conv1x1_K1024_C256_M784 | 444.0 | 39.81 | 🔴 Compute-Bound | 159.2% |
| Conv1x1_M49_C256_K512 | 44.7 | 39.77 | 🔴 Compute-Bound | 159.1% |
| Conv1x1_M3136_C64_K128 | 123.0 | 33.98 | 🔴 Compute-Bound | 135.9% |
| Conv1x1_M784_C32_K64 | 59.2 | 28.93 | 🔴 Compute-Bound | 115.7% |
| Conv1x1_Misaligned_C33_K64 | 62.7 | 28.73 | 🔴 Compute-Bound | 114.9% |
| Conv1x1_M49_C32_K64 | 27.8 | 28.67 | 🔴 Compute-Bound | 114.7% |
| Conv1x1_Misaligned_C31_K64 | 62.7 | 28.41 | 🔴 Compute-Bound | 113.6% |
| Conv1x1_K16_C64_M784 | 15.7 | 21.41 | 🔴 Compute-Bound | 85.6% |
| MatMul_512x512x512 | 1.0 | 16.91 | 🔴 Compute-Bound | 67.7% |
| MatMul_768x768x768 | 1.0 | 15.12 | 🔴 Compute-Bound | 60.5% |
| MatMul_3072x768 | 1.0 | 11.98 | 🔴 Compute-Bound | 47.9% |
| MatMul_768x3072 | 1.0 | 11.92 | 🔴 Compute-Bound | 47.7% |
| DWConv_C16_3x3 | 9.0 | 11.32 | 🔴 Compute-Bound | 45.3% |
| DWConv_C960_3x3 | 7.6 | 7.99 | 🔴 Compute-Bound | 32.0% |

## 分类汇总

| Bound 类型 | 算子数 | 平均利用率 | 最高利用率 |
|-----------|--------|----------|----------|
| Compute-Bound | 14 | 92.8% | 159.2% |

## 多线程加速比

| 算子 | 线程数 | 延迟 (ms) | GFLOPS | Bound |
|------|--------|----------|--------|-------|
| Conv1x1_K1024_C256_M784 | 1 | 10.3250 | 39.81 | Compute-Bound |
| Conv1x1_K16_C64_M784 | 1 | 0.0750 | 21.41 | Compute-Bound |
| Conv1x1_M3136_C64_K128 | 1 | 1.5120 | 33.98 | Compute-Bound |
| Conv1x1_M49_C256_K512 | 1 | 0.3230 | 39.77 | Compute-Bound |
| Conv1x1_M49_C32_K64 | 1 | 0.0070 | 28.67 | Compute-Bound |
| Conv1x1_M784_C32_K64 | 1 | 0.1110 | 28.93 | Compute-Bound |
| Conv1x1_Misaligned_C31_K64 | 1 | 0.4380 | 28.41 | Compute-Bound |
| Conv1x1_Misaligned_C33_K64 | 1 | 0.4610 | 28.73 | Compute-Bound |
| DWConv_C16_3x3 | 1 | 0.3190 | 11.32 | Compute-Bound |
| DWConv_C960_3x3 | 1 | 0.1060 | 7.99 | Compute-Bound |
| MatMul_3072x768 | 1 | 0.3940 | 11.98 | Compute-Bound |
| MatMul_512x512x512 | 1 | 0.0310 | 16.91 | Compute-Bound |
| MatMul_768x3072 | 1 | 0.3960 | 11.92 | Compute-Bound |
| MatMul_768x768x768 | 1 | 0.0780 | 15.12 | Compute-Bound |

