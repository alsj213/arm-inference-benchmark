# MobileNetV2 性能分析卡片

> **来源:** Phase 1 CPU Profiling + Phase 2 算子级 Benchmark + Phase 3 GPU 对比

---

## 模型概览

| 属性 | 值 |
|------|-----|
| 类型 | 轻量图像分类 |
| 参数量 | 3.5M |
| 输入尺寸 | 1×3×224×224 |
| 核心结构 | Inverted Residual Block (Conv1x1 升维 + DWConv 3x3 + Conv1x1 降维) |
| 计算模式 | **Conv1x1 GEMM 密集 + DWConv 访存密集** |

## 端到端性能

| 后端 | Mean | P50 | Std | FPS | 初始化 | 精度 |
|------|------|-----|-----|-----|--------|------|
| MNN CPU | 18.49 ms | 18.49 ms | 0.08 ms | 54.1 | 26.45 ms | PASSED |
| ORT CPU | 29.17 ms | 29.14 ms | 0.27 ms | 34.3 | 29.31 ms | PASSED |
| MNN GPU | ~15.9 ms | - | - | ~62.9 | >700 ms | PASSED |

**MNN/ORT: 1.58x** | **CPU/GPU: 1.17x**

## 热点算子 Top 3

| 排名 | 算子 | 耗时占比 | Bound 类型 | GFLOPS | GPU 加速 | 分析 |
|------|------|---------|----------|--------|---------|------|
| 1 | Conv1x1 GEMM | ~55% | Compute | ~40 | 1.2-1.5x | MNN 的 NCHW4c 布局 + K=4 打包极致优化 |
| 2 | DWConv 3x3 | ~25% | **Memory** | ~8 | ~1x | 访存密集，计算/访存比极低，带宽瓶颈 |
| 3 | Add (逐元素) | ~8% | Memory | - | <1x | GPU kernel launch 开销 > 计算量 |

## MNN vs ORT 差距分析

- **领先来源:** Conv1x1 GEMM -- MNN NCHW4c 内存布局 + K=4 打包达到 ~40 GFLOPS
- **限制因素:** DWConv 占比 25% 但优化空间有限（访存瓶颈，非计算瓶颈）

## simpleperf 火焰图热点

| 函数 | CPU 开销 | 说明 |
|------|---------|------|
| LoopL2 | 64.00% | DWConv 内层循环 |
| L16LoopW | 5.19% | GEMM 16 元素展开 |
| __memcpy | 4.97% | 内存拷贝 (Pack/Unpack) |
| LoopE12L4 | 2.99% | GEMM 外循环变体 |

## 异构部署建议

```
✅ 放 GPU: Conv1x1 (计算密集, 加速 1.2-1.5x)
❌ 留 CPU: DWConv (访存密集, GPU 无收益)
❌ 留 CPU: Add/ReLU/逐元素 (GPU 启动开销 > 计算)
```

## 优化潜力

- **DWConv:** 最大热点 (25%) 但 Memory-Bound，优化空间有限
- **Conv1x1:** 已接近峰值 (~40 GFLOPS)，优化空间小
- **异构部署:** GPU+CPU 配合可额外提升 10-15%
- **量化:** FP16/INT8 对 DWConv (Memory-Bound) 有收益 (减少访存)

---

> **结论:** MobileNetV2 是端侧轻量模型的典范，MNN 对其 Conv1x1 GEMM 的优化已近乎极致。最大瓶颈 DWConv 受限于内存带宽，异构部署 (Conv1x1 放 GPU, DWConv 留 CPU) 是最有希望的优化方向。
