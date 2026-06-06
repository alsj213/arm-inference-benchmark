# YOLOv8n 性能分析卡片

> **来源:** Phase 1 CPU Profiling + Phase 3 GPU 对比

---

## 模型概览

| 属性 | 值 |
|------|-----|
| 类型 | 目标检测 (单阶段) |
| 参数量 | 3.2M |
| 输入尺寸 | 1×3×640×640 |
| 核心结构 | CSPDarknet Backbone + PAN Neck + Detect Head |
| 计算模式 | **多尺度 Conv1x1/Conv3x3 + DWConv + Concat + Reshape** |

## 端到端性能

| 后端 | Mean | P50 | Std | FPS | 初始化 | 精度 |
|------|------|-----|-----|-----|--------|------|
| MNN CPU | 174.00 ms | 174.02 ms | 0.32 ms | 5.8 | 155.46 ms | PASSED |
| ORT CPU | 304.19 ms | 304.09 ms | 0.72 ms | 3.3 | 27.07 ms | PASSED |
| MNN GPU | ~899 ms | - | - | ~1.1 | >700 ms | PASSED |

**MNN/ORT: 1.75x** -- 4 模型中 MNN 领先幅度最大!
**CPU/GPU: 0.22x** -- GPU 反慢，是 4 模型中最不适合 GPU 的!

## 热点算子分析

| 排名 | 算子 | 耗时占比 | Bound 类型 | GPU 表现 | 分析 |
|------|------|---------|----------|---------|------|
| 1 | Conv1x1 (多尺度) | ~35% | Compute | 混合 | 不同尺度的 Conv1x1 表现不一 |
| 2 | Conv3x3 | ~20% | Compute | 2-3x | 大尺度 Conv3x3 GPU 有效 |
| 3 | DWConv | ~15% | **Memory** | **<1x** | GPU 拖累主因 |
| 4 | Concat/Reshape | ~10% | Memory | **极慢** | GPU 内存搬运开销大 |
| 5 | 逐元素 (SiLU/Add) | ~10% | Memory | **<1x** | kernel launch 开销 |

## YOLOv8n 特殊挑战

### 为什么 MNN CPU 领先 ORT 最多 (1.75x)?

1. **多尺度特征融合:** YOLOv8n 有大量不同尺度的 Conv1x1，MNN 的 GEMM 优化覆盖面广
2. **结构复杂度:** PAN Neck 的 Concat + Upsample 组合，MNN 内存管理更高效
3. **检测头:** 多尺度 Detect Head 的 Conv2d 序列，MNN 调度更紧凑

### 为什么 GPU 反而最慢 (0.22x)?

| 原因 | 影响 |
|------|------|
| **DWConv 占比高** | 访存密集，GPU 无并行优势 |
| **Concat 多** | GPU 内存搬运 + 重排布开销大 |
| **Reshape 频繁** | GPU 的 kernel launch 延迟累积 |
| **多尺度小 Tensor** | 小 Tensor 放 GPU 得不偿失 |
| **SiLU 逐元素** | GPU 启动开销 > 计算 |

```
YOLOv8n GPU 慢的根因:
  CPU 适合: 大量小算子 + 不规则内存访问 + 频繁调度的计算图
  GPU 适合: 少量大算子的规则密集计算
  YOLOv8n 的计算图特征是前者 → GPU 水土不服!
```

## 异构部署建议

```
⚠️ 不建议 GPU 部署 YOLOv8n (当前 MNN OpenCL 实现下)
✅ 全 CPU 推理是最优方案: 174ms, 5.8 FPS

如果一定要异构:
  ✅ 放 GPU: strides 8/16 的大尺度 Conv3x3
  ❌ 留 CPU: 所有 DWConv、Concat、Reshape、SiLU
  ❌ 留 CPU: 小尺度 (stride 32) 的 Conv
```

## 与 MobileNetV2 的对比

| 特征 | YOLOv8n | MobileNetV2 |
|------|---------|-------------|
| MNN/ORT | **1.75x** | 1.58x |
| GPU 加速 | **0.22x** (反慢) | 1.17x |
| 算子多样性 | 高 (Concat/Reshape/Upsample) | 低 (Conv1x1 + DWConv + Add) |
| GPU 适用性 | 不适合 | 勉强可用 |
| 计算图特征 | 多分支 + 多尺度 | 线性串行 |

---

> **结论:** YOLOv8n 是 MNN CPU 优化最充分的模型 (1.75x 领先)，但也是 GPU 最不适用的模型。这本质上是其计算图特征决定的: 大量小算子 + 不规则内存访问 + 多分支结构，完美匹配 CPU 的灵活性优势，也与 GPU 的规则密集型计算模式根本冲突。
