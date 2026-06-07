# ResNet50 性能分析卡片

> **来源:** Phase 1 CPU Profiling + Phase 2 算子级 Benchmark + Phase 3 GPU 对比

---

## 模型概览

| 属性 | 值 |
|------|-----|
| 类型 | 大卷积图像分类 |
| 参数量 | 25.6M |
| 输入尺寸 | 1×3×224×224 |
| 核心结构 | Bottleneck Block (Conv1x1 + Conv3x3 + Conv1x1 + Skip Connection) |
| 计算模式 | **Conv3x3 Winograd 密集 + Conv1x1 GEMM** |

## 端到端性能

| 后端 | Mean | P50 | Std | FPS | 初始化 | 精度 |
|------|------|-----|-----|-----|--------|------|
| MNN CPU | 143.88 ms | 143.87 ms | 0.38 ms | 7.0 | 693.27 ms | PASSED |
| ORT CPU | 221.56 ms | 221.58 ms | 0.35 ms | 4.5 | 388.03 ms | PASSED |
| MNN GPU | ~59.4 ms | - | - | ~16.8 | >700 ms | PASSED |

**MNN/ORT: 1.54x** | **CPU/GPU: 3.03x** -- GPU 大赢家!

## 热点算子 Top 3

| 排名 | 算子 | 耗时占比 | Bound 类型 | GPU 加速 | 分析 |
|------|------|---------|----------|---------|------|
| 1 | Conv3x3 (Winograd) | ~35% | Compute | **3-4x** | 规则密集型计算，GPU 并行度极高 |
| 2 | Conv1x1 GEMM | ~30% | Compute | 1.5-2x | 与 MobileNetV2 同类高效 GEMM |
| 3 | BN + ReLU + Add | ~15% | Memory | <1x | 逐元素操作，GPU 启动开销 > 计算 |

## ResNet50 vs MobileNetV2 计算特征对比

| 特征 | ResNet50 | MobileNetV2 |
|------|----------|-------------|
| 核心算子 | Conv3x3 + Conv1x1 | Conv1x1 + DWConv |
| 计算密度 | 高 (Conv3x3 大量乘加) | 中等 (DWConv 拉低密度) |
| GPU 加速潜力 | **极高** (3.03x) | 中等 (1.17x) |
| 瓶颈 | Compute-Bound | Mixed (Compute + Memory) |
| 最优后端 | **GPU** | CPU (异构可小幅提升) |

## 为什么 ResNet50 GPU 加速 3.03x

1. **Conv3x3 计算密集:** Winograd 变换后大量规则矩阵乘，GPU 并行度极高
2. **Channel 数大:** 256/512/1024/2048 channels，GPU 有足够的工作量分摊启动开销
3. **Adreno 650 对 Conv3x3 优化好:** 高通 GPU 驱动的标准卷积路径成熟
4. **Conv1x1 也有收益:** 虽然不如 Conv3x3，但大 C 的 Conv1x1 在 GPU 上也加速

## 异构部署建议

```
✅ 放 GPU: Conv3x3 (3-4x 加速, 计算密集)
✅ 放 GPU: Conv1x1 大通道 (256+ channels)
⚠️ 条件: Conv1x1 小通道 (64-), 视 GPU 启动开销
❌ 留 CPU: BN/ReLU/Add/Pooling (逐元素/Memory)
❌ 留 CPU: 第一层 Conv (3x224x224, 张量小, GPU 得利少)
```

## 关键数字

- Conv3x3 Winograd GFLOPS: ~35 (单核)
- GPU 加速比: **3.03x** (4 模型中最高的 GPU 加速比)
- 从 Phase 1 PMU 数据: simpleperf 显示 MNN 内部 GEMM/Conv 循环占主导

---

> **结论:** ResNet50 是本项目中 GPU 加速效果最好的模型。其规则密集的 Conv3x3 计算模式与 GPU 架构天然匹配。推荐将此模型作为 GPU 异构部署的标杆案例。
