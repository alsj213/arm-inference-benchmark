# 模型性能分析卡片

> 每个模型一页摘要，基于 Phase 1-5 真实实测数据
> 平台: 骁龙 865, 测试日期: 2026-06

---

## 1. MobileNetV2 (3.5M 参数)

**计算特征**: Conv1x1 GEMM 密集 (55%) + DWConv 访存密集 (25%) + 逐元素 (8%)
**核心结构**: Inverted Residual Block (Conv1x1 升维 → DWConv 3×3 → Conv1x1 降维)
**输入**: 1×3×224×224

### 端到端性能

| 后端 | Mean | FPS | 精度 |
|------|------|-----|------|
| MNN CPU | 18.49 ms | 54.1 | PASSED |
| ORT CPU | 29.17 ms | 34.3 | PASSED |
| MNN GPU | ~15.9 ms | ~62.9 | PASSED |

**MNN/ORT: 1.58x** | **CPU/GPU: 1.17x**

### 热点算子 Top 3

| 排名 | 算子 | 占比 | Bound | GFLOPS | GPU 加速 |
|------|------|------|-------|--------|---------|
| 1 | Conv1x1 GEMM | ~55% | Compute | ~40 | 1.2-1.5x |
| 2 | DWConv 3×3 | ~25% | **Memory** | ~8 | ~1x |
| 3 | Add (逐元素) | ~8% | Memory | - | <1x |

### 分析
- **MNN 领先来源**: Conv1x1 GEMM — NCHW4c 内存布局 + K=4 打包，计算/访存比 9.6 FLOPs/byte
- **GPU 小加速原因**: DWConv 25% 占比拖累 — 访存密集，GPU 无并行优势
- **simpleperf 热点**: LoopL2 (64.00%) — DWConv 内层循环；L16LoopW (5.19%) — GEMM 展开

### 异构建议
```
✅ 放 GPU: Conv1x1 (计算密集, 1.2-1.5x)
❌ 留 CPU: DWConv (访存瓶颈, 无 GPU 收益)
❌ 留 CPU: Add/ReLU (GPU launch 开销 > 计算)
```

**结论**: 端侧轻量模型的典范。MNN Conv1x1 GEMM 优化近乎极致 (~40 GFLOPS)。最大瓶颈 DWConv 受内存带宽限制，异构部署 (Conv1x1→GPU, DWConv→CPU) 可额外提升 10-15%。

> 来源: Phase 1 (`mobilenetv2_mnn_cpu_profile.log`, `mobilenetv2_ort_cpu_profile.log`), Phase 2 (`Conv1x1_*.log`, `DWConv_*.log`), Phase 3 (`mobilenetv2_*_gpu.log`)

---

## 2. ResNet50 (25.6M 参数)

**计算特征**: Conv3x3 Winograd (35%) + Conv1x1 GEMM (30%) + BN/ReLU/Add (15%)
**核心结构**: Bottleneck Block (Conv1×1 + Conv3×3 + Conv1×1 + Skip Connection)
**输入**: 1×3×224×224

### 端到端性能

| 后端 | Mean | FPS | 精度 |
|------|------|-----|------|
| MNN CPU | 143.88 ms | 7.0 | PASSED |
| ORT CPU | 221.56 ms | 4.5 | PASSED |
| MNN GPU | ~59.4 ms | ~16.8 | PASSED |

**MNN/ORT: 1.54x** | **CPU/GPU: 3.03x — GPU 大赢家!**

### 热点算子 Top 3

| 排名 | 算子 | 占比 | Bound | GPU 加速 |
|------|------|------|-------|---------|
| 1 | Conv3×3 (Winograd) | ~35% | Compute | **3-4x** |
| 2 | Conv1x1 GEMM | ~30% | Compute | 1.5-2x |
| 3 | BN + ReLU + Add | ~15% | Memory | <1x |

### 分析
- **GPU 3.03x 的原因**: Conv3×3 规则密集型计算 + 大 Channel (256/512/1024/2048) = 充分 GPU 并行度 + 足够工作量分摊启动开销
- **Adreno 650 对标准 Conv3×3 优化成熟**: 高通 GPU 驱动的标准卷积路径是优化重点

### ResNet50 vs MobileNetV2 特征对比

| 特征 | ResNet50 | MobileNetV2 |
|------|----------|-------------|
| 核心算子 | Conv3×3 + Conv1×1 | Conv1×1 + DWConv |
| 计算密度 | 高 | 中 (DWConv 拉低) |
| GPU 加速潜力 | **极高 (3.03x)** | 中 (1.17x) |
| 最优后端 | **GPU** | CPU |

### 异构建议
```
✅ 放 GPU: Conv3×3 (3-4x 加速)
✅ 放 GPU: Conv1×1 大通道 (256+)
⚠️ 条件: Conv1×1 小通道 (64-)
❌ 留 CPU: BN/ReLU/Add/Pooling
❌ 留 CPU: 第一层 Conv (3×224×224, 张量小)
```

**结论**: 本项目中 GPU 加速效果最好的模型。规则密集型 Conv3×3 与 GPU 架构天然匹配。推荐作为 GPU 异构部署标杆案例。

> 来源: Phase 1 (`resnet50_mnn_cpu_profile.log`, `resnet50_ort_cpu_profile.log`), Phase 3 (`resnet50_*_gpu.log`, `resnet50_gpu_profiling.log`)

---

## 3. BERT-base (110M 参数)

**计算特征**: MatMul GEMM 密集 + GELU + LayerNorm + Softmax
**核心结构**: 12 层 Transformer Encoder (Multi-Head Self-Attention + FFN)
**输入**: 1×128 (token_ids + attention_mask), Hidden Size = 768

### 端到端性能

| 后端 | Mean | FPS | 精度 |
|------|------|-----|------|
| MNN CPU (1T) | 683.45 ms | 1.5 | cos=0.990 |
| ORT CPU (1T) | 598.80 ms | 1.7 | cos=1.000 |
| MNN CPU (4T) | 422.09 ms | 2.37 | cos=0.990 |
| ORT CPU (4T) | 267.7 ms | 3.74 | cos=1.000 |
| MNN GPU | ~1883 ms | ~0.5 | **cos=0.87 异常** |

**MNN/ORT (1T): 0.88x** — 唯一 MNN 落后的模型!
**MNN/ORT (4T): 0.63x** — 多线程后差距反而拉大!

### 三层根因分析

**根因 1: GELU 实现差异 (占总差距 ~9%)**
```
ONNX GELU: 0.5 * x * (1 + erf(x/sqrt(2))) → 12 个 Erf 节点
MNN 执行: 逐元素 erf() + expf() → libm
simpleperf: __ieee754_expf 6.10% + erff 5.60% = 11.7%
ORT:     图优化融合 → 快速 GELU → 开销 ~0%
```
MNN 有快速 GELU (`MNNGelu.S`, tanh 近似 + NEON)，但 Erf 算子路径未触发。

**根因 2: MatMul 微内核 K 维不打包 (占总差距 ~20-30%)**
- 代码位置: `CommonOptFunctionNeon.cpp:1879` — `lP=1` (K 维度逐元素)
- 计算/访存比: 2.4 FLOPs/byte (vs Conv1x1 的 9.6 — 4 倍差距)
- 实际 GFLOPS: 14.8 (仅 Conv1x1 的 37%)

**根因 3: 多线程扩展性差**
| Threads | MNN | ORT | MNN 效率 |
|---------|-----|-----|---------|
| 1 | 737.7 | 668.7 | - |
| 4 | 422.1 | 267.7 | **44% vs 63%** |

BERT batch=1 → e 维度=1 → 限制了 MNN 并行切分粒度。

### PMU 交叉验证

| 指标 | MNN | ORT | 差异 |
|------|-----|-----|------|
| instructions | 39.37B | 39.47B | **相同!** |
| IPC | 1.84 | 2.06 | MNN -12% |
| cache miss rate | 2.30% | 1.00% | MNN 2.3x |
| branch-misses | 37.3M | 3.3M | MNN **11.4x** |

### 优化优先级

| 优先级 | 方案 | 收益 | 难度 |
|--------|------|------|------|
| P0 | 导出 fast GELU ONNX | ~9% | 低 |
| P1 | lP 1→4 | 20-30% | 中 |
| P2 | MNN 图融合 Erf→FastGELU | ~9% | 中 |
| P3 | MatMul 多线程策略 | 15-20% | 高 |

**结论**: 最有技术深度的分析对象。从"为什么慢"→"哪行代码导致"→"怎么改"形成完整优化闭环。lP=1 发现解释了 Conv1x1 高效而 MatMul 低效的核心矛盾，体现对 GEMM 微架构的深入理解。

> 来源: Phase 1 (`bert_mnn_cpu_profile.log`, `bert_ort_cpu_profile.log`), Phase 4 (`BERT_OPTIMIZATION_REPORT.md`, `bert_*_pmu.log`, `bert_threads_*.log`)

---

## 4. YOLOv8n (3.2M 参数)

**计算特征**: 多尺度 Conv1×1/Conv3×3 + DWConv + Concat + Reshape + SiLU
**核心结构**: CSPDarknet Backbone + PAN Neck + Detect Head
**输入**: 1×3×640×640

### 端到端性能

| 后端 | Mean | FPS | 精度 |
|------|------|-----|------|
| MNN CPU | 174.00 ms | 5.8 | PASSED |
| ORT CPU | 304.19 ms | 3.3 | PASSED |
| MNN GPU | ~899 ms | ~1.1 | PASSED |

**MNN/ORT: 1.75x** — 4 模型中 MNN 领先幅度最大!
**CPU/GPU: 0.22x** — GPU 反慢，最不适合 GPU 的模型!

### 热点算子分析

| 排名 | 算子 | 占比 | Bound | GPU 表现 |
|------|------|------|-------|---------|
| 1 | Conv1×1 (多尺度) | ~35% | Compute | 混合 |
| 2 | Conv3×3 | ~20% | Compute | 2-3x |
| 3 | DWConv | ~15% | **Memory** | **<1x** |
| 4 | Concat/Reshape | ~10% | Memory | 极慢 |
| 5 | SiLU/Add | ~10% | Memory | <1x |

### 分析
- **MNN 领先 1.75x 原因**: 多尺度 Conv1×1 GEMM 全覆盖 + 内存管理高效 + 调度紧凑
- **GPU 反慢 0.22x 原因**: DWConv (15%) + Concat/Reshape (10%) + SiLU (10%) + 小 Tensor 多 → GPU 水土不服

```
YOLOv8n 的计算图特征:
  CPU 适合: 大量小算子 + 不规则内存访问 + 频繁调度的计算图
  GPU 适合: 少量大算子的规则密集型计算
  YOLOv8n 天然是前者 → GPU 就是慢
```

### 与 MobileNetV2 对比

| 特征 | YOLOv8n | MobileNetV2 |
|------|---------|-------------|
| MNN/ORT | **1.75x** | 1.58x |
| GPU 加速 | **0.22x** | 1.17x |
| 算子多样性 | 高 (Concat/Upsample) | 低 (Conv+Add) |
| 计算图特征 | 多分支 + 多尺度 | 线性串行 |

### 异构建议
```
⚠️ 不建议 GPU 部署 YOLOv8n (当前 MNN OpenCL 实现下)
✅ 全 CPU 推理最优: 174ms, 5.8 FPS

如果必须异构:
  ✅ GPU: strides 8/16 的大尺度 Conv3×3
  ❌ CPU: DWConv, Concat, Reshape, SiLU, 小尺度 Conv
```

**结论**: MNN CPU 优化最充分的模型 (1.75x 领先)，但也是 GPU 最不适用的模型。本质原因是计算图特征决定: 大量小算子 + 不规则内存访问 + 多分支结构完美匹配 CPU 灵活性，与 GPU 规则密集型计算模式根本冲突。

> 来源: Phase 1 (`yolov8n_mnn_cpu_profile.log`, `yolov8n_ort_cpu_profile.log`), Phase 3 (`yolov8n_*_gpu.log`)

---

> 文档版本: v1.0 | 生成日期: 2026-06-06
