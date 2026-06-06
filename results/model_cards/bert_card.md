# BERT 性能分析卡片

> **来源:** Phase 1 CPU Profiling + Phase 4 BERT 优化闭环 + Phase 3 GPU 对比

---

## 模型概览

| 属性 | 值 |
|------|-----|
| 类型 | Encoder Transformer (NLP) |
| 参数量 | 110M (BERT-base) |
| 输入尺寸 | 1×128 (token_ids + attention_mask) |
| 层数 | 12 层 Transformer Encoder |
| Hidden Size | 768 |
| 核心结构 | Multi-Head Self-Attention + Feed-Forward Network |
| 计算模式 | **MatMul 密集 + GELU + LayerNorm + Softmax** |

## 端到端性能

| 后端 | Mean | P50 | Std | FPS | 初始化 | 精度 |
|------|------|-----|-----|-----|--------|------|
| MNN CPU (1T) | 683.45 ms | 683.44 ms | 0.99 ms | 1.5 | 1279.11 ms | cos=0.990 |
| ORT CPU (1T) | 598.80 ms | 598.85 ms | 0.54 ms | 1.7 | 641.74 ms | cos=1.000 |
| MNN CPU (4T) | 422.09 ms | 426.12 ms | - | 2.37 | - | cos=0.990 |
| ORT CPU (4T) | 267.7 ms | - | - | 3.74 | - | cos=1.000 |
| MNN GPU | ~1883 ms | - | - | ~0.5 | >3000 ms | **cos=0.87** |

**MNN/ORT (1T): 0.88x** -- BERT 是唯一 MNN 落后的模型!
**MNN/ORT (4T): 0.63x** -- 多线程后差距反而拉大!

## BERT 为什么是唯一反转的模型? -- 三层根因

### 根因 1: GELU 使用 Erf 实现 (~9% 差距)

```
ONNX 模型中的 GELU: 0.5 * x * (1 + erf(x/sqrt(2))) → 12 个 Erf 节点
MNN 执行: UnaryOpOperation_GELU_STANDARD → 逐元素 erf() → libm
simpleperf: __ieee754_expf 6.10% + erff 5.60% = 11.7% 总开销
ORT:     图优化器融合为快速 GELU → 开销 ~0%
```

**MNN 有快速 GELU (`MNNGelu.S`, tanh 近似 + NEON) 但 Erf 路径未触发!**

### 根因 2: MatMul K 维不打包 (~20-30% 差距)

位置: `third_party/MNN/source/backend/cpu/arm/CommonOptFunctionNeon.cpp:1879`

```c
*lP = 1;  // K 维度不打包 -- 根因!
```

| 特性 | Conv1x1 GEMM | MatMul GEMM |
|------|-------------|-------------|
| K 维度打包 | **K=4** (C4 格式) | **K=1** (逐元素) |
| 计算/访存比 | 9.6 FLOPs/byte | **2.4 FLOPs/byte** |
| 实际 GFLOPS | ~40 | **14.8** |
| 效率差距 | - | **仅 37%** |

### 根因 3: 多线程扩展性差

| Threads | MNN 加速比 | ORT 加速比 | MNN 并行效率 |
|---------|-----------|-----------|------------|
| 1 | 1.00x | 1.00x | - |
| 2 | 1.46x | 1.48x | 73% |
| 4 | 1.75x | 2.50x | **44%** |

BERT batch=1 时 e 维度=1，限制了 MNN 的并行切分。

## PMU 对比 -- 为什么 MNN 指令数相同但更慢?

| 指标 | MNN | ORT | 差异 |
|------|-----|-----|------|
| instructions | 39.37B | 39.47B | **几乎相同** (!) |
| cpu-cycles | 21.41B | 19.16B | MNN 多用 11.8% |
| IPC | 1.84 | 2.06 | MNN 低 12.1% |
| cache miss rate | 2.30% | 1.00% | MNN 高 2.3x |
| branch-misses | 37.3M | 3.3M | MNN 高 **11.4x** (!) |

**结论: MNN 和 ORT 做了同样的计算量，但 MNN 微架构效率更低 -- 更多周期浪费在 Cache Miss 和分支预测失败上。**

## GPU 为什么不适用

- MNN OpenCL 的 MatMul 在 Adreno 650 上未专项优化: 1883ms (0.41x)
- Transformer 大量小算子 (LayerNorm, Softmax, GELU, Add): kernel launch 开销累积
- **精度异常:** 余弦相似度仅 0.87，怀疑 OpenCL FP16 overflow

## 优化方案优先级

| 优先级 | 方案 | 预期收益 | 难度 | 说明 |
|--------|------|---------|------|------|
| P0 | 导出 ONNX 用 fast GELU | ~9% | 低 | `approximation='tanh'` |
| P1 | lP 1→4 | **20-30%** | 中 | 需修改微内核 + B 矩阵 Pack |
| P2 | MNN 图融合 (Erf→FastGELU) | ~9% | 中 | Post-Transform pass |
| P3 | MatMul 多线程策略 | 15-20% | 高 | 需调整调度逻辑 |

---

> **结论:** BERT 是最有技术深度的分析对象。从"为什么慢"到"哪行代码导致的"到"怎么改"，形成了完整的优化闭环。lP=1 这个单一参数的发现是项目最大的技术亮点 -- 它解释了为什么 Conv1x1 高效 (~40 GFLOPS) 而 MatMul 低效 (14.8 GFLOPS)，体现了对 MNN GEMM 微架构的深入理解。
