# 面试准备材料

> **项目:** 基于骁龙 865 的 ARM 端侧深度学习推理框架全量性能分析
> **用途:** 面试中讲述项目经历的结构化话术
> **覆盖:** 1 分钟电梯演讲 / 3 分钟项目展开 / 10 个追问应对 / 技术亮点速记 / 简历描述

---

## 1 分钟版（电梯演讲）

"我基于骁龙 865 搭了一套完整的端侧推理性能分析体系。选了 5 个模型覆盖 CV 到 LLM 的五种计算模式，用 MNN Profiler + simpleperf PMU 做全量 Profiling，按二八原则对热点算子做三层深度的 Benchmark。

核心发现三个:
一是 **MNN 在 CNN 上为什么快** — Conv1x1 GEMM 达到接近峰值算力，NCHW4c 布局 + K=4 打包；
二是 **MNN 在 BERT 上为什么反而不如 ORT** — Erf GELU 没做图融合 + 多线程扩展性差，不是简单的'GEMM 慢'的问题；
三是 **GPU 在端侧的真实价值** — 只有 Conv 密集的模型才受益，Transformer 算子大量不支持，异构调度是必须的。

从 Profiling → 瓶颈定位 → 源码行级根因分析 → 优化闭环，完整展示了推理引擎性能工程师的核心能力。"

---

## 3 分钟版（展开讲述）

### 第一分钟: 项目背景和方法论

**为什么做:**
业界缺乏标准化的端侧推理 Benchmark。各框架宣称的性能数据在不同硬件上差异很大，且缺乏系统性的瓶颈分析方法论。我做的不仅仅是"跑分"，而是建立一套 Profiling 驱动的性能分析方法体系 — 能定位到源码行级瓶颈、给出可量化优化方案。

**怎么设计:**
- **Profiling 驱动 + 二八原则**: 先全量 Profiling 定位热点算子，聚焦累计占比 >= 80% 的算子做深度分析
- **三层深度覆盖**: 模型级 (端到端 Latency) → 算子级 (14+ 单算子测例 + Roofline) → 指令级 (PMU 计数器 + Neon 汇编 + 源码)
- **交叉验证**: MNN Profiler + simpleperf PMU + simpleperf record 火焰图，三种工具互相印证
- **Roofline 模型**: 标注每类算子的 Compute-Bound vs Memory-Bound 属性，避免优化方向错误

**技术栈:**
MNN + ONNX Runtime + llama.cpp，骁龙 865 (Cortex-A77 + Adreno 650)，C++17 + CMake + Android NDK 交叉编译。

### 第二分钟: 核心数据和发现

**5 模型覆盖 5 种计算模式:**
MobileNetV2 (轻量 Conv + DWConv 混合) → ResNet50 (大卷积规则计算) → BERT (Encoder Transformer) → YOLOv8n (多尺度检测, 多分支) → Qwen2-0.5B (Decoder LLM, Prefill+Decode)。

**全模型 CPU 性能对比 (FP32, 单线程):**
- CNN 模型: MNN 领先 ORT 1.54-1.75x
- BERT: ORT 快 14% (683ms vs 599ms) — **唯一反转**
- GPU: ResNet50 加速 3.03x，但 BERT GPU 反慢 (0.41x)、YOLOv8n GPU 反慢 (0.22x)
- LLM: Qwen2-0.5B Q4_K_M 端侧 26.34 tok/s

### 第三分钟: BERT 根因分析（技术深度核心）

**完整分析链条:**
现象 (ORT 快 14%) → PMU 对比 (指令数相同 IPC 却低 12%) → 符号级热点 (expf+erff 占 11.7%) → 源码定位 (lP=1, CommonOptFunctionNeon.cpp:1879) → 量化 (计算/访存比差 4 倍) → 优化方案 (lP 1→4: 20-30% 收益)

**关键洞察:**
- MatMul 和 Conv1x1 走同一框架但不同 GEMM 路径
- Conv1x1: K=4 打包, 9.6 FLOPs/byte, ~40 GFLOPS
- MatMul: K=1 逐元素, 2.4 FLOPs/byte, 14.8 GFLOPS
- 同一硬件同一框架同一类运算 — 内存布局决定了 4 倍性能差距

**GPU 异构调度核心规则:**
- 放 GPU: 标准 Conv3×3 (3x), 大通道 Conv1×1
- 留 CPU: DWConv, LayerNorm, 逐元素, Concat/Reshape
- Transformer + 检测模型 → GPU 当前不适用

---

## 10 个追问应对

### Q1: "只测了一台手机，数据有说服力吗？"

**A:** "深度优先于广度。一台手机 CPU + GPU 双后端已提供足够分析维度。我定位到的 MNN MatMul lP=1 根因是跨 ARM 架构成立的 — 任何 AArch64 设备上 K 维不打包都是瓶颈。单设备深入比肤浅地测 10 台设备价值大 10 倍。如果有骁龙 8 Gen 2，可以加架构代际对比 — 但那是在单设备挖透之后的扩展。"

### Q2: "为什么不实际改代码验证 lP 优化？"

**A:** "定位到了具体行号和参数。但改 lP 从 1 到 4 不是改一行 — eP/hP 需要配合调整，B 矩阵 Pack 要重写，MNNPackedMatMul.S 的内循环汇编也要适配。预估 3-5 天工作量。面试场景下，定位能力比实施能力更体现技术深度。在真实工作中，我会把根因分析和优化方案写成 MR 描述提交。"

### Q3: "MNN Conv1x1 效率高但 MatMul 低，走同一条 GEMM 路径吗？"

**A:** "不是同一条路径 — 这是关键。Conv1x1 走 NCHW4c 布局 + K=4 打包，每次加载处理 4 个 K。MatMul 走通用 GEMM 路径，`MNNGetMatMulPackMode` 中 lP=1 意味着 K 维不做打包，每次迭代仅加载 20 个 float 执行 96 次 FMA。计算/访存比差 4 倍，所以一个 ~40 GFLOPS 一个 14.8 GFLOPS。这是 MNN 内部两条不同代码路径优化程度不同。"

### Q4: "Roofline 对你实际优化有什么帮助？"

**A:** "帮助我在选择优化方向时避免根本性错误。DWConv 在 Roofline 上落在 Memory-Bound 区域 — 优化的方向是减少内存访问（更小的数据类型、内存布局调整），不是调 Neon 指令。Conv1x1 在 Compute-Bound — 应该专注 SIMD 利用率和多线程。没有 Roofline 很容易在错误方向花时间。而且 Roofline 拐点 0.74 FLOP/Byte 是我从实测数据反推的真实硬件参数，不是文档理论值。"

### Q5: "GPU 为什么 BERT 反而慢？"

**A:** "三个原因。第一，MNN OpenCL 的 MatMul 在 Adreno 650 上是通用路径，没有专项优化。第二，Transformer 有大量小算子 — LayerNorm, Softmax, GELU, Add — 每个是一次 GPU kernel launch，启动开销累积超过计算本身。第三，精度也不对 — 余弦相似度降到 0.87，怀疑 OpenCL FP16 溢出。这告诉我们异构调度不是简单的'全放 GPU'，需要逐算子判断。"

### Q6: "和 llama.cpp 比 MNN 的 LLM 推理怎么样？"

**A:** "我没有强行用 MNN 的 llm.cpp 组件测 LLM，而是直接用 llama.cpp — 因为在 LLM 推理这个细分领域，llama.cpp 的优化深度远超通用框架。我的目标是为每个场景找到最合适的框架，不是凑数。Phase 5 实测 Qwen2-0.5B 达到 26.34 tok/s，同时做了完整的 Prefill vs Decode 计算特征分析。"

### Q7: "Prefill vs Decode 的区别是什么？"

**A:** "Prefill 是计算密集 — 一次性处理所有 tokens，类似 BERT 推理，算力利用率 60-80%。Decode 是访存密集 — 每 token 只做 1 token 的矩阵运算但要扫描整个 KV Cache（~22MB），算力利用率 <5%。这意味着 Decode 优化的根本方向不是算力，是内存带宽 — KV Cache 量化是最有效的优化。"

### Q8: "这个项目和推理引擎工程师的日常工作有多大关系？"

**A:** "几乎完全一致: Profile → 定位瓶颈 → 源码分析 → 优化 → 验证。不同的只是不直接提交到开源仓库。实际工作中性能工程师 80% 的时间也在分析和定位，20% 的时间写优化代码。这个项目完整展示了从数据到代码级的分析全过程。"

### Q9: "最大的挑战是什么？"

**A:** "BERT 的根因分析。一开始以为是 GEMM 微内核慢，但数据说 Conv1x1 GEMM 可以达到峰值算力。最后发现是 Erf GELU 图融合 + 线程调度的问题，需要横跨算子层、图层、调度层三个层次的联合分析。这需要真正理解 MNN 从图转换到几何运算到 Neon 指令的完整调用链。"

### Q10: "如果重新来一次会怎么做？"

**A:** "两个改进。第一，Phase 1 就用 simpleperf annotate 做指令级分析 — Phase 4 才发现 GELU expf+erff 占 11.7%，如果早发现可以早调整优先级。第二，加 CI pipeline 做性能回归检测 — 每次修改自动跑 benchmark。但整体方法论 (Profiling 驱动 + 二八原则 + 闭环验证) 是对的。"

---

## 技术亮点速记卡

面试时涉及具体技术细节时，以下关键数字可展示深度:

| 话题 | 关键数字 | 体现的能力 |
|------|---------|-----------|
| GEMM 微内核 | lP=1, eP=12, hP=8 | 源码阅读，微架构理解 |
| 计算/访存比 | Conv1x1 9.6 vs MatMul 2.4 | 性能建模能力 |
| Roofline | 拐点 0.74 FLOP/Byte | 硬件参数实测推算 |
| PMU | IPC 1.84 vs 2.06, cache miss 2.3x | simpleperf 使用能力 |
| PMU | branch-misses 37.3M vs 3.3M (11.4x) | 微架构瓶颈识别 |
| GPU 异构 | Conv3×3 3.03x, DWConv ~1x | 系统级架构决策 |
| GPU 反例 | BERT 0.41x, YOLOv8n 0.22x | 不盲目推 GPU |
| LLM Decode | ~0.01 FLOP/Byte, <5% 利用率 | Memory-Bound 本质理解 |
| 代码定位 | CommonOptFunctionNeon.cpp:1879 | 精准定位到行 |

---

## 面试 TIPS

### 回答框架: STAR

- **Situation**: 业界缺乏标准化端侧 Benchmark，各框架数据不可比
- **Task**: 建立 Profiling 驱动的性能分析体系，回答"MNN 为什么快"和"什么时候慢"
- **Action**: 5 模型 × 3 框架 × 2 后端全矩阵，三层深度分析，BERT 源码级根因定位
- **Result**: 定位 lP=1 根因 + 多线程优化后 BERT 提速 43% + 输出 GPU 异构调度规则 + LLM 优化清单

### 加分回答

- 主动引导到 BERT 根因故事 — 这是最体现技术深度的部分
- 提到 "同一个硬件同一个框架同一个运算，内存布局不同差 4 倍" — 展示对系统本质的理解
- 用 PMU 数据佐证 — "指令数相同但 IPC 低 12%" 比 "MNN 慢" 专业 10 倍
- 不要停留在 "MNN 比 ORT 快" 的层面 — 展示你知道 "什么时候不快" 和 "为什么不快"

### 避免的坑

- 不要说 "我只测了延迟" — 强调 Profiling + PMU + Roofline 多维分析
- 不要说 "GPU 加速所有模型" — 你有 BERT 0.41x 和 YOLOv8n 0.22x 的反例
- 不要只说成功不说失败 — BERT 反转反而是最有价值的发现
- 不要编造数据 — 所有数字都是从真实测绘中提取

---

> **版本:** v1.0 | **生成日期:** 2026-06-06
