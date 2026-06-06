# 面试讲述话术 -- 端侧推理引擎性能分析项目

> **项目:** 基于骁龙 865 的 ARM 端侧深度学习推理框架全量性能分析
> **用途:** 面试中讲述项目经历的结构化话术
> **覆盖时长:** 1 分钟电梯演讲 / 3 分钟项目展开 / 10 个追问应对

---

## 1 分钟版 (电梯演讲)

"我基于骁龙 865 搭了一套完整的端侧推理性能分析体系。选了 5 个模型覆盖 CV 到 LLM 的五种计算模式，用 MNN Profiler + simpleperf 做全量 Profiling，按二八原则对热点算子做三层深度的 Benchmark。

核心发现是 CNN 模型上 MNN 领先 ORT 1.5-1.75 倍，但 BERT 上 ORT 反超 14%。我定位到根因是 MNN MatMul 微内核的 lP=1 参数导致 K 维不打包--计算/访存比只有 Conv1x1 的四分之一。把 lP 改成 4，预期提速 20-30%。

还做了 CPU vs GPU 的异构调度分析和 Qwen2-0.5B 的端侧 LLM 推理 Profiling，输出了一套完整的算子放置规则和优化优先级清单。"

---

## 3 分钟版 (展开讲述)

### 第一分钟: 项目背景和设计

**为什么做:**
业界缺乏标准化的端侧推理 Benchmark。各框架宣称的性能数据在不同硬件上差异很大，且缺乏系统性的瓶颈分析方法论。我做的不仅仅是"跑分"，而是建立一套 Profiling 驱动的性能分析方法体系。

**怎么设计:**
- **Profiling 驱动 + 二八原则:** 先全量 Profiling 找热点，再聚焦累计占比 >= 80% 的算子做深度分析
- **三层覆盖:** 模型级 (端到端 Latency) -> 算子级 (80+ 单算子测例 + Roofline) -> 指令级 (PMU 计数器 + 汇编源码分析)
- **交叉验证:** MNN Profiler + simpleperf PMU + simpleperf record 火焰图，三种工具互相印证

**技术栈:**
MNN + ONNX Runtime + llama.cpp，骁龙 865 (Cortex-A77 + Adreno 650)，C++17 + CMake + Android NDK 交叉编译 + Python 数据分析。

### 第二分钟: Benchmark 方法论 + 核心数据

**5 模型覆盖 5 种计算模式:**
MobileNetV2 (轻量 Conv) -> ResNet50 (大卷积) -> BERT (Transformer Encoder) -> YOLOv8n (多尺度检测) -> Qwen2-0.5B (LLM Decoder)。每个代表一类端侧推理场景。

**核心数据:**
- CNN 模型: MNN 领先 ORT 1.54-1.75x (MobileNetV2/ResNet50/YOLOv8n)
- BERT 反转: ORT 快 14% (683ms vs 599ms)，4 模型中唯一反转
- GPU: ResNet50 加速 3.03x，但 BERT/YOLOv8n GPU 反慢
- LLM: Qwen2-0.5B Q4_K_M 端侧 26.34 tok/s

### 第三分钟: 优化闭环 + 技术深度

**BERT 根因定位完整链条:**
现象 (ORT 快 14%) -> PMU 对比 (IPC 1.84 vs 2.06) -> 符号级热点 (GELU expf+erff 占 11.7%) -> 源码定位 (lP=1, CommonOptFunctionNeon.cpp:1879) -> 优化方案 (lP 1->4, 快速 GELU 融合) -> 效果预估 (20-30%)

**技术洞察:**
- 计算/访存比: Conv1x1 = 9.6 FLOPs/byte vs MatMul = 2.4 FLOPs/byte -- 4 倍差距
- 这解释了为什么 MNN 在 CNN 上快而 BERT 上慢--同样的 GEMM 微内核，不同的打包策略
- Cache miss 2.30% vs 1.00% 和 branch-misses 11.4x 都是 lP=1 的直接后果

**GPU 异构调度核心规则:**
- 放 GPU: 标准 Conv3x3 (3x 加速, 计算密集)
- 留 CPU: DWConv/LayerNorm/逐元素算子 (GPU 启动开销 > 计算)
- Transformer: 当前 MNN OpenCL 不适合

---

## 10 个追问应对

### Q1: "只测了一台手机，数据有说服力吗？"

**A:** "深度优先于广度。一台手机 CPU + GPU 双后端已经提供了足够的分析维度。我在这一台设备上定位到了 MNN MatMul 的 lP=1 根因，这是跨架构有效的发现--它在任何 ARM 设备上都成立。多设备对比的价值在于架构差异分析 (比如骁龙 8 Gen 2 的独立 NPU vs 纯 CPU 推理)，那是我下一步要做的。但先把单设备的性能特征挖透，比肤浅地测 10 台设备更有价值。"

### Q2: "为什么不实际改代码验证 lP 优化？"

**A:** "定位到了具体行号和参数。改 lP 从 1 到 4 不只是改一行--整套分块参数 eP/hP 需要配合调整，B 矩阵的 Pack 逻辑也需要重新实现，MNNPackedMatMul.S 的内循环汇编也要重写。这是一个 3-5 天的工作量。我把根因分析和优化方向写清楚了，如果这是工作项目，下一步就是改代码跑回归测试。在面试场景下，定位能力比实施能力更体现技术深度。"

### Q3: "MNN 为什么 Conv1x1 效率高但 MatMul 低？走的是同一条 GEMM 路径吗？"

**A:** "不是同一条路径，这是关键。Conv1x1 走 NCHW4c 布局 + K=4 打包路径--通道维度以 4 为单位打包，每次内存加载处理 4 个 K 的数据。MatMul 走通用 GEMM 路径，`MNNGetMatMulPackMode` 中 lP=1 意味着 K 维不做打包--每次迭代只加载 20 个 float 做 96 次 FMA。两者的计算/访存比差 4 倍，所以一个达到 ~40 GFLOPS，另一个只有 14.8 GFLOPS。这是 MNN 内部两条不同代码路径，优化程度不同。"

### Q4: "Roofline 对你实际优化有什么帮助？"

**A:** "举两个例子。DWConv 在 Roofline 上落在 Memory-Bound 区域，算术强度极低，这意味着优化方向应该是减少内存访问而不是调计算指令--比如用更小的数据类型或内存布局调整。Conv1x1 在 Compute-Bound 区域，应该专注 SIMD 指令利用率和多线程。Roofline 帮助我避免了方向性错误--不会在 Memory-Bound 的算子上花时间优化计算，反之亦然。而且 Roofline 拐点的位置 (0.74 FLOP/Byte) 是我从实测数据反推的真实硬件参数，不是文档里的理论值。"

### Q5: "GPU 为什么 BERT 反而慢？"

**A:** "三个原因。第一，MNN OpenCL 的 MatMul 在 Adreno 650 上没有专项优化，用的是通用路径。第二，Transformer 有大量小算子--LayerNorm、Softmax、GELU、Add，每个都是一次 GPU kernel launch，启动开销累积起来比计算本身还大。第三，精度也不对，余弦相似度降到 0.87，怀疑是 OpenCL FP16 溢出。这告诉我们异构调度不是简单的'全部放 GPU'，而是需要逐算子判断。"

### Q6: "和 llama.cpp 比 MNN 的 LLM 推理怎么样？"

**A:** "这个问题本身就反映了我的设计思路。我没有强行用 MNN 的 llm.cpp 组件测 LLM，而是直接用 llama.cpp--因为在 LLM 推理这个细分领域，llama.cpp 的优化深度远超通用框架。MNN 的 llm.cpp 组件还在早期。我的目标是找到每个框架最适合的场景，而不是凑数。Phase 5 中 llama.cpp 实测 Qwen2-0.5B 达到 26.34 tok/s，同时我做了完整的 Prefill vs Decode 计算特征分析，这才是面试中有价值的。"

### Q7: "你觉得端侧推理最关键的优化是什么？"

**A:** "内存布局 > 指令优化 > 图优化。内存布局决定了数据复用率，直接影响计算/访存比。这也是为什么 MNN NCHW4c 布局的 Conv1x1 能到 ~40 GFLOPS 而 MatMul 只有 14.8 GFLOPS--同一个硬件，同一个框架，内存布局完全不一样。指令优化 (Neon Intrinsics) 是锦上添花，但如果数据都不在 Cache 里，再好的指令也是等内存。图优化 (算子融合) 减少 kernel launch，对 Transformer 类模型尤其重要。"

### Q8: "为什么选择这 5 个模型？"

**A:** "不是随便选的，每个代表一种计算模式: MobileNetV2 = 端侧轻量级 Conv + DWConv 混合；ResNet50 = 大卷积网络，规则密集型计算；BERT = Encoder Transformer，MatMul + GELU + LayerNorm；YOLOv8n = 多尺度检测，DWConv + Concat + Reshape 混合；Qwen2-0.5B = Decoder LLM，Prefill+Decode 两种完全不同的计算瓶颈。5 个模型覆盖了从 CV 到 NLP 到 LLM 的主流端侧场景，如果有人问我'某个新的端侧模型应该选什么框架'，我可以用这 5 个模型的计算模式做类比。"

### Q9: "如果让你重新设计这个项目，你会怎么做？"

**A:** "两个改进。第一，一开始就用 simpleperf annotate 做指令级热点分析--我 Phase 4 才发现 GELU 的 expf+erff 占了 11.7%，如果 Phase 1 就知道，可以提前调整测试优先级。第二，加一个自动化 CI pipeline--每次 commit 自动跑 benchmark，防止性能回归。但整体方法论是对的: Profiling 驱动 + 二八原则 + 闭环验证，这套思路用在任何性能分析项目上都成立。"

### Q10: "这个项目对你面试目标岗位的价值？"

**A:** "它展示了我完整的性能优化能力链: 测试体系搭建 -> Profiling 方法 -> 瓶颈定位 -> 根因分析 (到代码行级) -> 优化方案设计 -> 效果预估。这和推理引擎工程师的日常工作完全一致。更重要的是，它不是调 API 或跑脚本，而是深入到了 GEMM 微内核的分块参数和 Neon 汇编级别。我能在面试中把 Conv1x1 为什么 40 GFLOPS 而 MatMul 为什么 14.8 GFLOPS 解释清楚，是因为我真正读了 MNN 的源码。"

---

## 技术亮点速记卡

面试时如果被问到具体技术细节，以下关键数字可以用来展示深度：

| 话题 | 关键数字 | 体现的能力 |
|------|---------|-----------|
| GEMM 微内核 | lP=1, eP=12, hP=8 | 源码阅读，微架构理解 |
| 计算/访存比 | Conv1x1 9.6 vs MatMul 2.4 | 性能建模能力 |
| Roofline | 拐点 0.74 FLOP/Byte | 硬件参数实测推算 |
| PMU | IPC 1.84 vs 2.06, cache miss 2.3x | simpleperf 使用能力 |
| GPU 异构 | Conv3x3 3.03x, DWConv ~1x | 系统级架构决策 |
| LLM Decode | ~0.01 FLOP/Byte, <5% 算力利用率 | Memory-Bound 本质理解 |
| 代码定位 | CommonOptFunctionNeon.cpp:1879 | 精准定位能力 |

---

## 简历项目描述

```
★ ARM 端侧多框架推理性能 Benchmark 与优化实践
   技术栈: C++17 / ARM Neon / MNN / ONNX Runtime / llama.cpp / simpleperf
   硬件: 骁龙 865 (A77+A55+Adreno 650), 红米 K30S

   建立 Profiling 驱动的端侧推理性能分析体系，覆盖 5 模型 x 3 框架 x 2 后端全矩阵。
   按二八原则对热点算子做三层深度 Benchmark (80+ 测例)，结合 Roofline 模型标注
   算子瓶颈类型。定位 MNN 在 BERT 上的性能反转根因 (MatMul 微内核 lP=1，
   CommonOptFunctionNeon.cpp:1879)，预期优化空间 20-30%。完成 CPU vs GPU 异构
   调度分析 (ResNet50 GPU 3.03x) 和 Qwen2-0.5B 端侧 LLM 推理 Profiling (26.34 tok/s)，
   产出全套性能分析报告和优化建议清单。
```

---

> **版本:** v1.0 | **生成日期:** 2026-06-06
