# 简历项目描述

---

## ★ ARM 端侧多框架推理性能 Benchmark 与优化实践

```
项目使用技术: C++17, ARM Neon, MNN, ONNX Runtime, llama.cpp, simpleperf, CMake
硬件平台: 骁龙 865 (红米 K30S) — Cortex-A77 + A55 + Adreno 650
项目周期: 2026-05 至 2026-06
```

### 项目描述

建立端侧推理框架标准化性能 Benchmark 体系，以 MNN 为核心，深度对比 ONNX Runtime 和 llama.cpp 两种优化路线，输出系统级性能分析方法论。覆盖 5 模型 × 3 框架 × 2 后端（CPU + GPU）全矩阵，按 Profiling 驱动 + 二八原则对热点算子做三层深度分析，定位到源码行级瓶颈并给出可量化优化方案。

### 核心工作

**1. 搭建 Profiling 驱动的 Benchmark 体系**
- 设计 5 模型覆盖 CV 到 LLM 的五种计算模式（MobileNetV2 / ResNet50 / BERT / YOLOv8n / Qwen2-0.5B）
- 三层分析深度: 模型级（端到端 Latency） → 算子级（14+ 单算子测例 + Roofline 模型） → 指令级（PMU 计数器 + Neon 汇编）
- 交叉验证: MNN Profiler + simpleperf PMU + simpleperf record 火焰图，三种工具互相印证

**2. 热点算子深度 Benchmark 与 Roofline 分析**
- 设计 14+ 单算子测例，覆盖 Conv1x1/Conv3x3/DWConv/MatMul 的形状分桶和边界条件（对齐/未对齐、小/大 channel）
- 基于骁龙 865 实测数据构建 Roofline 模型（拐点 0.74 FLOP/Byte），标注每个算子的 Compute-Bound vs Memory-Bound 属性
- 量化多线程加速比 (1/2/4 线程) 和并行效率

**3. BERT 性能反转根因分析（项目最核心产出）**
- 发现 BERT 是 4 个模型中唯一 MNN 慢于 ORT 的案例（ORT 快 14%）
- 使用 simpleperf PMU 对比定位到三层根因:
  - 根因 1: GELU 使用 Erf 实现而非快速 tanh 近似（expf+erff 占 CPU 11.7%）
  - 根因 2: MatMul 微内核 K 维不打包（lP=1, CommonOptFunctionNeon.cpp:1879），计算/访存比仅 Conv1x1 GEMM 的 1/4
  - 根因 3: 多线程并行效率 44% vs ORT 63%（batch=1 限制并行粒度）
- 多线程优化实测: MNN BERT 推理 754ms → 422ms (-43%)

**4. CPU vs GPU 异构调度分析**
- 启用 MNN OpenCL GPU 后端，完成 4 模型 CPU vs GPU 全矩阵对比
- 量化 GPU 加速比: ResNet50 3.03x / MobileNetV2 1.17x / BERT 0.41x / YOLOv8n 0.22x
- 输出异构调度规则: 标准 Conv3×3 → GPU (3x)，DWConv/LayerNorm/逐元素 → CPU，Transformer/检测模型 → 当前不适合 GPU

**5. LLM 端侧推理 Profiling**
- 实测 Qwen2-0.5B Q4_K_M 量化模型在骁龙 865 的推理性能（26.34 tok/s）
- 量化 Prefill vs Decode 的计算特征差异: Prefill 计算密集 (60-80% 算力利用率) vs Decode 访存密集 (<5% 算力利用率)
- 输出 15 项优化建议清单（KV Cache INT8 量化、线程绑定、Prompt Cache 等）

### 项目成果

- MNN BERT 推理 4 线程优化 43% (754ms → 422ms)
- 建立端侧推理性能数据库: 5 模型 × 3 框架 × 14+ 测例
- 输出 CPU vs GPU 异构调度规则 (Conv 密集 → GPU，Transformer/检测 → CPU)
- 产出 LLM 端侧推理优化建议清单 (15 个优化点)
- 完整性能报告、模型分析卡片、面试准备材料

### 技术亮点

1. **源码级根因定位**: 从 Profiling 数据逐层下钻到 `CommonOptFunctionNeon.cpp:1879` 的 `lP=1` 参数，解释了为什么同一框架下 Conv1x1 达到 ~40 GFLOPS 而 MatMul 仅 14.8 GFLOPS
2. **PMU 交叉验证**: 使用 simpleperf 量化 IPC (1.84 vs 2.06)、cache miss rate (2.30% vs 1.00%)、branch-misses (11.4x) 等微架构指标，将感性判断转为定量对比
3. **Roofline 指导优化方向**: 实测反推硬件参数 (0.74 FLOP/Byte)，标注每个算子 Compute-Bound vs Memory-Bound 属性，避免方向性错误
4. **异构调度决策**: 不只是跑分，而是输出可操作的算子放置规则

---

## 针对不同岗位的定制版本

### 推理引擎开发 (MNN/ncnn/TNN)

```
★ ARM 端侧推理引擎性能分析与优化
   深入 MNN GEMM 微内核源码，定位 BERT Transformer MatMul 的性能瓶颈 (lP=1 导致
   K 维不打包，计算/访存比仅为 Conv1x1 的 1/4)。完成多线程优化 (754→422ms, -43%)，
   输出 MNN 图融合 (Erf→FastGELU) 和微内核优化方案。熟练使用 ARM PMU (simpleperf)
   做指令级性能分析。
```

### 端侧 AI 部署工程师

```
★ 端侧深度学习模型部署与性能调优 (骁龙 865)
   搭建 MNN/ORT/llama.cpp 多框架 Benchmark 体系，覆盖 CV+NLP+LLM 全场景。
   完成 CPU vs GPU 异构部署分析 (ResNet50 GPU 3.03x)，输出逐算子放置规则。
   实测 Qwen2-0.5B 端侧推理 (26.34 tok/s)，输出 15 项 LLM 推理优化建议。
```

### 性能分析工程师

```
★ C++ 性能分析与优化 — ARM 端侧推理引擎
   基于 ARM PMU (simpleperf) 建立 Profiling 驱动的性能分析体系。
   使用 Roofline 模型标注算子瓶颈类型，定位到 MatMul 微内核源码行级根因。
   IPC、cache miss rate、branch-misses 等微架构指标定量对比 (MNN vs ORT)。
   多线程并行效率分析 (1/2/4 线程, 44% vs 63%)。
```

---

## 面试预判问题自测清单

- [ ] 一句话说清楚项目是做什么的
- [ ] 为什么选这 5 个模型 — 每种代表一类计算模式
- [ ] MNN 为什么在 CNN 上快 — NCHW4c + K=4 打包
- [ ] BERT 为什么反转 — 能讲清三层根因 (GELU Erf / lP=1 / 多线程)
- [ ] 怎么定位到 lP=1 的 — Profiling → PMU → 源码的过程要清晰
- [ ] Conv1x1 GEMM 的 GFLOPS 是怎么算出来的 — FLOP 公式 + 耗时倒数
- [ ] CPU vs GPU 的规则是什么 — 能背出 3.03x / 0.22x / 0.41x
- [ ] Decode 为什么 <5% 利用率 — Memory-Bound 本质 + KV Cache 带宽压力
- [ ] 如果再来一次怎么做 — simpleperf annotate 提前 + CI pipeline
- [ ] 最有技术深度的是什么 — BERT 根因分析 (完整闭环)

---

> **版本:** v1.0 | **生成日期:** 2026-06-06
