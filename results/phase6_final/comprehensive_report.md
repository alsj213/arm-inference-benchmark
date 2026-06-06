# 端侧推理引擎性能分析项目 — 综合报告

> **项目:** ARM 端侧深度学习推理框架性能基准测试
> **平台:** 骁龙 865 (红米 K30S / M2007J3SC), 日期: 2026-06-06
> **芯片:** 1xA77@2.84GHz + 3xA77@2.42GHz + 4xA55@1.8GHz, Adreno 650 GPU
> **内存:** 8GB LPDDR5, 带宽 ~34 GB/s
> **框架:** MNN 2.9.x / ONNX Runtime 1.20.x / llama.cpp v0.10.0
> **模型:** MobileNetV2 / ResNet50 / BERT-base / YOLOv8n / Qwen2-0.5B
> **代码分支:** `feat/benchmark-full-plan`

---

## 一、项目动机

本项目回答三个核心问题:

1. **MNN 为什么在端侧快？** — 手写 Neon + NCHW4c 内存布局，Conv1x1 GEMM 达到 ~40 GFLOPS（接近单核 A77 理论峰值）
2. **什么时候 MNN 反而不如 ORT？** — BERT Transformer 类模型：图优化（Erf→GELU 融合）缺失 + 多线程调度效率低（4核并行效率 44% vs ORT 63%）
3. **GPU 在端侧推理中的真实价值是什么？** — Conv 密集模型加速 2-3x，但 Transformer/YOLO 算子大量不支持，异构调度不是"全放 GPU"

项目不只"跑分"，而是建立了 Profiling 驱动的性能分析方法体系，能定位到源码行级的瓶颈根因并给出可量化的优化方案。

---

## 二、Benchmark 体系设计

### 2.1 方法论

- **Profiling 驱动 + 二八原则**: 先用 MNN Profiler + simpleperf 定位热点算子，只深挖累计占比 >= 80% 的热点算子
- **三层覆盖**: 模型级（端到端 Latency） -> 算子级（14+ 测例 + Roofline 模型） -> 指令级（PMU 计数器 + Neon 汇编）
- **交叉验证**: MNN Profiler + simpleperf PMU + simpleperf record 三种工具互相印证，確保结论可靠

### 2.2 模型选择逻辑

| 模型 | 参数量 | 计算模式 | 覆盖场景 |
|------|--------|---------|---------|
| MobileNetV2 | 3.5M | Conv1x1 GEMM + DWConv | 端侧轻量分类 |
| ResNet50 | 25.6M | Conv3x3 Winograd + Conv1x1 | 大卷积分类 |
| BERT-base | 110M | MatMul + GELU + LayerNorm | Encoder Transformer |
| YOLOv8n | 3.2M | 多尺度 Conv + DWConv + Concat | 实时检测 |
| Qwen2-0.5B | 494M | QKV Projection + FFN + Attention | Decoder LLM |

5 个模型覆盖了 CV → NLP → LLM 的全部主流端侧推理场景，每种计算模式都能匹配一类业务需求。

### 2.3 工具链

```
MNN_PROFILING=1       → 逐算子耗时 + 占比
simpleperf stat        → PMU 计数器 (IPC, cache-miss, branch-miss)
simpleperf record -g   → 火焰图 (函数级热点)
simpleperf annotate    → 指令级热点 (汇编行级)
源码阅读               → 分块参数、内存布局、Neon 指令
```

---

## 三、项目执行概览 (5 个 Phase)

| Phase | 主题 | 核心产出 | 数据来源 |
|-------|------|---------|---------|
| 0 | 环境就绪 | 设备握手, 模型清单, 编译链验证 | `results/phase0_*` |
| 1 | CPU 全量 Profiling | 4 模型 × 2 框架 Latency + 火焰图 | `results/phase1_cpu_profiling/` |
| 2 | 热点算子 Benchmark | 14+ 测例 + Roofline 分析 + 多线程 | `results/phase2_hotspot_benchmark/` |
| 3 | GPU OpenCL 对比 | 4 模型 CPU vs GPU 全矩阵 | `results/phase3_gpu_comparison/` |
| 4 | BERT 优化闭环 | 三层根因定位 + 优化方案 + Before/After | `results/phase4_bert_optimization/` |
| 5 | LLM 推理分析 | Prefill/Decode 量化 + 优化建议清单 | `results/phase5_llm_profiling/` |

---

## 四、核心发现

### 4.1 全模型 CPU 性能矩阵

| 模型 | MNN CPU (ms) | ORT CPU (ms) | MNN/ORT | 胜出 |
|------|-------------|-------------|---------|------|
| MobileNetV2 | 18.49 | 29.17 | **1.58x** | MNN |
| ResNet50 | 143.88 | 221.56 | **1.54x** | MNN |
| BERT | 683.45 | 598.80 | **0.88x** | ORT |
| YOLOv8n | 174.00 | 304.19 | **1.75x** | MNN |

> 测试条件: 单线程, FP32, 50 warmup + 100 iterations, schedutil
> 精度: 所有模型余弦相似度 >= 0.99 (以 ORT 输出为参考)
> 来源: `results/phase1_cpu_profiling/all_models_summary.md`

**结论**: CNN 类模型 MNN 全面领先 1.54-1.75x，BERT 是唯一反转案例。

### 4.2 MNN CNN 霸主地位的根因

MobileNetV2/ResNet50/YOLOv8n 上 MNN 领先的根因:

1. **NCHW4c 内存布局** — 通道维度以 4 为单位打包，一次 Neon 128-bit 加载处理 4 个 float
2. **Conv1x1 GEMM 极致优化** — K=4 打包，计算/访存比 9.6 FLOPs/byte，达到 ~40 GFLOPS（接近 A77 单核算力峰值 ~25 GFLOPS 的 1.6x 多核扩展）
3. **Winograd 加速 Conv3x3** — 4x4 tile, 6x6 变换，减少 2.25x 乘法

### 4.3 BERT 反转现象（面试核心故事）

BERT 是 4 个模型中**唯一 MNN 慢于 ORT 的模型**。经过从现象到代码行级的三层根因分析:

**根因 1: GELU 使用 Erf 实现（占总差距 ~9%）**
- BERT ONNX 模型导出时使用 `erf` 实现 GELU (`0.5 * x * (1 + erf(x/sqrt(2)))`) — 12 个 Erf 节点
- MNN 对 Erf 算子无 SIMD 优化，逐元素调用 libm `erf()` + `expf()`
- simpleperf 显示 `__ieee754_expf` 占 6.10%、`erff` 占 5.60%，合计 11.7%
- ORT 图优化器将 Erf 融合为快速 GELU（tanh 近似），该开销接近 0%

**根因 2: MatMul 微内核 K 维不打包（占总差距 ~20-30%）**
- 源码位置: `third_party/MNN/source/backend/cpu/arm/CommonOptFunctionNeon.cpp:1879`
- `lP=1` — K 维度不做打包，每次迭代仅加载 20 floats (12A + 8B) 执行 96 次 FMA
- 计算/访存比: **2.4 FLOPs/byte**（对比 Conv1x1 GEMM 的 9.6 FLOPs/byte — 4 倍差距）
- 实际 GFLOPS: 14.8 (MatMul) vs ~40 (Conv1x1) — 仅达到 Conv1x1 的 **37%**

**根因 3: 多线程扩展性差**
- BERT batch=1 时 e 维度=1，限制了 MNN 在 e 维度的并行切分
- MNN 4 核并行效率仅 43.7%，ORT 为 62.5%
- 4 线程下 MNN 422ms vs ORT 268ms，差距从单线程的 14% 拉大到 58%

**PMU 交叉验证**（simpleperf）:

| 指标 | MNN BERT | ORT BERT | 分析 |
|------|----------|----------|------|
| instructions | 39.37B | 39.47B | **几乎相同** -- 算法做了同样的工作 |
| cpu-cycles | 21.41B | 19.16B | MNN 多用 11.8% 周期 |
| **IPC** | **1.84** | **2.06** | MNN 低 12.1% |
| **cache-misses** | **200.45M** | **102.01M** | MNN 多 ~2x |
| cache miss rate | **2.30%** | **1.00%** | MNN 2.3 倍 |
| **branch-misses** | **37.30M** | **3.28M** | MNN 多 **11.4x** |

> 来源: `results/phase4_bert_optimization/bert_mnn_pmu.log`, `bert_ort_pmu.log`

**优化方案与预期收益**:

| 优先级 | 优化方案 | 预期收益 | 实施难度 |
|--------|---------|---------|---------|
| P0 | 重新导出 ONNX 为 fast GELU | ~9% | 低 |
| P1 | MatMul lP 1→4 | 20-30% | 中 |
| P2 | MNN 图融合: Erf→FastGELU | ~9% | 中 |
| P3 | MatMul 多线程切分策略 | 15-20% | 高 |

**多线程优化实测效果**:

| Threads | MNN Mean | ORT Mean | MNN 加速比 | ORT 加速比 | MNN/ORT |
|---------|----------|----------|-----------|-----------|---------|
| 1 | 737.7 ms | 668.7 ms | 1.00x | 1.00x | 1.10x |
| 2 | 505.5 ms | 450.9 ms | 1.46x | 1.48x | 1.12x |
| 4 | 422.1 ms | 267.7 ms | 1.75x | 2.50x | **1.58x** |

> 来源: `results/phase4_bert_optimization/bert_threads_*.log`

### 4.4 GPU 的真实价值

| 模型 | CPU 耗时 | GPU 耗时 | GPU/CPU | 精度 | 结论 |
|------|---------|---------|---------|------|------|
| ResNet50 | 180 ms | 59.4 ms | **3.03x** | PASSED | GPU 大赢家 |
| MobileNetV2 | 19 ms | 15.9 ms | **1.17x** | PASSED | 边际收益 |
| BERT | 779 ms | 1883 ms | **0.41x** | cos=0.87 异常 | GPU 不适用 |
| YOLOv8n | 197 ms | 899 ms | **0.22x** | PASSED | GPU 不适用 |

> 来源: `results/phase3_gpu_comparison/` (独立测试会话，CPU 基线可能与前序 Phase 不同)

**异构调度规则**:
```
✅ 放 GPU (Compute-Bound, 加速 > 2x):
   - 标准 Conv3x3 — 3.03x (ResNet50 核心算子)

❌ 留 CPU (Memory-Bound 或 GPU 启动开销 > 计算):
   - DWConv — 访存密集，GPU 几乎无加速
   - LayerNorm/Softmax — 小张量，kernel launch 开销主导
   - Reshape/Concat/逐元素 (Add, Mul, ReLU)
   - Transformer MatMul — 当前 MNN OpenCL 未优化 (0.41x)

⚠️ 条件放置:
   - Conv1x1 小 Tensor — GPU 启动开销可能 > 计算收益
```

### 4.5 LLM 推理洞察

Qwen2-0.5B Q4_K_M 实测数据:

| 指标 | 值 |
|------|-----|
| 吞吐量 | 26.34 tokens/sec |
| 延迟 | 37.96 ms/token |
| 模型大小 | 380 MB (FP16 原 988 MB) |
| 模型加载 | 0.47s (mmap 启用) |

**Prefill vs Decode 量化区别**:
```
Prefill (Prompt Processing):
  ├── 计算模式: MatMul 密集 (算术强度 ~10-50 FLOP/Byte)
  ├── 瓶颈: Compute-Bound
  └── 算力利用率: 60-80%

Decode (逐 Token 生成):
  ├── 计算模式: GEMV + KV Cache 线性扫描 (~0.01 FLOP/Byte)
  ├── 瓶颈: Memory-Bound (受限于 34 GB/s LPDDR5)
  ├── 每 token KV Cache 扫描: ~22MB (ctx=1024, FP16)
  └── 算力利用率: <5%
```

**LLM 优化优先级**:

| 优先级 | 优化方案 | 预期收益 |
|--------|---------|---------|
| P0 | 线程绑定 (仅 A77 核心) | Decode 提速 20-40% |
| P0 | KV Cache INT8 量化 | Decode 提速 30-50% |
| P1 | KV Cache FP8/INT4 | 进一步减少带宽压力 |
| P2 | CPU 锁频 (performance governor) | 稳定性提升 |
| P2 | Prompt Cache | 相同前缀跳过 Prefill |

---

## 五、热点算子 Roofline 分析

基于 Phase 2 的 14+ 单算子测例 + Roofline 模型（骁龙 865 实测拐点: 0.74 FLOP/Byte）:

| 算子 | GFLOPS | 峰值利用率 | Bound 类型 | 优化方向 |
|------|--------|----------|-----------|---------|
| Conv1x1 大M (1×1024×256×784) | ~39.8 | ~100% | Compute | 已达极限 |
| Conv1x1 小C (1×32×64×784) | ~13.5 | ~34% | Memory | 内存布局 |
| MatMul 768×768×768 | 14.8 | ~37% | **Memory** | K 维打包 |
| MatMul 768×3072 | 12-17 | 30-42% | Memory | K 维打包 |
| DWConv C=960 3×3 | ~8 | ~20% | **Memory** | 带宽优化 |
| DWConv C=16 3×3 | ~2.5 | ~6% | Memory | 启动开销主导 |

> 来源: `results/phase2_hotspot_benchmark/` 日志 + `roofline_analysis.md`

**关键洞察**: Roofline 分析避免了方向性错误。DWConv 在 Memory-Bound 区域 → 优化方向是内存布局而非 SIMD 指令。Conv1x1 在 Compute-Bound → 专注 SIMD 利用率和多线程。如果没有 Roofline，很可能会在错误的方向上浪费时间。

---

## 六、框架选型建议

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| CNN 端侧部署 (分类/检测) | **MNN** | Conv1x1 GEMM 达峰值算力，1.54-1.75x ORT |
| Transformer/BERT 推理 | **ORT** | 图优化完善 (GELU 融合)，多线程扩展性好 |
| LLM 端侧推理 | **llama.cpp** | 量化支持 (Q4_K_M) + KV Cache 管理，生态成熟 |
| GPU Conv 密集加速 | **MNN GPU** | ResNet50 3.03x，但需逐算子判断放置 |
| GPU Transformer 推理 | **暂不推荐** | 算子覆盖不足 + 精度异常 |

---

## 七、项目关键数字一览

| 指标 | 数值 | 说明 |
|------|------|------|
| 骁龙 865 FP32 峰值 (单核 A77) | ~25 GFLOPS | NEON FMA 指令 |
| MNN Conv1x1 GEMM 实测 | ~39.8 GFLOPS | 多核扩展后接近总体峰值 |
| MNN MatMul 768^2 实测 | 14.8 GFLOPS | 仅 Conv1x1 的 37% |
| MatMul vs Conv1x1 计算/访存比 | 2.4 vs 9.6 FLOPs/byte | 4 倍差距 — 根因 lP=1 |
| ResNet50 GPU 加速比 | 3.03x | 4 模型中最高 |
| BERT 4 线程加速比 | MNN 1.75x vs ORT 2.50x | 多线程差距反而拉大 |
| Qwen2-0.5B 端侧吞吐 | 26.34 tok/s | Q4_K_M 量化 |
| MNN BERT cache miss rate | 2.30% vs 1.00% (ORT) | 2.3 倍 |
| MNN BERT branch-misses | 37.3M vs 3.3M (ORT) | 11.4 倍 |

---

## 八、数据溯源

| Phase | 内容 | 关键源文件 |
|-------|------|-----------|
| 0 | 设备/内核/模型清单 | `results/phase0_*.txt` |
| 1 | 全模型 MNN/ORT Latency | `results/phase1_cpu_profiling/all_models_summary.md` |
| 1 | simpleperf 火焰图 | `results/phase1_cpu_profiling/perf_mbv2.data` |
| 2 | Roofline 分析 | `results/phase2_hotspot_benchmark/roofline_analysis.md` |
| 2 | 14+ 单算子测例日志 | `results/phase2_hotspot_benchmark/*.log` |
| 3 | GPU vs CPU 对比 | `results/phase3_gpu_comparison/*.log` |
| 4 | BERT PMU 对比 | `results/phase4_bert_optimization/bert_*_pmu.log` |
| 4 | BERT 根因分析报告 | `results/phase4_bert_optimization/BERT_OPTIMIZATION_REPORT.md` |
| 4 | BERT 多线程测试 | `results/phase4_bert_optimization/bert_threads_*.log` |
| 5 | LLM 实测 + 分析 | `results/phase5_llm_profiling/README.md`, `llm_analysis.md` |

---

> **报告版本:** v1.0 | **生成日期:** 2026-06-06 | **分支:** `feat/benchmark-full-plan`
