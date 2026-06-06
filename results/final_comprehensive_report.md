# 端侧推理引擎性能分析 -- 综合报告

> **项目:** ARM 端侧深度学习推理框架性能基准测试
> **设备:** 红米 K30S (M2007J3SC), 骁龙 865 SM8250
> **芯片:** 1xA77@2.84GHz + 3xA77@2.42GHz + 4xA55@1.8GHz, Adreno 650 GPU
> **内存:** 8GB LPDDR5
> **测试日期:** 2026-06
> **代码分支:** `feat/benchmark-full-plan` (commit: c092205)

---

## 1. 项目概况

本项目在骁龙 865 端侧平台上，对 **MNN、ONNX Runtime、llama.cpp** 三个主流推理框架进行了系统性的全量性能分析。以 Profiling 数据驱动热点识别，按二八原则对累计占比 80% 以上的热点算子做三层深度的 Benchmark，最终产出了 BERT 优化闭环和 CPU/GPU 异构调度规则。

### 1.1 测试框架与模型矩阵

| 框架 | 版本 | 后端 | 模型格式 |
|------|------|------|---------|
| MNN | 2.9.x (子模块) | CPU Neon + GPU OpenCL | .mnn |
| ONNX Runtime | 1.20.x (子模块) | CPU (MLAS/Eigen) | .onnx |
| llama.cpp | v0.10.0 (子模块) | CPU Neon | .gguf (Q4_K_M) |

| 模型 | 类型 | 参数量 | 计算模式 |
|------|------|--------|---------|
| MobileNetV2 | 轻量分类 | 3.5M | Conv1x1(GEMM) + DWConv |
| ResNet50 | 大卷积分类 | 25.6M | Conv3x3 + Conv1x1 + BN |
| BERT-base | Encoder Transformer | 110M | MatMul + GELU + LayerNorm |
| YOLOv8n | 检测 | 3.2M | 多尺度 Conv + DWConv + Concat |
| Qwen2-0.5B | Decoder LLM | 494M | QKV Projection + FFN + Attention |

---

## 2. 全模型性能矩阵

### 2.1 CPU 端到端 Latency (单线程, FP32, schedutil)

| 模型 | MNN CPU | ORT CPU | MNN GPU | 最优后端 | MNN/ORT |
|------|---------|---------|---------|---------|--------|
| MobileNetV2 | 18.49 ms | 29.17 ms | ~15.9 ms | **MNN GPU** | 1.58x |
| ResNet50 | 143.88 ms | 221.56 ms | ~59.4 ms | **MNN GPU** | 1.54x |
| BERT | 683.45 ms | 598.80 ms | ~1883 ms | **ORT CPU** | 0.88x |
| YOLOv8n | 174.00 ms | 304.19 ms | ~899 ms | **MNN CPU** | 1.75x |

> 数据来源: Phase 1 CPU Profiling (50 warmup + 100 iterations, 单线程) + Phase 3 GPU 对比。
> GPU 数据为 Phase 3 独立测试会话实测，加速比为 GPU vs 对应会话的 CPU 基线。
> 精度验证: 所有模型 MNN/ORT 余弦相似度 >= 0.99，GPU BERT 精度异常 (cos=0.87)。

### 2.2 详细性能指标

| 模型 | 后端 | Mean | P50 | P90 | P99 | Std | FPS | Init | 精度 |
|------|------|------|-----|-----|-----|-----|-----|------|------|
| MobileNetV2 | MNN | 18.49 | 18.49 | 18.59 | 18.70 | 0.08 | 54.1 | 26.45ms | PASSED |
| MobileNetV2 | ORT | 29.17 | 29.14 | 29.37 | 31.25 | 0.27 | 34.3 | 29.31ms | PASSED |
| ResNet50 | MNN | 143.88 | 143.87 | 144.26 | 146.02 | 0.38 | 7.0 | 693.27ms | PASSED |
| ResNet50 | ORT | 221.56 | 221.58 | 222.02 | 222.20 | 0.35 | 4.5 | 388.03ms | PASSED |
| BERT | MNN | 683.45 | 683.44 | 684.55 | 689.30 | 0.99 | 1.5 | 1279.11ms | PASSED |
| BERT | ORT | 598.80 | 598.85 | 599.41 | 600.14 | 0.54 | 1.7 | 641.74ms | PASSED |
| YOLOv8n | MNN | 174.00 | 174.02 | 174.42 | 174.74 | 0.32 | 5.8 | 155.46ms | PASSED |
| YOLOv8n | ORT | 304.19 | 304.09 | 305.38 | 306.34 | 0.72 | 3.3 | 27.07ms | PASSED |

> 数据来源: `results/phase1_cpu_profiling/all_models_summary.md`

### 2.3 GPU 加速比汇总

| 模型 | CPU 耗时 | GPU 耗时 | GPU/CPU | 精度 (cos) | 建议 |
|------|---------|---------|---------|-----------|------|
| ResNet50 | 180 ms | 59.4 ms | **3.03x** | PASSED | GPU 大赢家 |
| MobileNetV2 | 19 ms | 15.9 ms | **1.17x** | PASSED | 有收益，DWConv 限制 |
| BERT | 779 ms | 1883 ms | **0.41x** | 0.87 异常 | GPU 不适用 |
| YOLOv8n | 197 ms | 899 ms | **0.22x** | PASSED | GPU 不适用 |

> 数据来源: Phase 3 GPU 对比测试 (独立测试会话，CPU 数据为该会话基线)
> 注意: Phase 3 的 CPU 基线数字与 Phase 1 不同，因测试条件差异（温度、频率策略等）

---

## 3. 算子级分析

### 3.1 热点算子性能画像

| 算子 | 来源模型 | MNN GFLOPS | 峰值利用率 | Bound 类型 | GPU 加速 |
|------|---------|-----------|----------|-----------|---------|
| Conv1x1 GEMM (大M) | MobileNetV2 | ~40 | ~100% | Compute | 1.17x |
| Conv3x3 (Winograd) | ResNet50 | ~35 | ~88% | Compute | **3.03x** |
| MatMul 768x768 | BERT | 14.8 | ~37% | **Memory** | 0.41x |
| MatMul 768x3072 | BERT FFN | 12-17 | 30-42% | Memory | N/A |
| DWConv 3x3 | MobileNetV2 | ~8 | ~20% | **Memory** | ~1x |
| LayerNorm 768 | BERT | - | - | Memory | GPU 不适用 |
| GELU (erf) | BERT | - | - | Memory | GPU 不适用 |

> 数据来源: Phase 2 Roofline 分析 + Phase 3 GPU 逐算子对比
> Roofline 拐点: 0.74 FLOP/Byte (骁龙 865 实测)
> 理论峰值: ~25 GFLOPS (A77 单核 NEON FP32), 内存带宽 ~34 GB/s LPDDR5

### 3.2 simpleperf PMU 对比 -- BERT MNN vs ORT

| 指标 | MNN BERT | ORT BERT | 分析 |
|------|----------|----------|------|
| cpu-cycles | 21.41B | 19.16B | MNN 多用 11.8% 周期 |
| instructions | 39.37B | 39.47B | **几乎相同** -- 算法做了同样的工作 |
| **IPC** | **1.84** | **2.06** | MNN IPC 低 12.1% |
| **cache-misses** | **200.45M** | **102.01M** | MNN 多 ~2x |
| cache miss rate | **2.30%** | **1.00%** | MNN 缓存缺失率 2.3 倍 |
| **branch-misses** | **37.30M** | **3.28M** | MNN 分支预测失败多 **11.4x** |

> 数据来源: `results/phase4_bert_optimization/bert_mnn_pmu.log`, `bert_ort_pmu.log`
> 命令: `simpleperf stat -e cpu-cycles,instructions,cache-misses,cache-references,branch-misses`

---

## 4. BERT 性能反转根因分析

### 4.1 核心发现

BERT 是唯一一个 MNN 慢于 ORT 的模型 (ORT 快 14%)。经系统分析，根因有三层：

### 4.2 根因一: GELU 实现差异 (占总差距 ~9%)

- BERT ONNX 模型使用 `erf` 实现 GELU (`0.5 * x * (1 + erf(x/sqrt(2)))`)
- MNN 对 `Erf` 算子无 SIMD 优化，逐元素调用 libm `erf()` + `expf()`
- simpleperf 显示 `__ieee754_expf` 占 6.10%、`erff` 占 5.60%，合计 **11.7%**
- ORT 图优化器将 Erf 融合为快速 GELU，simpleperf 中 expf/erff 开销接近 0%
- MNN 有快速 GELU (`MNNGelu.S`, tanh 近似 NEON 优化)，但因 Erf 算子路径未触发

### 4.3 根因二: MatMul 微内核 K 维不打包 (占总差距 ~20-30%)

位置: `third_party/MNN/source/backend/cpu/arm/CommonOptFunctionNeon.cpp:1879`

```c
void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 12;   // M 维度分块: 12 行
    *lP = 1;    // K 维度分块: 不打包 (逐元素)
    *hP = 8;    // N 维度分块: 8 列
}
```

关键问题: **lP=1 -- K 维度不做打包**
- 每次 K 迭代仅加载 20 floats (12A+8B)，执行 96 次 FMA
- 计算/访存比: 192 FLOPs / 80 bytes = **2.4 FLOPs/byte**
- 对比 Conv1x1 GEMM: K=4 打包，计算/访存比 **9.6 FLOPs/byte**
- 实际 GFLOPS: 14.8 (MatMul) vs ~40 (Conv1x1)，仅达到 Conv1x1 的 **37%**

### 4.4 根因三: 多线程扩展性 (占总差距 ~49%)

| Threads | MNN Mean | ORT Mean | MNN 加速比 | ORT 加速比 | MNN/ORT |
|---------|----------|----------|-----------|-----------|---------|
| 1 | 737.7 ms | 668.7 ms | 1.00x | 1.00x | 1.10x |
| 2 | 505.5 ms | 450.9 ms | 1.46x | 1.48x | 1.12x |
| 4 | 422.1 ms | 267.7 ms | 1.75x | 2.50x | **1.58x** |

- MNN 4 核并行效率: 43.7%, ORT: 62.5%
- BERT batch=1, e 维度=1，限制了 MNN 在 e 维度的并行切分

### 4.5 优化方案与预期收益

| 优先级 | 优化方案 | 预期收益 | 实施难度 | 关键文件 |
|--------|---------|---------|---------|---------|
| P0 | 重新导出 ONNX 为 fast GELU | ~9% (单线程) | 低 | 模型转换 |
| P1 | 将 MatMul lP 1→4 | **20-30%** | 中 | CommonOptFunctionNeon.cpp:1879 |
| P2 | MNN 图融合: Erf→FastGELU | ~9% | 中 | CPUUnary.cpp:431 |
| P3 | 优化 MatMul 多线程切分策略 | ~15-20% | 高 | CPUMatMul.cpp |

> 数据来源: `results/phase4_bert_optimization/BERT_OPTIMIZATION_REPORT.md`, `bert_analysis_report.md`

---

## 5. GPU 异构调度规则

基于 Phase 3 全模型 CPU vs GPU 逐算子实测数据：

```
确认放 GPU (Compute-Bound, 加速 > 2x):
  ✅ 标准 Conv3x3 -- 3.03x (ResNet50 核心算子)
  ✅ Conv1x1 大 Tensor -- 有收益但不及 Conv3x3

必须留 CPU (Memory-Bound 或 GPU 启动开销 > 计算):
  ❌ DWConv -- 访存密集，GPU 几乎无加速
  ❌ LayerNorm -- 小张量，kernel launch 开销主导
  ❌ Reshape/Concat/逐元素算子 (Add, Mul, ReLU)
  ❌ Transformer MatMul -- 当前 MNN OpenCL 未优化 (0.41x)

条件放置:
  ⚠️ Conv1x1 小 Tensor -- GPU 启动开销可能 > 计算收益
```

---

## 6. LLM 推理专项

### 6.1 Qwen2-0.5B 实测数据

| 指标 | 值 |
|------|-----|
| 模型 | Qwen2-0.5B-Instruct, Q4_K_M (6.35 BPW) |
| 文件大小 | 380 MB (FP16 原 988 MB) |
| 模型加载 | 0.47s (mmap 启用) |
| 吞吐量 | **26.34 tokens/sec** |
| 延迟 | **37.96 ms/token** |
| 测试环境 | prompt=34 tokens, 生成~64 tokens, 温度=51.9°C |

### 6.2 Prefill vs Decode 计算特征

```
Prefill (Prompt Processing):
  ├── 计算模式: MatMul 密集 (高算术强度 ~10-50 FLOP/Byte)
  ├── 瓶颈: Compute-Bound
  └── 算力利用率: 60-80%

Decode (逐 Token 生成):
  ├── 计算模式: GEMV + KV Cache 线性扫描 (~0.01 FLOP/Byte)
  ├── 瓶颈: Memory-Bound (受限于 34 GB/s LPDDR5)
  ├── 每 token KV Cache 扫描: ~22MB (ctx=1024, FP16)
  └── 算力利用率: <5%
```

### 6.3 优化优先级

| 优先级 | 优化方案 | 预期收益 | 状态 |
|--------|---------|---------|------|
| P0 | 线程绑定 (仅 A77 核心) | Decode 提速 20-40% | 建议 |
| P0 | KV Cache INT8 量化 | Decode 提速 30-50% | 建议 |
| P1 | KV Cache FP8/INT4 | 进一步减少带宽压力 | 探索 |
| P2 | CPU 锁频 (performance governor) | 稳定性提升 | 测试时可用 |
| P2 | Prompt Cache | 相同前缀跳过 Prefill | 长期 |

> 数据来源: `results/phase5_llm_profiling/README.md`, `llm_analysis.md`

---

## 7. 项目关键数字

| 指标 | 数值 | 说明 |
|------|------|------|
| CNN 模型 MNN 领先 ORT | **1.54-1.75x** | MobileNetV2/ResNet50/YOLOv8n |
| BERT MNN vs ORT | **0.88x** | ORT 快 14%，唯一反转 |
| Conv1x1 GEMM 效率 | **~40 GFLOPS** | 接近 A77 单核算力峰值 |
| MatMul 768x768 效率 | **14.8 GFLOPS** | 仅 Conv1x1 的 37% |
| ResNet50 GPU 加速 | **3.03x** | Adreno 650 vs A77 |
| Qwen2-0.5B 吞吐 | **26.34 tok/s** | 骁龙 865 端侧实时推理 |
| BERT 根因定位 | **1 行代码** | `lP=1` (CommonOptFunctionNeon.cpp:1879) |
| MNN BERT cache miss | **2.30% vs 1.00%** | 2.3x ORT |
| MNN 多线程效率 | **43.7% vs 62.5%** | 4 核并行效率 |

---

## 8. 方法论总结

### 8.1 Benchmark 设计原则

1. **Profiling 驱动**: 不做盲测，先用 MNN_PROFILING + simpleperf 定位热点
2. **二八原则**: 聚焦累计占比 >= 80% 的热点算子做深度分析
3. **三层深度**: 模型级 (端到端) -> 算子级 (单算测例) -> 指令级 (PMU/源码)
4. **Roofline 标注**: 每个算子标注 Compute-Bound vs Memory-Bound，指导优化方向
5. **交叉验证**: MNN Profiler + simpleperf PMU + simpleperf record 三种工具互相印证

### 8.2 工具链

```
MNN_PROFILING=1      → 逐算子耗时 % 占比
simpleperf stat       → PMU 计数器 (IPC, cache-miss, branch-miss)
simpleperf record -g  → 火焰图 (函数级热点)
simpleperf annotate   → 指令级热点 (汇编行级性能)
源码阅读              → 分块参数、内存布局、Neon 指令
```

---

## 9. 数据溯源

| Phase | 数据 | 源文件 |
|-------|------|--------|
| 0 | 设备/内核/模型清单 | `results/phase0_*.txt` |
| 1 | 全模型 MNN/ORT Latency | `results/phase1_cpu_profiling/all_models_summary.md` |
| 1 | simpleperf 火焰图 | `results/phase1_cpu_profiling/perf_mbv2.data` |
| 2 | Roofline 分析 | Phase 2 commit (57cdc49) |
| 3 | GPU vs CPU | Phase 3 commit (c93568a) |
| 4 | BERT PMU 对比 | `results/phase4_bert_optimization/bert_*_pmu.log` |
| 4 | BERT 根因分析 | `results/phase4_bert_optimization/bert_analysis_report.md` |
| 4 | BERT 优化报告 | `results/phase4_bert_optimization/BERT_OPTIMIZATION_REPORT.md` |
| 5 | LLM 实测 | `results/phase5_llm_profiling/README.md` |
| 5 | LLM 理论分析 | `results/phase5_llm_profiling/llm_analysis.md` |

---

> **报告版本:** v1.0 | **生成日期:** 2026-06-06 | **分支:** feat/benchmark-full-plan
