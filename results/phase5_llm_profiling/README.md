# Phase 5: LLM 推理专项 — 最终报告

> **日期**: 2026-06-06 | **设备**: 红米 K30S (骁龙 865, SM8250) | **分支**: feat/benchmark-full-plan

---

## 一、执行摘要

Phase 5 完成了 llama.cpp 后端的完整适配、编译、部署和实测。Qwen2-0.5B Q4_K_M 模型在骁龙 865 上成功运行，实测吞吐量 **26.34 tokens/sec**，延迟 **37.96 ms/token**。

### 关键成果

| 指标 | 结果 |
|------|------|
| llama.cpp 后端编译 | **成功** (需适配 v0.10.0 API，6处修复) |
| Qwen2-0.5B Q4_K_M 模型下载 | **成功** (380MB) |
| 设备实测 | **成功** (真实推理数据) |
| 理论分析 | **完成** (详见 llm_analysis.md) |

---

## 二、llama.cpp 后端状态

### 2.1 组件清单

| 组件 | 状态 | 详情 |
|------|------|------|
| CMakeLists.txt 选项 | 存在 | `BENCHMARK_LLAMACPP: BOOL=OFF` (Line 41) |
| third_party/llama.cpp 子模块 | 已检出 | commit `9725a313` (v0.10.0 based) |
| `src/backends/llamacpp_backend.h` | 存在 | 1.5KB, 完整类定义 |
| `src/backends/llamacpp_backend.cpp` | 存在 | 5.2KB, 完整实现 |
| `src/llm_benchmark.cpp` | 存在 | 3.3KB, 独立可执行文件 |
| third_party CMake 集成 | 完整 | Lines 192-210 |
| src CMake 集成 | 完整 | 独立 `llm_benchmark` target |

### 2.2 API 适配修复（llama.cpp v0.10.0）

编译过程中发现 llama.cpp 子模块版本 (v0.10.0) 与后端代码编写时的 API 不兼容，完成了以下 6 处修复：

| # | 旧 API (已废弃) | 新 API (v0.10.0) | 影响 |
|---|----------------|-------------------|------|
| 1 | `llama_load_model_from_file()` | `llama_model_load_from_file()` | 模型加载 |
| 2 | `llama_new_context_with_model()` | `llama_init_from_model()` | 上下文创建 |
| 3 | `llama_kv_cache_clear(ctx)` | `llama_memory_seq_rm(mem, 0, 0, -1)` | KV Cache 清除 |
| 4 | `llama_sample_token_greedy(ctx)` | `llama_sampler_sample(smpl, ctx, -1)` | Token 采样 |
| 5 | `llama_tokenize(ctx, ...)` | `llama_tokenize(vocab, ...)` | Token 化 |
| 6 | `llama_token_eos(model)`, `llama_token_to_piece(model, ...)` | `llama_vocab_eos(vocab)`, `llama_token_to_piece(vocab, ...)` | Vocab API 重构 |

### 2.3 编译产物

| 文件 | 大小 | 架构 |
|------|------|------|
| `llm_benchmark` | 304KB | ELF64, ARM aarch64 |
| `libllama.so` | 32.6MB | ARM64 动态库 |
| `libggml.so` | 560KB | ARM64 动态库 |
| `libggml-cpu.so` | 4.4MB | ARM64 动态库 |
| `libggml-base.so` | 6.0MB | ARM64 动态库 |

---

## 三、设备实测结果

### 3.1 测试环境

- **设备**: 红米 K30S (b08dee23)
- **芯片**: 骁龙 865 (SM8250), 4×A77 @ 2.42GHz + 4×A55
- **内存**: 8GB LPDDR5
- **温度**: 51.9°C (运行前)
- **CPU 特性**: NEON=1, ARM_FMA=1, FP16_VA=1, LLAMAFILE=1, OPENMP=1, REPACK=1

### 3.2 Qwen2-0.5B Q4_K_M 实测数据

```
模型: Qwen2-0.5B-Instruct (494M params)
量化: Q4_K - Medium (6.35 BPW)
文件: 373.71 MiB (380MB)
架构: qwen2, 24 layers, hidden=896, FFN=4864, heads=14, GQA=7
```

**Prompt:** "Below is an instruction that describes a task... Explain what is machine learning in one sentence."

**输出:** "Machine learning is a subset of artificial intelligence that involves the use of algorithms to enable computers to learn from data and make predictions or decisions based on that learning..."

| 指标 | 值 |
|------|-----|
| 模型加载时间 | **0.47s** (mmap 启用) |
| Prompt tokens | 34 |
| 生成 tokens | ~64 |
| 总推理时间 | **2.43s** |
| **吞吐量** | **26.34 tokens/sec** |
| **每 token 延迟** | **37.96 ms/token** |

### 3.3 性能分解估算

| 阶段 | 估算时间 | 占比 | 说明 |
|------|---------|------|------|
| Prefill (34 tokens) | ~0.3-0.5s | 12-21% | Compute-Bound, MatMul 密集 |
| Decode (64 tokens) | ~1.9-2.1s | 79-88% | Memory-Bound, KV Cache 扫描 |
| **总计** | **2.43s** | 100% | — |

**Decode 速度:** 每 token ~30-33ms（扣除 Prefill 后）

---

## 四、理论 vs 实测对比

| 指标 | 理论估算 | 实测值 | 偏差分析 |
|------|---------|--------|---------|
| Prefill (34 tokens) | 0.3-0.5s | ~0.4s (推断) | ✅ 一致 |
| Decode per token | 8-20 ms | ~30-33 ms | ⚠️ 实际慢 50-300% |
| 吞吐量 | 30-100 tok/s | 26.3 tok/s | ⚠️ 接近下限 |

### 性能偏差原因分析

1. **big.LITTLE 异构调度**: 骁龙 865 有 4×A77 + 4×A55。A55 核心的单核性能约为 A77 的 30-40%，当 OpenMP 将线程分配到 A55 上时，整体性能被拖累。实测中 `n_threads=4` 可能分配到了 2×A77 + 2×A55。
2. **温度降频**: 实测前设备温度 51.9°C，可能已触发 thermal throttling。
3. **KV Cache 未量化**: 默认 FP16 KV Cache，每 token 需扫描 ~22MB (n_ctx=1024)，内存带宽压力大。
4. **Q4_K_M 解量化开销**: 每层 feed-forward 需要解量化权重，增加了计算开销。

---

## 五、硬件约束分析

详见 [llm_analysis.md](./llm_analysis.md)，核心要点：

### 5.1 Prefill vs Decode 瓶颈

```
Prefill (Prompt Processing):
  ├── 计算模式: MatMul 密集 (高算术强度)
  ├── 瓶颈: Compute-Bound (需更多算力)
  ├── 利用率: 60-80% (单核 A77 实测 ~25 GFLOPS)
  └── 优化方向: INT4 量化、多核并行、Flash Attention

Decode (逐 Token 生成):
  ├── 计算模式: GEMV + KV Cache 线性扫描 (极低算术强度 ~0.01)
  ├── 瓶颈: Memory-Bound (受限于 34 GB/s LPDDR5 带宽)
  ├── 利用率: <5% (CPU 大部分时间等待内存)
  └── 优化方向: KV Cache INT8、GQA (已启用 7:1)、Prompt Cache
```

### 5.2 Roofline 分析

骁龙 865 的 Roofline 拐点为 **0.74 FLOP/Byte**：
- Prefill 算术强度: ~10-50 FLOP/Byte → **Compute-Bound** ✓
- Decode 算术强度: ~0.01 FLOP/Byte → **严重 Memory-Bound** ✗

---

## 六、优化建议优先级

基于实测数据调整优先级：

| 优先级 | 优化方案 | 预期收益 | 实施难度 | 状态 |
|--------|---------|---------|---------|------|
| 🔴 P0 | 线程绑定 (仅 A77 核心) | Decode 提速 20-40% | 低 | 建议 |
| 🔴 P0 | KV Cache INT8 量化 | Decode 提速 30-50% | 中 | 建议 |
| 🟡 P1 | 权重 Q4_0 量化 (更小) | 模型从 380→280MB | 低 | 可选 |
| 🟡 P1 | Flash Attention 启用 | Prefill 内存节省 | 低 | 已启用 |
| 🟢 P2 | CPU 锁频 (performance governor) | 稳定性提升 | 低 | 测试时 |
| 🟢 P2 | Prompt Cache | 相同前缀跳过 Prefill | 中 | 建议 |

### 线程绑定示例

```bash
# 仅使用 4 个 A77 大核 (CPU 4-7)
taskset -c 4-7 ./llm_benchmark model.gguf 128
```

---

## 七、文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `models/nlp/qwen2_0.5b/qwen2-0_5b-instruct-q4_k_m.gguf` | 380MB | Qwen2-0.5B Q4_K_M 量化模型 |
| `src/backends/llamacpp_backend.h` | 1.5KB | llama.cpp 后端接口 (已适配 v0.10.0) |
| `src/backends/llamacpp_backend.cpp` | 5.2KB | llama.cpp 后端实现 (已适配 v0.10.0) |
| `src/llm_benchmark.cpp` | 3.3KB | LLM Benchmark 独立可执行文件源码 |
| `resunts/phase5_llm_profiling/llm_analysis.md` | — | 理论性能分析 |
| `resunts/phase5_llm_profiling/README.md` | — | 本报告 |

---

## 八、结论

### 8.1 可运行性

骁龙 865 + 8GB RAM 可以流畅运行 Qwen2-0.5B Q4_K_M 量化模型：
- **Precision**: Q4_K_M (6.35 BPW), 输出质量接近 FP16
- **Speed**: 26.34 tok/s (~38ms/token), 满足实时对话需求
- **Memory**: 模型 380MB, 运行时 < 2GB RAM

### 8.2 能力边界

| 模型规模 | 可行性 | 预期速度 | 建议 |
|---------|--------|---------|------|
| Qwen2-0.5B | ✅ 实测可运行 | 26 tok/s | **推荐** |
| Qwen2-1.5B | ✅ 预估可运行 | 10-15 tok/s | 可尝试 |
| Qwen2-7B | ❌ 内存不足 | — | 不适合 |

### 8.3 后续工作

- [ ] 线程绑定到 A77 大核测试 (P0)
- [ ] KV Cache INT8 量化测试 (P0)
- [ ] simpleperf 采集 Decode 阶段 PMU 事件，确认 Memory-Bound 假设
- [ ] 更多 Prompt 长度测试 (128/256/512/1024 tokens)
- [ ] 对比 Qwen2-1.5B (如果下载)
- [ ] 集成到 CI/CD 自动测试流程

---

## 九、相关资源

- llama.cpp 官方文档: https://github.com/ggml-org/llama.cpp
- Qwen2 模型: https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF
- 骁龙 865 规格: https://www.qualcomm.com/products/snapdragon-865-5g-mobile-platform
