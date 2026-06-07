# 骁龙 865 LLM 推理性能分析

## 硬件约束（来源：Phase 2 Roofline 实测）

| 参数 | 值 | 来源 | 对 LLM 推理的影响 |
|------|-----|------|-------------------|
| 内存带宽 | ~34 GB/s (LPDDR5) | Phase 2 roofline 实测 | **Decode 阶段瓶颈** — KV Cache 扫描受限于此 |
| CPU 峰值 FP32 算力 | ~25 GFLOPS (A77 单核 NEON) | Phase 2 roofline 实测 | Prefill 阶段可利用多核扩展 |
| CPU 理论 FP32 算力 | ~90.8 GFLOPS (4×A77 @ 2.42GHz, NEON) | ARM 架构手册 | 多线程 Prefill 理论上限 |
| L2 Cache | 4×256KB (A77) + 4×128KB (A55) | 骁龙 865 规格 | 小 batch Prefill 部分数据 L2 命中 |
| L3 Cache | 2MB (shared) | 骁龙 865 规格 | KV Cache 滑窗时可减少 DRAM 访问 |
| Roofline 拐点 | 0.74 FLOP/Byte | Phase 2 实测计算 | 算术强度 < 0.74 的操作是 Memory-Bound |

## Prefill vs Decode 计算特征

### Prefill 阶段 (Prompt Processing)

| 属性 | 特征 |
|------|------|
| 计算模式 | MatMul 密集 (QKV Projection, FFN) |
| 瓶颈类型 | **Compute-Bound** (高算术强度) |
| 并行度 | 高度并行 — prompt 内所有 token 可同时处理 |
| 算力利用率 | 60-80% (大矩阵 MatMul 可高效利用 SIMD) |
| 内存访问 | 权重复用率高 — 每个 token 共享同一份权重 |

### Decode 阶段 (逐 Token 生成)

| 属性 | 特征 |
|------|------|
| 计算模式 | GEMV (矩阵-向量乘法) — KV Cache 线性扫描 |
| 瓶颈类型 | **Memory-Bound** (极低算术强度 ~0.01 FLOP/Byte) |
| 并行度 | 串行 — 每个 token 依赖前一个 token |
| 算力利用率 | <5% (cpu 大部分时间在等待内存) |
| 内存访问 | 每个 token 需要扫描完整 KV Cache |

## 以 Qwen2-0.5B 为例的理论估算

> **模型规格** (Qwen2-0.5B): 24 layers, hidden_dim=896, intermediate_dim=4864, num_heads=14, head_dim=64, vocab_size=151936

### 参数量和内存占用

| 组件 | 参数数量 | FP16 大小 | Q4_K_M 大小 (实测) |
|------|---------|----------|------------------|
| QKV Weight (per layer) | 3 × 896 × 896 = 2.4M | 4.8 MB | ~1.4 MB |
| FFN Up+Gate (per layer) | 2 × 896 × 4864 = 8.7M | 17.4 MB | ~5.1 MB |
| FFN Down (per layer) | 896 × 4864 = 4.4M | 8.7 MB | ~2.5 MB |
| 其他 (Embed/LN/Output) | ~25M | 50 MB | ~14 MB |
| **总计** | **~494M** | **~988 MB** | **~380 MB (Q4_K_M)** |

### Prefill 计算量 (seq_len=512)

**单层计算量:**

| 操作 | 公式 | FLOPs |
|------|------|-------|
| Q 投影 | 2 × 512 × 896 × 896 | 822 MFLOPS |
| K 投影 | 2 × 512 × 896 × 896 | 822 MFLOPS |
| V 投影 | 2 × 512 × 896 × 896 | 822 MFLOPS |
| Attention Score (QK^T) | 2 × 14 × 512 × 512 × 64 | 470 MFLOPS |
| Attention Output (AV) | 2 × 14 × 512 × 512 × 64 | 470 MFLOPS |
| Output Projection | 2 × 512 × 896 × 896 | 822 MFLOPS |
| FFN Gate | 2 × 512 × 896 × 4864 | 4,461 MFLOPS |
| FFN Up | 2 × 512 × 896 × 4864 | 4,461 MFLOPS |
| FFN Down | 2 × 512 × 4864 × 896 | 4,461 MFLOPS |
| **单层总计** | | **~17.6 GFLOPS** |

**24层总计:** 24 × 17.6 = **~422 GFLOPS**

理论 Prefill 时间估算:
- FP32 (多核): 422 / 90.8 = **4.65s** (100% util, 4-core A77)
- Q4_K_M (多核, 考虑解量化开销): **6-10s** (50-80% util)
- Q4_K_M (双核, 考虑热降频): **10-15s**

### Decode 计算量 (per token, seq_len=512)

**单层计算量:**

| 操作 | 公式 | FLOPs |
|------|------|-------|
| QKV 投影 (batch=1) | 3 × 2 × 1 × 896 × 896 | 4.8 MFLOPS |
| Attention (K+V 扫描) | 2 × 14 × 1 × 512 × 64 × 2 | 1.8 MFLOPS |
| Output Projection | 2 × 1 × 896 × 896 | 1.6 MFLOPS |
| FFN | 3 × 2 × 1 × 896 × 4864 | 26.1 MFLOPS |
| **单层总计** | | **~34.3 MFLOPS** |

**24层总计 (计算):** 24 × 34.3 = **~823 MFLOPS** (极小!)
**24层总计 (KV Cache 数据移动):** 2 × 24 × 896 × 512 × 2 bytes = **~44 MB/token**

- 带宽限制理论下限: 44 MB / 34 GB/s = **1.3 ms/token** (纯内存拷贝, 无计算)
- 实测估计 (单核): **8-20 ms/token** (内存延迟 + 计算 + 调度开销)
- 实测估计 (4核): **5-12 ms/token** (多线程可并行 KV Cache 分段扫描)

### 端到端推理估算 (1024 token 输出)

| 阶段 | 计算量 | 瓶颈 | 估算时间 |
|------|--------|------|---------|
| 模型加载 | 380MB 读入 | IO/内存 | 2-5s |
| Prefill (prompt=64 tokens) | ~53 GFLOPS | Compute | 0.5-2s |
| Decode (1024 tokens) | ~843 GFLOPS + 45 GB 数据移动 | **Memory** | 10-20s |
| **总计** | | | **15-27s** |

## 优化建议

### 短期可落地（框架侧已完成）

| 优化 | 实现状态 | 效果 |
|------|---------|------|
| ✅ Flash Attention | `flash_attn_type = ENABLED` | 避免完整 QK^T 矩阵写回内存, 降低 Prefill 内存占用 |
| ✅ Q4_K_M 量化 | 模型已量化 | 权重从 988MB → 380MB, 带宽压力降低 62% |
| ✅ ARM NEON SIMD | llama.cpp 自动启用 | 利用 A77 NEON 128-bit SIMD |
| ✅ 多线程 | `n_threads = 4` | Prefill 阶段接近线性加速 |

### 中期（建议实施）

| 优化 | 预期收益 | 实施难度 |
|------|---------|---------|
| KV Cache INT8 量化 | Decode 内存流量减半 (44→22 MB/token), 提速 30-50% | 中 (需修改 llama.cpp 配置) |
| QKV 合并 MatMul | 减少 kernel launch 开销, Prefill 提速 5-10% | 低 (llama.cpp 可能已支持) |
| 预热 KV Cache (Prompt Cache) | 相同 system prompt 时跳过 Prefill | 中 |
| Continuous Batching | 多请求共享 Prefill 计算 | 高 (需要服务层) |

### 长期（异构计算）

- **CPU + GPU 混合**: Prefill (计算密集) → GPU Adreno 650, Decode (访存密集) → CPU
- **挑战**: 跨设备 KV Cache 同步开销可能抵消收益
- **骁龙 865 限制**: CPU 和 GPU 共享同一 LPDDR5 控制器, 带宽瓶颈一致

## 模型推荐等级

| 模型规模 | 参数量 | Q4 大小 | 骁龙 865 适用场景 | 期望速度 |
|---------|--------|---------|-----------------|---------|
| Qwen2-0.5B | 0.5B | 380MB | **适合** — 实时对话 | 5-15 ms/token |
| Qwen2-1.5B | 1.5B | ~1GB | 可运行 — 需 4GB+ RAM | 15-30 ms/token |
| Qwen2-7B | 7B | ~4.2GB | 不适合 — 超过设备 RAM | 30-100 ms/token |

> **结论**: 骁龙 865 + 8GB RAM 的配置最适合 0.5B-3B 参数规模的量化模型。
> Qwen2-0.5B Q4_K_M 是当前配置下的最佳选择。
