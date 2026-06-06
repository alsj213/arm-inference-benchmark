# BERT MNN vs ORT 性能 Gap 根因分析报告

> 日期: 2026-06-06 | 设备: Redmi K30S (骁龙 865 / SM8250) | 分支: `feat/benchmark-full-plan`
> 代码版本: `c93568a feat: Phase 3 GPU OpenCL 后端启用与 CPU vs GPU 全模型对比`

---

## 1. 现象确认 —— 差距可复现且稳定

### 1.1 端到端三次测试结果

| Run | MNN Mean | MNN P50 | MNN Std | ORT Mean | ORT P50 | ORT Std | Gap |
|-----|----------|---------|---------|----------|---------|---------|-----|
| 1   | 787.61ms | 752.07ms | 75.54ms | 664.79ms | 664.73ms | 0.66ms | ORT 快 18.5% |
| 2   | 778.30ms | 756.78ms | 52.28ms | 888.79ms\* | 884.06ms | 26.63ms | (热降频) |
| 3 (PMU) | 757.19ms | 756.24ms | 2.41ms | 672.20ms | 671.78ms | 1.93ms | ORT 快 12.6% |

\* Run 2 ORT 受热降频影响，设备温度达 51.9°C。

### 1.2 结论

- **冷启动最优数据**: MNN 757ms vs ORT 665ms，**ORT 快约 14%**
- **差距稳定可复现**: Run 1 和 Run 3 显示 gap 在 12.6%~18.5% 范围
- **MNN 方差显著更大**: Std 2.41~75.54ms vs ORT 0.66~1.93ms，MNN 性能不够稳定
- **精度无问题**: Cosine Similarity = 0.990439，MNN 与 ORT 输出高度一致

---

## 2. PMU 计数器对比 —— 关键证据

| 指标 | MNN BERT | ORT BERT | 分析 |
|------|----------|----------|------|
| **cpu-cycles** | 21.41B | 19.16B | MNN 多用 11.8% 周期 |
| **instructions** | 39.37B | 39.47B | **几乎完全相同** (!) |
| **IPC** | 1.838 | 2.060 | MNN IPC 低 12.1% |
| **cache-misses** | 200.45M | 102.01M | MNN 多 ~2x cache miss |
| **cache-references** | 8.72B | 10.20B | ORT 更多 cache 访问 |
| **cache miss rate** | 2.30% | 1.00% | MNN 缓存缺失率 2.3 倍 |
| **branch-misses** | 37.30M | 3.28M | MNN 分支预测失败多 **11.4x** (!) |

### 2.1 核心发现

**指令数几乎相同（39.37B vs 39.47B），但 MNN 花费了更多周期。** 这说明：

1. **MNN 和 ORT 在算法层面做了同样多的工作** — 它们执行了几乎相同的计算量
2. **MNN 的微架构效率更低** — 更多周期浪费在缓存缺失和分支预测失败上
3. **根本原因是 GEMM 内核的内存访问模式** — 而非算法选择问题

### 2.2 GFLOPS 估算

BERT-base (12层, hidden=768, seq_len=128) 总 FLOPs ≈ 11.2 GFLOPs：

- MNN: 11.2 / 0.757s = **14.8 GFLOPS**
- ORT: 11.2 / 0.672s = **16.7 GFLOPS**

这与 Phase 2 的单算子 MatMul 768×768 基准测试一致（MNN 12-17 GFLOPS），远低于 Conv1x1 的 ~40 GFLOPS。

---

## 3. 符号级热点分析 —— simpleperf Top Down

### 3.1 MNN BERT 热点排行

| Rank | 符号/函数 | 开销 | 类别 |
|------|----------|------|------|
| 1 | **LoopL2** (MNNPackedMatMul 内循环) | **46.87%** | GEMM 微内核 |
| 2 | L16Loop (Conv 相关) | 5.62% | 卷积循环 |
| 3 | `__ieee754_expf` | 4.57% | GELU 激活 (exp) |
| 4 | LoopL (MatMul 外循环) | 4.21% | GEMM 外循环 |
| 5 | `erff` | 4.06% | GELU 激活 (erf) |
| 6 | MNNPackTranspose | 1.49% | B 矩阵 Pack |
| 7 | `__memcpy` | 1.40% | 内存拷贝 |
| 8 | MNNUnpackTranspose | 1.03% | 输出解包/转置 |
| — | **GEMM 总计** | **~55%** | 包括 LoopL2 + LoopL + LoopH |
| — | **GELU 总计** | **~9%** | expf + erff |
| — | **Pack/Unpack** | **~4%** | MNNPackTranspose + MNNUnpackTranspose + memcpy |

### 3.2 ORT BERT 热点排行

ORT 符号表被 strip，但热点分布明显更均匀：顶级热点仅 12.18%，且均匀分布在上百个代码区域。说明 ORT 的 GEMM 实现更好地分摊了瓶颈。

### 3.3 结论

- **MNN GEMM 微内核 (LoopL2) 是单一最大瓶颈**，占 47% CPU 时间
- **GELU 激活函数 (exp+erf)** 占 9%，是第二热点
- **Pack/Unpack 开销** 约 4%，虽不算大但对整体性能有影响
- ORT 的热点分布更均匀，说明其 GEMM 内核有更好的内存层次利用

---

## 4. MNN GEMM 源码根因分析

### 4.1 分块参数

位置: `third_party/MNN/source/backend/cpu/arm/CommonOptFunctionNeon.cpp:1879-1887`

```c
void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 12;   // M 维度分块: 每块处理 12 行
    *lP = 1;    // K 维度分块: 不打包 K (逐元素处理!)
    *hP = 8;    // N 维度分块: 每块处理 8 列
}
```

**关键问题: `lP = 1` — K 维度不做打包！**

### 4.2 微内核设计 (MNNPackedMatMul.S)

位置: `third_party/MNN/source/backend/cpu/arm/arm64/MNNPackedMatMul.S`

**寄存器布局 (12x8 微内核):**
- `v0, v1, v2`: 加载 A 矩阵值 (12 个 float = 3 × float4)
- `v3, v4`: 加载 B 矩阵值 (8 个 float = 2 × float4)
- `v8-v31`: C 矩阵累加 (24 个 float4 = 96 个 float = 12×8)
- `v5, v6, v7`: Post 参数 (scale, bias, min, max)
- **总计: 29/32 个 NEON 寄存器被占用**

**内循环结构 (LoopL2, 行 86-150):**
```asm
LoopL2:
    // 每次迭代:
    ld1 {v3,v4}, [x2], #32     // 加载 B: 8 float (32 bytes)
    ld1 {v0,v1,v2}, [x15], #48  // 加载 A: 12 float (48 bytes)
    fmla v8..v31, v3/v4, v0/v1/v2.s[...]  // 96 次 FMA = 192 FLOPs
    // 再次迭代 (2x unrolled K=1)
    ld1 {v3,v4}, [x2], #32
    ld1 {v0,v1,v2}, [x15], #48
    fmla v8..v31, v3/v4, v0/v1/v2.s[...]
    sub x12, x12, #2           // K -= 2
    cmp x12, #2
    bge LoopL2
```

### 4.3 为什么 K=768 效率低 — 量化分析

#### 问题 1: K 维度未打包 (lP=1)

- lP=1 意味着**每个 K 步只加载 20 个 float (12A+8B)，执行 96 次 FMA**
- 计算/访存比: 192 FLOPs / (20 × 4 bytes) = **2.4 FLOPs/byte**
- 骁龙 865 的 A77 核内存带宽约 34 GB/s (理论)，单个大核可用的 L2 带宽约 50-80 GB/s
- 2.4 FLOPs/byte 对应 ~96 GFLOPS 的理论上限（假设 40 GB/s 可用带宽）
- 实测 14.8 GFLOPS = **仅达到理论值的 15.4%**

#### 问题 2: K=768 对齐分析

- K=768, lP=1 → K 步被完美处理（无剩余）
- 但内循环 2x unroll: 768/2 = 384 次循环迭代
- 每次迭代 2 次比较 + 2 次分支
- **每 K 步有 768 次分支指令**（因为 lP=1，没有 K 方向打包减少循环）

#### 问题 3: B 矩阵 Pack 后的 Cache 行为

- B 矩阵 768×768, 打包后: (768/8) × 768 × 8 = **96 × 768 × 8 floats = 2.36 MB**
- 骁龙 865 A77 L1 cache: 64KB, L2: 256KB per core (或 1MB shared)
- **打包后的 B 矩阵 (2.36MB) 远超 L2 cache 容量**
- 每次 M=12 的 tile 需要访问 B 的特定 N=2 tile（2 × 32 bytes = 64 bytes），大部分 B 不在 cache 中
- **Cache miss rate 2.30% vs ORT 1.00% 确认了这个猜测**

#### 问题 4: 与 Conv1x1 GEMM 的效率对比

MNN 的 Conv1x1 达到 ~40 GFLOPS，而 MatMul 仅 14.8 GFLOPS。虽然两者都是 GEMM，但关键区别：

| 特性 | Conv1x1 GEMM | MatMul GEMM |
|------|-------------|-------------|
| K 维度打包 | C4 格式 (K 以 4 为单位打包) | lP=1 (不打 K) |
| 内循环 K 步 | 4 (处理 4 个 K) | 1 (每次处理 1 个 K) |
| 分支/迭代 | 少 4 倍 | 多 4 倍 |
| A 矩阵格式 | NC4HW4 (channel packed by 4) | 原始列主格式 |
| B 矩阵 Pack | Conv kernel 预 pack (im2col) | MNNPackForMatMul_B |

**Conv1x1 的高效来自两个关键因素:**
1. **K=4 打包**: 每次从内存加载 4 倍的 K 数据，计算密度高 4 倍
2. **NC4HW4 格式**: A 矩阵的通道维度被 4 路打包，减少 gather 操作

而 MatMul 的 lP=1 意味着它无法享受这些优化。

### 4.4 B 矩阵 Pack 策略分析 (MNNPackForMatMul_B)

位置: `third_party/MNN/source/backend/cpu/arm/CommonOptFunctionNeon.cpp:1893`

```c
// output shape is (UP_DIV(h, 8), l, 8)
// 对于 B[768, 768]: packed = [96, 768, 8] = 2.36 MB
void MNNPackForMatMul_B(float* dest, const float* source, ...) {
    auto hP = h / 8;  // 96
    for (int y=0; y<hP; ++y) {
        for (int x=0; x<l; ++x) {
            for (int i=0; i<8; ++i) {
                dest[(y*8+i)*l + x] = source[x*h + y*8 + i];  // 列转行
            }
        }
    }
}
```

这个 Pack 将列主格式的 B 转换为行主格式，使 GEMM 内循环可以顺序访问 B。但代价是 **2.36MB 的内存占用**和**Cache 不友好**。

---

## 5. 根因总结

### 5.1 直接原因

1. **MNN MatMul 的 K 维度不打包 (lP=1)** — 导致计算/访存比极低（2.4 FLOPs/byte），陷入内存带宽瓶颈
2. **Pack 后的 B 矩阵 (2.36MB) 远大于 L2 cache** — 导致 2.3% cache miss rate (vs ORT 1.0%)
3. **循环分支过多** — lP=1 导致 K 内循环 768 次迭代，分支预测失败率为 ORT 的 11.4 倍

### 5.2 深层原因

MNN 的 MatMul 微内核是为**通用场景**设计的（任意 M×K×N），没有针对 BERT 典型形状（768×768, 768×3072）做特化优化。相比之下：
- MNN Conv1x1 针对 Conv 特性（C4 打包, im2col）做了深度优化
- ORT 可能使用了更高级的缓存分块策略或 K 维度打包

### 5.3 为什么 ORT 更快但指令数相同

ORT 的执行效率更高不是因为做了更少的计算（指令数相同），而是因为：
- **更好的缓存局部性**: cache miss rate 低 2.3 倍
- **更少的分支预测失败**: branch-misses 低 11.4 倍
- **更高的 IPC**: 2.06 vs 1.84

---

## 6. 优化建议

### 6.1 短期（代码微调，可立即尝试）

| 优先级 | 优化项 | 预期收益 | 难度 |
|--------|--------|---------|------|
| P0 | 将 **lP 从 1 改为 4 或 8**，增加 K 维度打包 | **20-30%** | 中 |
| P1 | 为 K=768 特化：4x unroll 替换 2x unroll | **5-10%** | 低 |
| P2 | B 矩阵 Pack 分块：不一次性 Pack 全部 B，按 tile 分批 | **5-10%** | 中 |

#### P0: K 维度打包 (最核心)

```c
// 当前
*lP = 1;  // 每个 K 步处理 1 个元素

// 优化后
*lP = 4;  // 每个 K 步处理 4 个元素（或 8，取决于寄存器）
```

这需要：
1. 修改 `MNNGetMatMulPackMode` 将 lP 改为 4
2. 修改 `MNNPackForMatMul_B` 支持 lP=4 的 B 矩阵打包
3. 修改 `MNNPackedMatMul.S` 内循环支持 lP=4 的数据布局

预期将计算/访存比从 2.4 FLOPs/byte 提升到 9.6 FLOPs/byte，GFLOPS 从 14.8 提升到 19-22。

#### P1: 循环展开优化

当前 2x unroll 对 K=768 需要 384 次循环 + 768 次分支。改为 4x 或 8x unroll 可减少分支预测失败。

### 6.2 中期（指令级优化）

- [ ] 检查是否启用了 **SDOT 指令** (ARMv8.2-A, 骁龙 865 支持)：可加速 int8 GEMM
- [ ] 考虑 **FP16 路径**：对于 BERT 推理，FP16 精度足够且计算吞吐翻倍
- [ ] **Softmax + LayerNorm 融合**：减少内存往返

### 6.3 长期（算子融合）

- [ ] **MatMul + Bias + GELU 融合**: 将 GELU (exp+erf, 占 9%) 融入 GEMM 的 post-processing 步骤，避免单独的 kernel launch 和内存往返
- [ ] **Multi-Head Attention 融合**: QKV 投影合并为一个 MatMul
- [ ] **Flash Attention** 适配：减少 Attention 中间结果的显存占用

---

## 7. 数据溯源

| 数据 | 来源文件 | 来源命令 |
|------|---------|---------|
| 端到端延时 (Run 1) | `results/phase4_bert_optimization/bert_mnn_run1.log` | `adb shell benchmark_inference --model bert --backend mnn --runs 100` |
| PMU 计数器 | `results/phase4_bert_optimization/bert_mnn_pmu.log` | `simpleperf stat -e cpu-cycles,instructions,cache-misses,...` |
| 符号级热点 | `results/phase4_bert_optimization/perf_bert_mnn.data` | `simpleperf record -e cpu-cycles -g ...` | 
| GEMM 分块参数 | `third_party/MNN/source/backend/cpu/arm/CommonOptFunctionNeon.cpp:1879` | 源码阅读 |
| 微内核汇编 | `third_party/MNN/source/backend/cpu/arm/arm64/MNNPackedMatMul.S` | 源码阅读 |
| MatMul 调度 | `third_party/MNN/source/backend/cpu/CPUMatMul.cpp` | 源码阅读 |

---

## 8. 状态与下一步

**状态: DONE_WITH_CONCERNS** — 根因定位完成，优化方向明确，但未实际修改和验证代码。

核心发现：**MNN MatMul 的 K 维度不打包 (lP=1) 是导致 BERT 性能落后的根因**。将 lP 从 1 改为 4 是最有希望的优化方向，预期可减少 20-30% 的 MNN BERT 延时，使 MNN 达到或超越 ORT 的性能水平。

下一步（Phase 5）：如果时间允许，可以实际修改 `MNNGetMatMulPackMode` 的 lP 参数，编译并重新 benchmark 验证。这需要修改 MNN 源码并重新编译 libMNN.so。
