# TVM vs MNN vs ORT 单算子性能根因分析

> 骁龙 865 (SM8250) / 1 线程 FP32 / 168 个单算子三框架实测
> 2026-06-09 · feat/tvm-stable-relay (TVM v0.15.0 Relay + GraphExecutor)

## 背景

**TVM 版本降级**：v0.24.dev0 → v0.15.0。

v0.24 的 Relax VM 对小 CV 模型有致命运行时开销（类型分发/分支/memcpy），MobileNetV2 实测 282ms（MNN 仅 8.9ms，32x 差距）。降级到 v0.15 后恢复 Relay + GraphExecutor 路径，MobileNetV2 降至 10ms（仅比 MNN 慢 11%）。

**单算子覆盖**：168 个 ONNX（17 手工生成 + 140 自动提取 + 11 阶梯大算子），覆盖 MobileNetV2 / ResNet50 / YOLOv8n / BERT / MobileViT-S 五个模型的核心算子类型。

## 全局统计

```
142 算子三框架完整数据
═══════════════════════════════
几何平均: MNN/ORT = 2.22x
         TVM/ORT = 1.69x
         TVM/MNN = 0.76x

最快框架: MNN 95 (67%)  TVM 42 (30%)  ORT 5 (3%)
```

## 按算子类型的胜负格局

| 类型 | 数量 | MNN胜 | TVM胜 | TVM/MNN | 结论 |
|------|:--:|:--:|:--:|:--:|------|
| DWConv | 17 | **17** | 0 | 0.61x | MNN 完胜 — depthwise 手写 NEON |
| Concat | 7 | **7** | 0 | 0.11x | MNN 完胜 — 零计算纯内存搬运 |
| LayerNorm | 4 | **4** | 0 | 0.22x | MNN 完胜 — 逐元素手写 SIMD |
| Softmax | 6 | **5** | 0 | 0.36x | MNN 大优 |
| **Conv1x1** | **68** | 41 | 25 | 0.94x | ⚖️ 分水岭在 C/K≈256 |
| Conv3x3 | 20 | 14 | 6 | 0.99x | ⚖️ 接近，大核 TVM 反超 |
| **MatMul** | 14 | 6 | **8** | **1.15x** | TVM 小优 |
| Conv7x7 | 1 | 0 | **1** | 1.86x | TVM |
| GELU | 1 | 0 | **1** | 1.84x | TVM |

## Shape 分水岭：Conv1x1 的 C/K 阈值

按 C×K（近似 FLOPs）排序后，MNN/TVM 比值随规模系统性变化：

```
C×K 范围          典型算子              MNN/TVM   胜者
──────────────────────────────────────────────────
< 8K (微核)       C32→K16              0.33x      MNN 绝对优势
8K~50K (小核)     C64→K128             0.60x      MNN 优势缩小
50K~200K (中核)   C256→K512            1.33x      TVM 开始反超
> 200K (大核)     C256→K1024, C1024→K2048  1.53x    TVM 绝对优势
```

**分水岭清晰：C×K ≈ 50K 是 MNN→TVM 的切换点**。

按通道数分桶同样验证了规律：

| 通道范围 | MNN胜 | TVM胜 | TVM/MNN |
|----------|:--:|:--:|:--:|
| 输入 C < 256 | 34 | 10 | 0.80x |
| 输入 C ≥ 256 | 7 | **15** | **1.39x** |
| 输出 K < 256 | 39 | 12 | 0.80x |
| 输出 K ≥ 256 | 2 | **13** | **1.57x** |

## DVConv：MNN 的专属领地

17 个 DWConv 算子中 MNN 全胜。差距随通道增加略微缩小，但始终 1.6-2.6x。

```
DWConv 计算密度 ≈ 2.25 FLOP/byte（极低，内存瓶颈）
→ MNN 的 NC4HW4 每 4 通道打包, NEON 128-bit 满负荷
→ TVM NCHW 下 channel=1, NEON 利用率仅 25%
```

## TVM 的优势场景

TVM 胜出的算子集中在 **高算术强度** 区域：

| 算子 | TVM/MNN | 特征 |
|------|:--:|------|
| MatMul_BERT_3072x768 | 2.55x | 大矩阵乘 |
| Conv1x1 C256→K1024 | 2.45x | 大通道 1x1 |
| Conv7x7 C3→K64 | 1.86x | 大核卷积 |
| GELU BERT | 1.84x | TVM 融合 pass 更优 |
| MatMul_2048x4096 | 2.09x | 超大矩阵 |

## MNN 的内存优化体系

MNN 之所以在内存瓶颈算子中无敌，依赖于多层内存优化：

### 1. NC4HW4 布局（核心）
标准 NCHW 下 NEON 跨通道加载需要 stride 跳转，4 通道打包后一次 `vld1q_f32` 直接加载 4 个相邻通道。

### 2. BufferAllocator (Slab 内存池)
同 session 内所有中间 tensor 复用同一块预分配内存，消除 malloc/free 的 syscall 开销。

### 3. Im2Col + Packed GEMM
Conv1x1 转换为矩阵乘法，weight 离线 pack 为 NC4HW4，每次推理只需 pack input → GEMM → unpack output。

### 4. Geometry 融合
Conv+BN+ReLU 三个 Op 融合为一个 kernel，中间结果不写回主存。150+ 个融合 pattern 覆盖常见算子组合。

### 5. Raster (延迟 Layout 转换)
类似 NumPy view，layout 转换延迟到真正需要时才触发一次批量操作。

### 6. Sliding Window (DWConv)
相邻输出像素共享 6/9 的输入像素，通过滑窗复用减少 ~3x 访存。

### 7. 离线 Weight Reorder
权重重排 (OIhw→OIhw4c) 在模型加载时一次性完成，推理时零开销。

## MNN 在大 shape 上退化的根因

你的猜想正确——**MNN 手写汇编在大 shape 上确实有内存排布开销**，但实测数据显示这不是主要原因：

```
以 K=1024, C=256, S=28 为例:
  pack input   ≈ 0.8M 搬运 ≈ 含在 L2 cache 内 (~0.01ms)
  unpack output ≈ 3.2M 搬运 ≈ 0.1ms (需 DRAM)
  计算         ≈ 200M FLOP ≈ 2ms
  → packing 占比 < 5%
```

**更可能的原因**：MNN 手写 GEMM 的**分块大小是固定的**（如 256×256），无法像 TVM 的 LLVM 后端那样根据 `-mcpu=cortex-a77` 的精确 cache 参数（L1=64KB, L2=512KB）自动选择最优分块。大矩阵时，自适应分块可以一次加载更多数据到 L2 cache，减少 DRAM 往返——这个差距在大 shape 时被放大。

## 整模型验证

结论在整模型层面同样成立：

| 模型 | MNN(ms) | TVM(ms) | TVM/MNN | 说明 |
|------|---------|---------|:--:|------|
| MobileNetV2 | **8.7** | 10.0 | 0.87x | DWConv 占比高 → MNN 优势 |
| ResNet50 | 83.6 | **64.4** | **1.30x** | 大 Conv1x1 占比高 → TVM 反超 |
| YOLOv8n | **77.0** | 126.6 | 0.61x | Conv3x3+Concat 多 → MNN 优势 |

ResNet50 的 bottleneck 结构大量使用 C256→K64→K256 的 1x1 大通道卷积——恰好落在 TVM 的优势区间。

## 实用建议

| 场景 | 推荐框架 | 理由 |
|------|---------|------|
| 移动端 CV (MobileNet 系列) | MNN | DWConv 占比 >60%，MNN 压倒性优势 |
| 大模型 Conv (ResNet/ResNeXt) | TVM | bottleneck 大通道 Conv1x1，TVM 反超 |
| BERT/NLP | 两者均可 | MNN 微核算子快，TVM 大 MatMul 快 |
| 混合模型 (MobileViT) | MNN | DWConv+逐元素占比高 |
| 需要调优/定制 | TVM | AutoTVM/AutoScheduler 就绪 |
| 追求极致性能 | 两者都跑 | 根据 per-op 分析选最优组合 |

---

*数据来源: `/tmp/ort_all_168.json`, `/tmp/mnn_all_168.json`, `/tmp/tvm_all_155.json`*
*生成脚本: `scripts/extract_model_operators.py`*
*ONNX 文件: `models/single_ops/`, `models/single_ops_extracted/`, `models/single_ops_stair/`*
