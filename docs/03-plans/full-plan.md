# 端侧推理引擎性能分析项目 — 完整计划

## 项目目标

以骁龙 865 (红米 K30S) 为硬件平台，MNN 为主要分析对象，建立从**模型级 Profiling → 算子级性能画像 → 瓶颈定位 → 算子优化闭环**的完整技术故事线，服务于推理引擎开发工程师（大模型方向）的面试。

### 面试叙事主线

> "我搭了一套完整的端侧推理性能分析体系——不是跑几个现成模型的 Latency，而是从 Profiling 数据出发，定位到框架内部的具体算子和执行路径，然后用 4 年的算子优化经验去改它。下面是我怎么一步步做的。"

---

## 硬件平台

| 设备 | 芯片 | CPU | GPU | DSP | ISA |
|------|------|-----|-----|-----|-----|
| 红米 K30S 至尊纪念版 | 骁龙 865 (SM8250) | 1×A77@2.84GHz + 3×A77@2.42GHz + 4×A55@1.8GHz | Adreno 650 | Hexagon 698 | ARMv8.2-A, FP16, SDOT |

### 后端覆盖

| 后端 | 当前状态 | 优先级 | 定位 |
|------|---------|--------|------|
| MNN CPU | ✅ 已跑通 | — | 🎯 主优化对象——手写极致优化路线 |
| MNN GPU (OpenCL) | ❌ 未启用 | 🔴 必做 | Adreno 650 支持 OpenCL 3.0 |
| ONNX Runtime CPU | ✅ 已跑通 | — | 📏 通用基线——跨平台可移植路线 |
| llama.cpp | ✅ 已跑通 | — | 🧠 LLM 基线——纯 CPU 大模型推理路线 |
| TVM | ⚠️ 可恢复 | 🟡 可加 | 🔧 编译器基线——自动代码生成路线 |
| QNN (高通 NPU) | ❌ | ❌ 不在本期 | 骁龙 865 无独立 NPU

### 多设备策略

**本期不扩展设备。** K30S 的 CPU + GPU 双后端已经够讲一个完整故事。面试时如果被问到 NPU/多设备，如实说"目前聚焦 CPU 和 GPU 的深度分析"即可。后续如果需要，首选骁龙 8 Gen 2（独立 NPU）做高低端对比。

---

## 模型选择（5 个，覆盖 5 种计算模式）

| 序号 | 模型 | 参数量 | 代表场景 | 核心计算模式 |
|------|------|--------|---------|------------|
| 1 | **MobileNetV2** | 3.5M | 轻量级端侧分类 | Conv1x1(GEMM) + DepthwiseConv + 逐元素 |
| 2 | **ResNet50** | 25M | 服务器级 CNN 分类 | Conv3×3(Winograd/大核) + GEMM(大矩阵) |
| 3 | **BERT** | 110M | Encoder Transformer | MatMul(768×768) + MatMul(768×3072) + LayerNorm + GELU |
| 4 | **Qwen2-0.5B** | 620M | Decoder LLM | KV Cache + RoPE + MatMul(变长) + INT4 量化 |
| 5 | **MobileViT-S** | 5.6M | CNN+Transformer 混合 | Conv + Attention 交替（端侧多模态前奏） |

### 为什么不多不少就这 5 个

```
MobileNetV2  → 覆盖 80% 端侧模型的计算模式变体
ResNet50     → 覆盖所有大型卷积网络的算子特征
BERT         → 覆盖 Encoder Transformer 的计算特征
Qwen-0.5B    → 覆盖 Decoder LLM + KV Cache + 自回归生成
MobileViT   → 覆盖 CNN+Attention 混合架构

不选的：
  YOLOv8n   → 和 ResNet50 + MobileNetV2 计算模式高度重叠
  Llama-1B  → 和 Qwen-0.5B 计算模式完全一致
  ViT       → 纯 Transformer，被 BERT + MobileViT 覆盖
```

---

## 框架选择（4 个，代表 4 种优化路线）

| 框架 | 定位 | 优化路线 | 为什么不可替代 |
|------|------|---------|--------------|
| **MNN** | 🎯 主优化对象 | 人手写 Neon 汇编 + NCHW4c 内存布局 | 端侧极致性能的代表，你的 4 年经验全在这上面 |
| **ONNX Runtime** | 📏 通用基线 | 跨平台 + 编译器/运行时分离 | 回答"MNN 为什么快"的参照物 |
| **llama.cpp** | 🧠 LLM 基线 | 纯 CPU + 激进量化 | 回答"MNN 在大模型推理上还有哪些差距"的参照物 |
| **TVM** | 🔧 编译器基线 | 自动代码生成 + AutoTVM 调优 | 回答"手写 vs 自动生成的边界在哪里"的参照物 |

### 为什么不是 5 个、不是 3 个

```
ncnn 不加：和 MNN 路线完全一致（端侧轻量级手写优化），不增加分析维度
TFLite 不加：和 ORT 路线重叠（通用跨平台），且 Android 生态已萎缩

MNN + ORT + llama.cpp + TVM
  = 手写极致 × 通用可移植 × LLM 专用 × 编译器自动生成
  = 四种不同的优化哲学，每一种都是你的面试素材
```

---

## Benchmark 设计方法论

### 原则一：Profiling 驱动，二八原则

```
不是人选择测哪些算子 → Profiling 数据决定哪些算子重要

跑每个模型的完整推理 Profiling（MNN Profiler + simpleperf 火焰图）
  → 导出每个算子的耗时占比
  → 累计占比 ≥80% 的算子（通常 3-5 个）→ 深度测试
  → 剩余算子 → 快速通道（功能验证 1-2 个测例）
```

### 原则二：对热点算子做三层覆盖

```
第一层：经典网络提取（15-20 个测例）
  从 5 个模型里直接抽取真实算子块，保证和实际场景对齐

第二层：形状分桶（10-15 个测例）
  沿"计算量"和"矩阵形状"两个维度取极端值，
  覆盖性能行为差异最大的 corner case

第三层：框架特化路径（5-8 个测例）
  Winograd、NCHW4c 对齐退化、多线程加速比、量化路径
```

### 原则三：覆盖计算模式，不覆盖参数组合

```
以 Conv 为例——不是穷举 Kernel/Stride/Padding/Dilation 的组合，
而是确保三条执行路径各有一个代表性测例：

  路径 A：Im2Col + GEMM（通用卷积，计算密集）
  路径 B：Winograd（3×3 s=1 d=1，减小乘法次数）
  路径 C：Depthwise（groups=C，访存密集）
```

---

## 分阶段执行计划

### Phase 0：环境就绪（0.5 天）

```
□ 确认设备连接正常（adb devices）
□ 确认 MNN Profiler 可用（MNN_PROFILING=1 能正常输出）
□ 确认 simpleperf 可用（能抓火焰图）
□ 确认所有 5 个模型文件齐全
□ 确认 GPU OpenCL 后端可加载（MNN 编译时已启用）
```

### Phase 1：CPU 全量 Profiling（2-3 天）

#### 1.1 跑 5 个模型的完整推理 Profiling

```
□ 每个模型用 MNN_PROFILING=1 导出逐算子耗时
□ 每个模型用 simpleperf 抓火焰图（交叉验证）
□ 记录：总耗时、每个算子的耗时和占比
```

#### 1.2 产出每个模型的热点算子排行

```
目标输出（预期）：

  MobileNetV2  → Conv1x1 55% + DWConv 25% + Add 8%
  ResNet50     → Conv3×3 35% + Conv1×1 30% + BN/Add 15%
  BERT         → MatMul(768×768) 38% + MatMul(768×3072) 30% + LayerNorm 10%
  Qwen-0.5B    → MatMul(变长) + Attention + RoPE（待实测）
  MobileViT-S  → Conv + MatMul 混合（待实测）
```

#### 1.3 确定热点算子清单

```
□ 标记累计占比 ≥80% 的算子 → 深度测试
□ 标记占比 <20% 但框架特有的（如 Winograd）→ 中等深度
□ 标记其余算子 → 快速通道
```

### Phase 2：CPU 热点算子深度 Benchmark（3-5 天）

#### 2.1 Conv1x1 / GEMM（第一热点）

```
测例设计（≈30 个）：

  经典网络提取（15 个）：
    MobileNetV2: 各 block 的 Conv1x1（小 M 到大 M）
    ResNet50: 降维/升维 Conv1x1（M=56²~7²）
    YOLOv8n: 各尺度 Conv1x1（补充不同分辨率）

  形状分桶（10 个）：
    M 维极端：M=49(7×7) / M=784(28×28) / M=3136(56×56)
    K 维极端：K=16 / K=256 / K=1024
    覆盖：访存密集端 → 平衡态 → 计算密集端

  特化路径（5 个）：
    NCHW4c 对齐退化（C%4≠0 的 case）
    多线程加速比（1/2/4/8 线程）
    FP32 vs FP16 对比
```

#### 2.2 DepthwiseConv（第二热点）

```
测例设计（≈10 个）：

  经典网络提取（5 个）：
    MobileNetV2 各 block 的 DWConv（3×3, C=16~960）

  形状分桶（3 个）：
    大通道（C=960）vs 小通道（C=16）
    大图（H=W=112）vs 小图（H=W=7）

  特化（2 个）：
    FP32 vs FP16 在 DWConv 上的表现（访存密集 vs 带宽）
```

#### 2.3 Conv3×3（ResNet50 核心）

```
测例设计（≈10 个）：

  网络提取（5 个）：
    ResNet50 各阶段的 Conv3×3（s=1 和 s=2）

  特化（5 个）：
    Winograd 路径（s=1 d=1）：不同通道数下的变换开销占比
    Im2Col 路径（s=2）：作为对比
```

#### 2.4 MatMul（BERT/LLM 核心）

```
测例设计（≈15 个）：

  Attention MatMul（方阵）：768×768、512×512、1024×1024
  FFN MatMul（长矩阵）：768×3072、3072×768
  KV Cache 场景：M 从小到大（1 → 512 → 2048 逐步增长）

  精度对比：FP32 vs FP16 vs INT8
```

#### 2.5 LayerNorm / Softmax / GELU（注意力配套设施）

```
测例设计（≈8 个）：
  LayerNorm：不同 hidden_size（512/768/1024）+ 不同 seq_len（1/128/512）
  Softmax：同上
  GELU：同上
```

### Phase 3：GPU 后端启用与对比（3-4 天）

#### 3.1 启用 MNN OpenCL 后端

```
□ 确认 libMNN_CL.so 或 MNN 已编译 OpenCL
□ 修改 run_benchmark_android.sh 支持 --backend mnn_gpu
□ 确保 GPU 推理结果精度和 CPU 对标
```

#### 3.2 CPU vs GPU 全模型对比

```
□ 5 个模型各跑 CPU 和 GPU 的端到端延迟
□ 对比项：初始化时间 / 推理延迟 / FPS / 内存峰值
□ 记录哪些模型 GPU 加速明显、哪些不明显
```

#### 3.3 CPU vs GPU 逐算子对比（关键）

```
□ 每个模型拆分到算子级，对比每个算子在 CPU 和 GPU 上的耗时
□ 核心发现预设：
  - 计算密集的 GEMM/Conv：GPU 有优势
  - 访存密集的 DWConv/LayerNorm/逐元素：GPU 可能没优势甚至更慢
  - 小算子 GPU 启动开销可能大于计算本身
  → 这就是你为什么理解异构调度的价值
```

#### 3.4 跨后端数据传输分析

```
□ 分析 CPU→GPU→CPU 的数据拷贝开销
□ 计算"GPU 计算加速 vs 传输开销"的净收益
□ 输出：哪些算子适合放 GPU、哪些适合留 CPU 的判断规则
```

### Phase 4：关键优化闭环（7-10 天）

#### 4.1 BERT 反常现象深度分析（面试核心故事）

```
现象：你的数据已显示 BERT 在 MNN 上比 ORT 慢 15%
      而 CV 模型 MNN 全面领先 1.3-2.4x

分析步骤：
  □ MNN Profiler 拆 BERT 每个算子的 CPU 耗时
  □ ORT Profiler 拆 BERT 每个算子的 CPU 耗时
  □ 逐算子对比例：找到最慢的那个算子（预计是 MatMul 768×768）
  □ 分析为什么 MNN 的 GEMM 在 768 这个尺寸上不如 ORT
    可能原因：分块策略、Pack 参数、内核对齐、线程切分方式
  □ simpleperf 火焰图交叉验证
```

#### 4.2 针对性优化

```
□ 基于 4.1 的分析结果，定位到具体优化点
□ 修改对应算子代码
□ 跑 before/after 的性能对比
□ 目标：BERT MNN 推理从 689ms 降到 ≤598ms（追平 ORT）或更快
□ 记录：做了什么改动、为什么这样改、性能提升数据
```

#### 4.3 其他算子优化（视时间而定）

```
□ 如果有分析发现 MNN 的 DepthwiseConv 在大通道场景下利用率低 → 优化
□ 如果有分析发现 Winograd 在小通道场景下变换开销占比太高 → 调整切换阈值
```

### Phase 5：LLM 推理专项（3-5 天）

#### 5.1 MNN vs llama.cpp 对比

```
□ 用 MNN 的 llm.cpp 加载 Qwen2-0.5B Q4
□ 跑 Prefill 和 Decode 阶段的 Profile
□ 和 llama.cpp 对比：
  - Prefill 处理速度（tok/s）
  - Decode 生成速度（tok/s）
  - 内存峰值
  - KV Cache 管理方式差异
```

#### 5.2 LLM 推理 Profiling

```
□ 拆分 Prefill 阶段：哪些算子占大头？
□ 拆分 Decode 阶段：KV Cache 访问模式？哪个算子瓶颈？
□ 分析 Decode 阶段为什么是访存密集（和 Prefill 的计算密集完全不同）
```

#### 5.3 产出一个 LLM 推理的简易优化建议

```
□ 不需要真的做一个完整的 FlashAttention 实现
□ 但需要能说出：
  - 标准 Attention 在 Decode 阶段的内存瓶颈
  - KV Cache 量化能省多少带宽
  - 哪些算子融合能减少 Kernel Launch 次数
```

### Phase 6：收尾项目文档与面试准备（2-3 天）

```
□ 整理所有 Profiling 数据为结构化报告
□ 为每个模型产出一份性能分析卡片（算子排行 + 瓶颈分析 + 优化建议）
□ 准备面试时的项目讲述话术
□ 更新简历中的项目描述
```

---

## 项目产出物清单

```
benchmark/
├── results/
│   ├── cpu_full_profile_20260X.md     # Phase 1 全量 Profiling 结果
│   ├── cpu_hotspot_benchmark_20260X.md# Phase 2 热点算子深度测试
│   ├── gpu_vs_cpu_comparison_20260X.md# Phase 3 CPU vs GPU 对比
│   ├── bert_optimization_20260X.md    # Phase 4 BERT 优化报告
│   └── llm_profile_20260X.md          # Phase 5 LLM 推理分析
└── src/
    └── single_op_benchmark.cpp        # 扩充满全量热点算子测例
```

---

## 面试项目讲述话术模板

```
1 分钟版（总括）：
  "我基于骁龙 865 搭了一套完整的端侧推理性能分析体系。选了 5 个模型
  覆盖 CV 到 LLM 的五种计算模式，用 MNN Profiler + simpleperf 做
  全量 Profiling，按二八原则对热点算子做三层深度的 Benchmark。
  然后用这个体系发现了 MNN 上 BERT 的异常瓶颈，做了针对性优化。"

3 分钟版（展开）：
  [1 min]  项目目标 + 平台 + 5 模型选择逻辑
  [1 min]  Benchmark 设计方法论：Profiling 驱动 + 二八原则 + 三层覆盖
  [1 min]  核心发现 + 优化闭环（BERT 从 689ms 降到 XXms）
            或：CPU vs GPU 异构调度的关键发现

追问应对：
  Q: "只测了一台手机？"
  A: "是的，但我在一台手机上开了 CPU 和 GPU 两个后端做深度对比。
      多设备的价值在于架构差异分析，而我当前的优先级是把单设备的
      性能特征挖透——这和写算子是一个思路，先做深再做宽。"
```

---

## 时间总览

| Phase | 内容 | 预估天数 | 累计 |
|-------|------|---------|------|
| 0 | 环境就绪 | 0.5 | 0.5 |
| 1 | CPU 全量 Profiling | 2-3 | 3.5 |
| 2 | CPU 热点算子深度 Benchmark | 3-5 | 8.5 |
| 3 | GPU 后端启用与对比 | 3-4 | 12.5 |
| 4 | 关键优化闭环（BERT） | 7-10 | 22.5 |
| 5 | LLM 推理专项 | 3-5 | 27.5 |
| 6 | 收尾文档与面试准备 | 2-3 | 30.5 |

**总计约 4-5 周**（按业余时间估算，每周可投入 3-4 个晚上 + 周末一天）。

如果时间紧张，Phase 5 和 Phase 6 可适当压缩，Phase 4 的 BERT 优化闭环是绝对不能跳过的核心。

---

## 关键原则

1. **数据要真**：所有性能数据必须是设备上真实跑出来的，不准伪造
2. **闭环要完整**：Profiling → 发现问题 → 分析根因 → 优化 → 再测 → 数据对比，每一步都不能少
3. **故事要好讲**：BERT 反常慢 15% 是你数据里目前最大的金子，优先挖
4. **先深后宽**：一台手机 + 两个后端搞透，比三台手机各测一遍有价值 10 倍
