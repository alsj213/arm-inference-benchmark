# MNN Add+LayerNorm 算子融合策略 — 最终结论

日期：2026-06-28  
分支：`opt/qwen3vl-add-rmsnorm-fusion` (origin/alsj213/MNN)

---

## 1. 融合策略概述

采用 **三层融合** 架构，确保所有 MNN 推理路径都能受益：

```
┌─────────────────────────────────────────────────────┐
│  第3层：Converter 层（writeFb.cpp）                 │
│  覆盖路径：ONNX→MNN 导出（所有模型）               │
│  时机：MNNConvert 序列化前修改 NetT FlatBuffer     │
│  状态：✅ 已实现，编译通过                          │
├─────────────────────────────────────────────────────┤
│  第2层：图优化层（GeometryComputerUtils.cpp）       │
│  覆盖路径：Session API 运行时（CNN / 小模型）       │
│  时机：Session::createSession 时扫描 command buffer │
│  状态：✅ 已验证，CNN 模型自动生效                  │
├─────────────────────────────────────────────────────┤
│  第1层：Kernel 层（CPUFusedAddRMSNorm.cpp）          │
│  覆盖路径：所有 2-input LayerNorm 调用的底层计算    │
│  实现：NEON intrinsics fused kernel                 │
│  状态：✅ 已验证，单算子 8.62×，模型解码 +16.6%     │
└─────────────────────────────────────────────────────┘
```

## 2. 各层详细说明

### 第1层：Kernel 层 (NEON Fused Kernel)

**文件**：`source/backend/cpu/compute/CPUFusedAddRMSNorm.cpp` (新文件, 136行)
- `source/backend/cpu/compute/CommonOptFunction.h` (声明)
- `source/backend/cpu/CPULayerNorm.cpp` (dispatch 逻辑)

**触发条件**：CPULayerNorm 检测到 inputs >= 2 且 mRMSNorm=true 时，调用 `MNNAddAndRMSNorm`

**核心算法**：
```cpp
// 2-pass NEON：
// Pass 1: vld1q → vaddq(src0+src1) → vmlaq(累加 sum_of_squares)
// Pass 2: scale = 1/sqrt(mean_sq + ε) → vmulq(scale) → vmulq(gamma) → vst1q
```

**验证数据**：
| 测试类型 | 独立 Add+LayerNorm | Fused Kernel | 加速比 |
|---------|-------------------|-------------|-------|
| 单算子 (2560维) | 0.052 ms | 0.006 ms | **8.62×** |
| 模型解码 (Qwen3-VL-4B) | 基准 | +16.6% | **1.17×** |

### 第2层：图优化层 (Runtime Graph Fusion)

**文件**：
- `source/geometry/GeometryComputerUtils.cpp` (+21行)
- `source/geometry/GeometryLayernorm.cpp` (+5行)

**触发条件**：Session API (`Interpreter::createSession`) 初始化时扫描 command buffer

**逻辑**：
```
command buffer 扫描 → 检测 BinaryOp(Add) → LayerNorm 相邻 → 
LayerNorm 的 inputs 改为 {Add.inputs[0], Add.inputs[1]} → 删除 Add op
```

**覆盖范围**：Session API（CNN 模型、单图推理）

### 第3层：Converter 层 (模型导出时融合)

**文件**：`tools/converter/source/common/writeFb.cpp` (+29行)

**触发条件**：`MNNConvert` 导出 ONNX/TF→MNN 时，在 `postTreat()` 阶段

**逻辑**：
```
lambda fuseAddLayerNorm(ops):
  for i in 0..len(ops)-1:
    if ops[i]是BinaryOp(ADD) 且 ops[i+1]是LayerNorm(useRMSNorm=true)
    且 ops[i].output[0] == ops[i+1].input[0]:
      → ops[i+1].inputs = {ops[i].inputs[0], ops[i].inputs[1]}
      → 删除 ops[i]
      → fused++

主图融合:
  fuseAddLayerNorm(netT->oplists)
子图融合 (MoE 模型):
  for subgraph in netT->subgraphs:
    fuseAddLayerNorm(subgraph->nodes)
```

**关键设计决策**：
- 只融合 `useRMSNorm=true` 的 LayerNorm（RMSNorm），不处理普通 LayerNorm
- 子图遍历确保 MoE 模型的 expert subgraphs 也能融合
- 融合在 `_postTreatOp` 之后、FlatBuffer 序列化之前执行

**覆盖范围**：所有通过 MNNConvert 导出的模型（CNN / LLM / MoE）

## 3. 模型导出验证流程

### CNN 模型（已验证 ✅）
```
ONNX → MNNConvert (writeFb.cpp 融合) → .mnn
                                   ↓
                       Interpreter::createSession
                                   ↓
                  GeometryComputerUtils (融合) → Session::run
```

### LLM 模型（代码就绪，待完整导出验证）
```
HF weights → llmexport.py → ONNX → MNNConvert (writeFb.cpp 融合) → .mnn
                                                              ↓
                                            Module::load → PipelineModule
                                                              ↓
                                          CPULayerNorm (Kernel 融合)
```

### 待验证步骤
1. 使用带融合代码的 MNNConvert 运行 llmexport（预计 30-60 分钟）
2. 将导出的 .mnn 模型推送到设备测试
3. 对比融合前后的解码性能

> **前提条件**：HF 权重需在 `/mnt/e/wsl/home_liu/models/llm/qwen3-vl-4b-hf/`（已确认存在）

## 4. 尝试过但放弃的方案

| 方案 | 问题 | 结论 |
|------|------|------|
| MNN→MNN 重转 | 破坏 LLM 模型的外部权重和子图结构 | ❌ 不适用 LLM |
| JSON patcher (patch_add_layernorm.py) | 修改 inputIndexes 不更新 FlatBuffer op 定义中的 input count | ❌ 模型无法加载 |
| PipelineModule 运行时融合 | 修改 oplists 破坏 tensor 生命周期 | ❌ 太危险 |
| 只保留 Add 不删除 | 模型静默失败 | ❌ 不可靠 |

## 5. 文件清单 (8 文件, +327 行)

```
MNN 子模块 (opt/qwen3vl-add-rmsnorm-fusion):
├── source/backend/cpu/compute/CPUFusedAddRMSNorm.cpp  [新文件] NEON kernel
├── source/backend/cpu/compute/CommonOptFunction.h      [+1行]  声明
├── source/backend/cpu/CPULayerNorm.cpp                 [+8行]  2-input dispatch
├── source/geometry/GeometryComputerUtils.cpp           [+21行] Session 融合
├── source/geometry/GeometryLayernorm.cpp               [+5行]  放宽 2-input
├── tools/converter/source/common/writeFb.cpp           [+29行] Converter 融合
└── tools/patch_add_layernorm.py                        [新文件] JSON patcher (参考用)

llama.cpp 子模块 (opt/qwen3vl-rmsnorm-neon):
└── ggml/src/ggml-cpu/ops.cpp                           [+24行] NEON RMSNorm
```

## 6. 验证结果 (2026-06-28)

### Converter 融合端到端验证 ✅

用 qwen2.5-1.5B 测试，使用带融合代码的 MNNConvert 运行 llmexport：

```
Fused 28 Add+LayerNorm pairs (converter pass)  ← 28层模型，正好28对
```

| 指标 | 原始模型 | 融合模型 (v2, keep-Add) |
|------|---------|----------------------|
| 融合对数 | 0 | 28 |
| 模型大小 | 488KB | 489KB |
| 加载 | ✅ | ✅ |
| 推理 | ✅ | ✅ |

### 关键策略修正

**v1 (erase)**: 删除 Add op → 模型加载失败（tensor 索引用变化导致 LLM 引擎出错）

**v2 (keep-Add)**: 保留 Add op，只改变 LayerNorm 的 inputIndexes → ✅ 加载成功

```cpp
// v2 策略：只重写 LayerNorm 输入，不删除 Add op
ops[i+1]->inputIndexes = {ops[i]->inputIndexes[0], ops[i]->inputIndexes[1]};
// 不删除 Add op，保留 FlatBuffer 索引稳定性
```

## 7. 后续行动计划

| 优先级 | 任务 | 预计时间 |
|-------|------|---------|
| P0 | Qwen3-VL-4B llmexport（带融合 MNNConvert） | 30-60 min |
| P0 | 设备上 A/B 测试 Qwen3-VL-4B 融合性能 | 15 min |
| P1 | 提交 MNN 分支 PR（6 文件，含 kernel + geometry + converter） | - |
| P1 | 提交 llama.cpp 分支 PR（1 文件，NEON RMSNorm） | - |
| P2 | CPUSoftmax 改动的处理（提交或还原） | 5 min |
