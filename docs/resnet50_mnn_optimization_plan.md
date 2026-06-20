# ResNet50 MNN 优化计划

> 基线: MNN 4T 84.5ms / TVM 64.6ms / ORT 133.3ms
> 目标: 分阶段将 MNN 延迟降至 TVM 水平或更好

---

## 0. 优化漏斗

```
基线 MNN: 84.5ms
  │
  ├─ Phase 1: 低风险快速见效        → 预计 55-65ms  (↓23-35%)
  │   ├─ 1.1 MATMUL 分块动态化        (↓5-8%)
  │   └─ 1.2 Winograd Conv3x3        (↓15-20%)
  │
  ├─ Phase 2: 中等改动               → 预计 40-50ms  (↓41-53%)
  │   ├─ 2.1 INT8 量化               (↓40-50%)
  │   └─ 2.2 Geometry Fusion 增强     (↓5-8%)
  │
  └─ Phase 3: 混合后端               → 预计 35-45ms  (↓47-59%)
      └─ 3.1 大算子 TVM 编译替换       (↓15-20%)
```

---

## Phase 1: 低风险快速见效

### 1.1 MATMUL 分块动态化

**现状**: MNN CPUMatMul 固定分块 eP=12, hP=8 (96 元素)，为 Cortex-A53 设计。

**改法**: 在 `MNNGetMatMulPackMode` 中按矩阵规模动态选择分块:

```cpp
// source/backend/cpu/arm/CommonOptFunctionNeon.cpp
void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    // 默认: 12×8 (适合 <256 通道的小矩阵)
    *eP = 12; *lP = 1; *hP = 8;
}

// 新增: 大矩阵专用分块
void MNNGetMatMulPackModeLarge(int* eP, int *lP, int* hP) {
    *eP = 24;   // 2x 原值
    *lP = 1;   
    *hP = 16;   // 2x 原值, 使用 32 个 NEON 寄存器
}
```

然后在 `CPUMatMul::onResize` 中根据 e×h 大小选择:

```cpp
if (e * h > 65536) {  // > 256×256 等价
    core->MNNGetMatMulPackModeLarge(&eP, &lP, &hP);
} else {
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
}
```

**影响范围**: 仅 ResNet50 的 Stage3+4 大 Conv1x1 (9 个算子)。  
**风险**: 低，分块增大需要更多栈内存，12×8→24×16 从 384B→1.5KB 仍在 L1 范围内。  
**预期**: Stage3+4 Conv1x1 加速 20-25%，整模型 5-8% → **78-80ms**。

### 1.2 Winograd Conv3x3

**现状**: ResNet50 Stage3+4 的 9 个大 Conv3x3 (231M FLOPs 每个) 是最大瓶颈 (36ms)。

**原理**: Winograd F(2×2, 3×3) 将 3×3 卷积的乘法量减少到原来的 2.25/9 = 25%。代价是需要额外的 transform 步骤和更多内存。

**改法**: MNN 已有 Winograd 基础设施 (`source/backend/cpu/CPUConvolution.cpp`)，需确认 ResNet50 的 stride=1 Conv3x3 已启用:

```
检查: CPUConvolution::onResize 中是否对 3×3 stride=1 启用了 Winograd
如果没有: 开启 Winograd 路径
```

**影响范围**: Stage3×6 + Stage4×3 = 9 个算子。  
**风险**: 中，Winograd 的 transform 开销对小 feature map (如 7×7) 可能得不偿失。需实测 14×14 和 7×7 两种情况。  
**预期**: Conv3x3 加速 2-3x，整模型 15-20% → **68-72ms**。

**Phase 1 完成后预期: 68-78ms，接近 TVM 的 64ms。**

---

## Phase 2: 中等改动

### 2.1 INT8 量化

**现状**: ResNet50 FP32 模型 97MB，INT8 可降至 25MB。

**步骤**:
```
1. 用 MNN 量化工具生成 INT8 模型
   $ ./MNNConvert -f ONNX --modelFile resnet50.onnx --MNNModel resnet50.mnn --quantize

2. 或用 MNN 的离线量化:
   $ python pymnn/examples/quantization.py --model resnet50.onnx --calib_dataset imagenet_calib/

3. 精度验证: 在 ImageNet val 上对比 FP32 vs INT8 Top-1 精度
   预期精度损失: <1%
```

**影响范围**: 全局，所有 Conv/FC 算子。  
**风险**: 中，需验证精度。SD865 的 A77 支持 i8sdot 指令，INT8 加速明显。  
**预期**: 整模型 1.5-2x → **35-45ms**。

### 2.2 Geometry Fusion 增强

**现状**: MNN 已融合 Conv+BN+ReLU，但 ResNet50 的 Add skip connection 无法融合。

**改法**: 增加 `Conv + BN + ReLU + Add` 融合 pattern:

```cpp
// source/geometry/ 新增融合规则
REGISTER_GEOMETRY(OpType_Add, GeometryConvAdd);
// 当 Add 的输入之一是 Conv+BN+ReLU 的输出时, 融合为单 kernel
```

**影响范围**: ResNet50 的 16 个 Bottleneck block 中的 skip connection。  
**风险**: 中，需要处理 Add 的 broadcasting 情况。  
**预期**: 逐元素开销减半，整模型 5-8% → **已在 INT8 之后约 32-42ms**。

**Phase 2 完成后预期: 32-45ms，超越 TVM 的 64ms。**

---

## Phase 3: 混合后端

### 3.1 大算子 TVM 编译替换

**思路**: 不替换整个模型，只替换瓶颈算子。

**步骤**:
```
1. 提取 ResNet50 瓶颈算子为独立 ONNX:
   Stage4 Conv3x3 (512→512, 7×7) ×3
   Stage3 Conv3x3 (256→256, 14×14) ×6
   Stage3/4 升维 Conv1x1 ×9

2. TVM 编译每个算子 (AutoScheduler 调优):
   $ python compile_op.py --op Conv3x3_R50_S14_C256_K256 --target "llvm -mcpu=cortex-a77"

3. 在 MNN 中注册为 Custom Op:
   MNN::OpType_Custom + 外部 .so 调用 TVM kernel

4. MNN 图优化: 识别这些算子 → 替换为 Custom Op → 调用 TVM kernel
```

**影响范围**: 18 个最高代价算子。  
**风险**: 高，Custom Op 注册需要处理 MNN/TVM 之间的 Tensor 格式转换 (NC4HW4 ↔ NCHW)。  
**预期**: 目标算子加速 30-40%，整模型 15-20%。Phase 2 INT8 之后再做此步预期有限（因为 INT8 已经大幅缩小瓶颈）。

**Phase 3 预期: 30-40ms (INT8 基础上)，或 55-65ms (FP32 基础上)。**

---

## 执行路线图

| 阶段 | 内容 | 工时 | 预期 MNN 延迟 | 风险 |
|:--:|------|:--:|:-----:|:--:|
| — | 基线 | — | 84.5ms | — |
| 1.1 | MATMUL 分块动态化 | 2天 | 78-80ms | 低 |
| 1.2 | Winograd Conv3x3 | 3天 | 68-72ms | 中 |
| 2.1 | INT8 量化 | 2天 | 35-45ms | 中 |
| 2.2 | Geometry Fusion Add | 5天 | 32-42ms | 中 |
| 3.1 | 混合 TVM 后端 | 10天 | 30-40ms | 高 |

**推荐执行策略**: Phase 1.1 + 1.2 + 2.1 三轮，投入 7 天，预期延迟从 84ms 降至 ~40ms，超越 TVM。

---

## 每个 Phase 的验证方法

| Phase | 验证方式 | 成功标准 |
|-------|---------|---------|
| 1.1 | GEMM 单算子 before/after | 大矩阵 GFLOPS 提升 20%+ |
| 1.2 | Stage3 Conv3x3 单算子 before/after | 延迟降低 2x+ |
| 2.1 | ImageNet Top-1 + 整模型延迟 | 精度损失 <1%, 延迟 <50ms |
| 2.2 | ResNet50 整模型 before/after | 逐元素 Profiling 占比减半 |
| 3.1 | Custom Op 精度 + 整模型延迟 | Cosine > 0.999, 延迟 <降 15% |

---

*基线数据: 骁龙 865 (SM8250) / 4T / FP32 / taskset 80 CPU7*
*分析分支: analysis/resnet50-mnn-profile*
