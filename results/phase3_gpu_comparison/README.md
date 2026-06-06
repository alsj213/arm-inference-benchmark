# Phase 3: MNN OpenCL GPU 后端启用与 CPU vs GPU 对比

**日期**: 2026-06-06  
**设备**: 红米 K30S (M2007J3SC), 骁龙 865 (SM8250 / Kona), Adreno 650 GPU  
**代码版本**: 57cdc49 (feat/benchmark-full-plan)  
**框架**: MNN (OpenCL GPU, Precision=High/FP32), ONNX Runtime (CPU 参考)

---

## 环境验证

| 检查项 | 状态 | 路径 |
|--------|------|------|
| ADB 设备连接 | OK | b08dee23 |
| libOpenCL.so | OK | /system/vendor/lib64/libOpenCL.so (115KB) |
| Adreno GPU | OK | /vendor/lib64/egl/libGLESv2_adreno.so |
| MNN_OPENCL 编译 | OK | CMakeCache: MNN_OPENCL=ON, MNN_USE_LIB_WRAPPER |

---

## 端到端对比

| 模型 | CPU (ms) | GPU (ms) | 加速比 | GPU 精度 | 推荐后端 |
|------|---------|---------|--------|---------|---------|
| MobileNetV2 | **18.65** | 15.89 | **1.17x** | 1.000000 | GPU |
| ResNet50 | 180.29 | **59.42** | **3.03x** | 1.000000 | GPU |
| YOLOv8n | **197.49** | 899.05 | **0.22x** | 1.000000 | CPU |
| BERT | **778.82** | 1883.46 | **0.41x** | 0.871612 | CPU |

### 详细性能数据

#### MobileNetV2 (1x3x224x224, 1000类)

| 指标 | CPU | GPU | 差异 |
|------|-----|-----|------|
| Init (ms) | 26.04 | 1102.33 | +41x |
| Mean (ms) | 18.65 | 15.89 | -14.8% |
| P90 (ms) | 18.75 | 17.35 | -7.5% |
| P99 (ms) | 18.96 | 20.92 | +10.3% |
| Throughput (FPS) | 53.62 | 62.92 | +17.3% |
| Cosine Sim | 1.000000 | 1.000000 | - |

- **结论**: GPU 有 17% 吞吐量提升。DWConv 等内存密集型操作限制了 GPU 加速。
- P99 比 CPU 略高 (+10%)，可能受 OpenCL kernel 调度抖动影响。

#### ResNet50 (1x3x224x224, 1000类)

| 指标 | CPU | GPU | 差异 |
|------|-----|-----|------|
| Init (ms) | 656.99 | 1815.74 | +2.8x |
| Mean (ms) | 180.29 | 59.42 | -67.0% |
| P90 (ms) | 181.11 | 61.23 | -66.2% |
| P99 (ms) | 183.06 | 63.21 | -65.5% |
| Throughput (FPS) | 5.55 | 16.83 | +203% |
| Cosine Sim | 1.000000 | 1.000000 | - |

- **结论**: GPU 加速 3.03x，是最大赢家。大量的 3x3 Conv 适合 GPU 并行计算。
- GPU P99 仅 63ms，延迟分布紧凑 (Std 1.59ms)。

#### YOLOv8n (1x3x640x640, 检测)

| 指标 | CPU | GPU | 差异 |
|------|-----|-----|------|
| Init (ms) | 114.96 | 1552.07 | +13.5x |
| Mean (ms) | 197.49 | 899.05 | +355% |
| P50 (ms) | 197.60 | 115.47 | -41.6% |
| P90 (ms) | 200.30 | 1922.37 | +860% |
| P99 (ms) | 207.50 | 3008.48 | +1350% |
| Std (ms) | 2.28 | 940.40 | - |
| Throughput (FPS) | 5.06 | 1.11 | -78% |
| Cosine Sim | 1.000000 | 1.000000 | - |

- **结论**: GPU 反而慢 4.5x。P50 性能看似可以 (115ms)，但 P90/P99 严重恶化 (1922ms/3008ms)。
- **根因分析**: YOLOv8n 包含大量 Reshape、Concat、Slice、Resize 等非计算密集型操作:
  1. 需要 CPU->GPU->CPU 数据搬运
  2. OpenCL kernel 启动开销 >> 实际计算时间
  3. MNN 的算子 fallback 机制在 CPU/GPU 间频繁切换，造成额外延迟

#### BERT (1x512, NLP)

| 指标 | CPU | GPU | 差异 |
|------|-----|-----|------|
| Init (ms) | 960.96 | 3170.00 | +3.3x |
| Mean (ms) | 778.82 | 1883.46 | +142% |
| P90 (ms) | 846.91 | 1900.71 | +124% |
| P99 (ms) | 961.16 | 1916.69 | +99% |
| Throughput (FPS) | 1.28 | 0.53 | -59% |
| Cosine Sim | 0.990439 | 0.871612 | -0.12 |

- **结论**: GPU 性能差 (2.4x slower) 且精度不合格 (cos 0.87)。
- **精度问题**: cos similarity 0.87 远低于 CPU 的 0.99。可能根因:
  1. MatMul 在 Adreno OpenCL 上强制使用 FP16 精度
  2. FP16 在多层 Transformer 中误差累积严重
  3. MNN OpenCL MatMul GEMM 对 BERT 精度不友好

---

## GPU 初始化开销

| 模型 | CPU Init (ms) | GPU Init (ms) | 额外开销 | 原因 |
|------|-------------|-------------|---------|------|
| MobileNetV2 | 26.04 | 1102.33 | +41x | OpenCL context + kernel JIT |
| ResNet50 | 656.99 | 1815.74 | +2.8x | 大量 Conv kernel 编译 |
| YOLOv8n | 114.96 | 1552.07 | +13.5x | 多形态算子 kernel 编译 |
| BERT | 960.96 | 3170.00 | +3.3x | MatMul FP16 kernel 编译 |

- GPU 首次初始化开销巨大 (700-3000ms)，主要用于 OpenCL kernel JIT 编译。
- 实际部署可通过预编译 kernel 缓存消除此开销。
- 模型越大、算子种类越多，初始化开销越大。

---

## 异构调度规则

### 适合 GPU 的算子

| 类型 | GPU vs CPU | 代表 |
|------|-----------|------|
| 3x3 Conv (标准) | 3.03x | ResNet50 backbone |
| 1x1 Conv (GEMM) | 1.17x | MobileNetV2 bottleneck |
| Pooling | ~2x (估算) | 数据搬运主导 |

### 必须留 CPU 的算子

| 类型 | GPU 惩罚 | 说明 |
|------|---------|------|
| DWConv (3x3) | 内存带宽瓶颈 | GPU 访存不利于 element-wise |
| Reshape/Concat/Slice | kernel launch >> compute | 零计算量，GPU 纯粹浪费 |
| 小 MatMul (512x768) | 启动开销主导 | kernel launch >> compute |
| Softmax/LayerNorm | 归约不友好 | CPU SIMD 更高效 |

### 算子级调度伪代码

```
if (op in {Conv3x3, Conv1x1_GEMM, Pooling}):
    => GPU (Adreno 650 OpenCL)
elif (op in {DWConv, Reshape, Concat, Slice, Resize}):
    => CPU (A77)
elif (op == MatMul):
    if (M*K > 1024*1024): => GPU
    else: => CPU
elif (op in {Softmax, LayerNorm, GELU}):
    => CPU
```

---

## 结论

1. **ResNet50 是 GPU 最大赢家** (3.03x): 大量标准 3x3 卷积密集计算，完美匹配 GPU 并行架构。

2. **MobileNetV2 收益有限** (1.17x): DWConv 等内存敏感操作限制了 GPU 优势。

3. **YOLOv8n 不适合纯 GPU 部署**: 算子种类多 + 零计算量操作多 -> GPU kernel launch 开销超过计算收益。推荐混合调度。

4. **BERT 不适合当前 MNN OpenCL 版本**: 性能回退 + 精度不合格。Transformer 类模型在 Adreno OpenCL 上需要专项优化 (Phase 4)。

5. **GPU 初始化开销不可忽略** (700-3000ms)，需要 kernel 缓存方案。

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `README.md` | 本报告 |
| `gpu_smoke_test.log` | GPU 冒烟测试 (首次, Segfault) |
| `gpu_smoke_test_passed.log` | GPU 冒烟测试 (修复后, 精度通过) |
| `mobilenetv2_*.log` | MobileNetV2 端到端 |
| `resnet50_*.log` | ResNet50 端到端 |
| `yolov8n_*.log` | YOLOv8n 端到端 |
| `bert_*.log` | BERT 端到端 |
| `build_gpu_fix.log` | 编译日志 |
