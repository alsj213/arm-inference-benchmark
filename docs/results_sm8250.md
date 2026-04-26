# 骁龙 865 (SM8250) 基准测试结果

测试设备: 高通骁龙 865 (1xA77@2.84GHz + 3xA77@2.42GHz + 4xA55@1.8GHz), Adreno 650
精度: FP32

---

## 🎯 精度对比说明

从 2026-04-26 开始，所有测试包含精度对比：
- ✅ 使用 ONNX Runtime 作为标杆，获取参考输出
- ✅ 计算余弦相似度、绝对误差、相对误差
- ✅ 确保各框架推理结果与 ORT 一致

**精度对比指标：**
| 指标 | 含义 | 优秀标准 |
|------|------|---------|
| Cosine Similarity | 余弦相似度 | >0.999 |
| Mean Absolute Error | 平均绝对误差 | <0.001 |
| Max Absolute Error | 最大绝对误差 | <0.01 |
| Mean Relative Error | 平均相对误差 | <0.1% |

**精度测试数据汇总 (4线程, FP32, 2026-04-26)**

| 模型 | 对比框架 | Cosine Similarity | Mean Abs Error | Max Abs Error | Mean Rel Error | 状态 |
|------|---------|-------------------|----------------|---------------|----------------|------|
| **MobileNetV2** | MNN vs ORT | 1.000000 | 0.000002 | 0.000009 | 0.0005% | ✅ Excellent |
| **MobileViT-S** | MNN vs ORT | 1.000000 | 0.000173 | 0.000595 | 0.0002% | ✅ Excellent |

**结论：** MNN 框架与 ONNX Runtime 输出高度一致，精度损失可忽略不计。

---

## 🆕 4 线程测试 (2026-04-26 更新)

### 多模型对比测试

| 模型 | 参数量 | 框架 | 初始化(ms) | P50(ms) | P90(ms) | P99(ms) | Mean(ms) | FPS | MNN vs ORT 加速比 |
|------|-------|------|-----------|---------|---------|---------|----------|-----|-------------------|
| **MobileNetV2** | 3.5M | ONNX Runtime | 49.05 | 18.16 | 18.36 | 18.53 | 18.17 | 55.0 | - |
| **MobileNetV2** | 3.5M | MNN | 29.53 | **8.70** | 8.96 | 9.14 | 8.70 | **114.9** | **2.09x** |
| **ShuffleNetV2 x0.5** | 1.4M | ONNX Runtime | 183.76 | 2.89 | 2.92 | 4.15 | 2.94 | 339.8 | - |
| **ShuffleNetV2 x0.5** | 1.4M | MNN | 89.19 | **1.69** | 1.92 | 1.97 | 1.73 | 577.6 | **1.71x** |
| **ResNet50** | 25.6M | ONNX Runtime | 377.52 | **78.53** | 79.01 | 79.79 | 78.61 | 12.7 | - |
| **MobileViT-S** | ~5M | ONNX Runtime | 162.55 | 71.31 | 76.63 | 85.52 | 72.49 | 13.79 | - |
| **MobileViT-S** | ~5M | MNN | 95.16 | **59.65** | **61.84** | 67.67 | **60.16** | **16.62** | **1.20x** |

### MNN 优势分析

1. **NCHW4c 内存格式** - 将 4 个通道打包成 SIMD 向量，提升 Cache 命中率
2. **手写 NEON 汇编** - 核心卷积算子深度优化
3. **Winograd 算法** - 3x3 卷积乘法次数减少
4. **高效的图优化** - 算子融合、内存复用

---

## 1 线程测试 (历史数据)

测试模型: MobileNetV2 (1x3x224x224)

| 框架 | Init(ms) | P50(ms) | P90(ms) | Mean(ms) | FPS | 状态 |
|------|----------|---------|---------|----------|-----|------|
| **TNN** | 151.29 | **13.14** | 13.22 | **13.15** | **76.07** | ✅ 真实推理 - ARM汇编优化 |
| MNN  | 26.82    | 18.54   | 18.68   | 18.55    | 53.90 | ✅ 真实推理 - ARM优化 |
| ncnn | 15.55    | 19.39   | 19.68   | 19.38    | 51.61 | ✅ 真实推理 - ARM优化 |
| TFLite | 15.07   | 22.70    | 22.80    | 22.67     | 44.11 | ✅ 真实推理 - 官方TFLite C API |
| ONNX Runtime | 47.82 | 29.27 | 29.40 | 29.26 | 34.18 | ✅ 真实推理 |

---

## 框架状态说明

### ✅ 已完成真实测试
1. **MNN** - 阿里 MNN 框架，移动端深度优化，**4线程性能最佳**
2. **ONNX Runtime** - Microsoft ONNX Runtime Mobile，跨平台通用
3. **TNN** - 字节跳动 TNN 框架，ARM 汇编优化，单线程性能领先
4. **ncnn** - 腾讯 ncnn 框架，成熟的 ARM 优化
5. **TFLite** - Google TensorFlow Lite，官方 C API 集成

### 📁 模型目录
```
models/classification/
├── mobilenetv2/           # MobileNetV2 模型
│   ├── mobilenetv2.onnx
│   └── mobilenetv2_MNN.mnn
├── shufflenet_v2/         # ShuffleNetV2 x0.5 模型
│   ├── shufflenet_v2_x0_5.onnx
│   └── shufflenet_v2_x0_5_MNN.mnn
├── mobilevit_s/            # MobileViT-S 模型
│   ├── mobilevit_s.onnx
│   └── mobilevit_s_MNN.mnn
└── resnet50/              # ResNet50 模型
    └── resnet50.onnx
```

---

## 下一步工作
1. 增加 INT8 / FP16 量化测试对比
2. 增加更多模型测试（YOLOv8n, BERT）
3. 增加 llama.cpp LLM 性能测试
4. 增加 GPU delegate 测试对比

---
*更新时间: 2026-04-26*
*测试平台: Qualcomm Snapdragon 865 SM8250*
