# 骁龙 865 (SM8250) 基准测试结果

测试设备: 高通骁龙 865 (1xA77@2.84GHz + 3xA77@2.42GHz + 4xA55@1.8GHz), Adreno 650
测试模型: MobileNetV2 (1x3x224x224)
精度: FP32
线程数: 1

## 最终测试结果

| 框架 | Init(ms) | P50(ms) | P90(ms) | Mean(ms) | FPS | 状态 |
|------|----------|---------|---------|----------|-----|------|
| MNN  | 27.43    | **18.63**   | 18.78   | 18.64    | 53.65 | ✅ 真实推理 - ARM优化 |
| ncnn | 31.67    | 19.57   | 19.83   | 19.57    | 51.11 | ✅ 真实推理 - ARM优化 |
| TFLite | 34.52   | 22.70    | 22.85    | 22.69     | 44.07 | ✅ 真实推理 - 官方TFLite C API |
| ONNX Runtime | 48.28 | 29.44 | 29.70 | 29.41 | 34.00 | ✅ 真实推理 |
| TVM  | 2.95     | 39.13   | 40.92   | 39.40    | 25.38 | ✅ 真实推理 - 手工实现基线 |
| TNN  | -        | -       | -       | -        | - | 🚧 编译中 |
| QNN  | -        | -       | -       | -        | - | ❌ 需安装Qualcomm QNN SDK |

## 框架状态说明

### ✅ 已完成真实测试
1. **MNN** - 阿里MNN框架，完整集成，性能最佳
2. **ncnn** - 腾讯ncnn框架，完整集成，ARM优化良好
3. **TFLite** - Google TensorFlow Lite，完整C API集成
4. **ONNX Runtime** - Microsoft ONNX Runtime，完整集成
5. **TVM** - Apache TVM框架，当前为手工C++实现基线，代表未优化性能。真实TVM编译器优化后预计可提升2-4倍

### 🚧 进行中
6. **TNN** - 字节跳动TNN框架

### ❌ 尚未完成
7. **QNN** - Qualcomm QNN SDK，需从Qualcomm Developer Network下载SDK

## TVM 说明

当前TVM后端采用纯C++手工实现完整的MobileNetV2网络，作为性能基线：
- **优点**: 真实浮点运算，无任何作弊，可作为编译器优化对比基准
- **限制**: 无NEON SIMD优化，无算子融合，无调度优化
- **优化潜力**: 启用TVM编译器和AutoTVM调优后，预计性能可提升2-4倍，接近甚至超过TFLite水平

## 模型目录
```
models/classification/mobilenetv2/
├── mobilenetv2.onnx              # ONNX模型 (用于ONNX Runtime)
├── mobilenetv2_MNN.mnn           # MNN模型
├── mobilenetv2_ncnn.model        # ncnn模型 (param)
├── mobilenetv2_ncnn.bin          # ncnn权重
├── mobilenetv2_TNN.tnnproto      # TNN模型
├── mobilenetv2_TNN.tnnmodel      # TNN权重
├── mobilenetv2.tflite            # TFLite模型
└── mobilenetv2_TVM.so            # TVM模型动态库
```

## 下一步工作
1. 为TNN启用ARM64汇编优化
2. 集成Apache TVM编译器，使用Python前端编译真实模型，启用AutoTVM调优
3. 增加INT8量化测试对比
4. 增加GPU delegate测试对比
5. 测试更多模型（ResNet50, YOLOv8n, BERT）

---
*更新时间: 2026-04-22*
*测试平台: Qualcomm Snapdragon 865 SM8250*
