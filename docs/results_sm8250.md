# 骁龙 865 (SM8250) 基准测试结果

测试设备: 高通骁龙 865 (1xA77@2.84GHz + 3xA77@2.42GHz + 4xA55@1.8GHz), Adreno 650
测试模型: MobileNetV2 (1x3x224x224)
精度: FP32
线程数: 1

## 最终测试结果

| 框架 | Init(ms) | P50(ms) | P90(ms) | Mean(ms) | FPS | 状态 |
|------|----------|---------|---------|----------|-----|------|
| **TNN** | 151.29 | **13.14** | 13.22 | **13.15** | **76.07** | ✅ 真实推理 - ARM汇编优化 |
| MNN  | 26.82    | 18.54   | 18.68   | 18.55    | 53.90 | ✅ 真实推理 - ARM优化 |
| ncnn | 15.55    | 19.39   | 19.68   | 19.38    | 51.61 | ✅ 真实推理 - ARM优化 |
| TFLite | 15.07   | 22.70    | 22.80    | 22.67     | 44.11 | ✅ 真实推理 - 官方TFLite C API |
| ONNX Runtime | 47.82 | 29.27 | 29.40 | 29.26 | 34.18 | ✅ 真实推理 |

## 框架状态说明

### ✅ 已完成真实测试
1. **TNN** - 字节跳动TNN框架，完整集成，ARM汇编优化，性能最佳
2. **MNN** - 阿里MNN框架，完整集成，ARM优化良好
3. **ncnn** - 腾讯ncnn框架，完整集成，ARM优化良好
4. **TFLite** - Google TensorFlow Lite，完整C API集成
5. **ONNX Runtime** - Microsoft ONNX Runtime，完整集成

## 模型目录
```
models/classification/mobilenetv2/
├── mobilenetv2.onnx              # ONNX模型 (用于ONNX Runtime)
├── mobilenetv2_MNN.mnn           # MNN模型
├── mobilenetv2_ncnn.model        # ncnn模型 (param)
├── mobilenetv2_ncnn.bin          # ncnn权重
├── mobilenetv2_TNN.tnnproto      # TNN模型
├── mobilenetv2_TNN.tnnmodel      # TNN权重
└── mobilenetv2.tflite            # TFLite模型
```

## 下一步工作
1. 增加INT8量化测试对比
2. 增加GPU delegate测试对比
3. 测试更多模型（ResNet50, YOLOv8n, BERT）

---
*更新时间: 2026-04-23*
*测试平台: Qualcomm Snapdragon 865 SM8250*
