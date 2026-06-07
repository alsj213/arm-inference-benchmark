
## MNN vs ORT 全模型 Latency 对比

| 模型 | 后端 | Mean(ms) | P50(ms) | P90(ms) | P99(ms) | Std(ms) | FPS | Init(ms) | 精度 |
|------|------|----------|---------|---------|---------|---------|-----|----------|------|
| bert | MNN | 683.45 | 683.44 | 684.55 | 689.30 | 0.99 | 1.5 | 1279.11 | PASSED |
| bert | ORT | 598.80 | 598.85 | 599.41 | 600.14 | 0.54 | 1.7 | 641.74 | PASSED |
| mobilenetv2 | MNN | 18.49 | 18.49 | 18.59 | 18.70 | 0.08 | 54.1 | 26.45 | PASSED |
| mobilenetv2 | ORT | 29.17 | 29.14 | 29.37 | 31.25 | 0.27 | 34.3 | 29.31 | PASSED |
| resnet50 | MNN | 143.88 | 143.87 | 144.26 | 146.02 | 0.38 | 7.0 | 693.27 | PASSED |
| resnet50 | ORT | 221.56 | 221.58 | 222.02 | 222.20 | 0.35 | 4.5 | 388.03 | PASSED |
| yolov8n | MNN | 174.00 | 174.02 | 174.42 | 174.74 | 0.32 | 5.8 | 155.46 | PASSED |
| yolov8n | ORT | 304.19 | 304.09 | 305.38 | 306.34 | 0.72 | 3.3 | 27.07 | PASSED |

## MNN vs ORT 加速比

| 模型 | MNN Mean(ms) | ORT Mean(ms) | MNN/ORT 加速比 | 胜出 |
|------|-------------|-------------|---------------|------|
| bert | 683.45 | 598.80 | 0.88x | **ORT** |
| mobilenetv2 | 18.49 | 29.17 | 1.58x | **MNN** |
| resnet50 | 143.88 | 221.56 | 1.54x | **MNN** |
| yolov8n | 174.00 | 304.19 | 1.75x | **MNN** |

## 测试配置
- Warmup: 50 iterations, Test: 100 iterations
- CPU: 单线程 (Threads=1)
- Precision: FP32
- 频率策略: schedutil (未锁频)
- Accuracy: vs ONNX Runtime 参考输出
