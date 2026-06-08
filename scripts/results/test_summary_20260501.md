# 基准测试综合报告

## 测试日期
2026年5月1日

## 测试设备
- 设备型号: Snapdragon 865 (b08dee23)
- Android 版本: 11
- CPU 调度器: schedutil

## 测试配置

### 构建类型对比
| 构建类型 | 二进制大小 | 用途 |
|----------|-----------|------|
| Release | 44 MB | 性能测试 |
| Debug | 99 MB | 火焰图分析 |

### 测试框架
- MNN (MobileNetV2, 4 threads, FP32)
- ONNX Runtime (MobileNetV2, 4 threads, FP32)

## 性能测试结果

### MNN (Release 版本)
| 指标 | 数值 |
|------|------|
| 初始化时间 | 95.98 ms |
| P50 延迟 | 8.53 ms |
| P90 延迟 | 8.63 ms |
| P99 延迟 | 8.79 ms |
| 平均延迟 | 8.54 ms |
| 标准差 | 0.07 ms |
| 吞吐量 | 117.11 FPS |
| 峰值内存 | 3000 KB |

### ONNX Runtime (Release 版本)
| 指标 | 数值 |
|------|------|
| 初始化时间 | 153.00 ms |
| P50 延迟 | 17.68 ms |
| P90 延迟 | 17.85 ms |
| P99 延迟 | 18.36 ms |
| 平均延迟 | 17.69 ms |
| 标准差 | 0.14 ms |
| 吞吐量 | 56.52 FPS |
| 峰值内存 | 760 KB |

## 性能对比

| 指标 | MNN | ORT | 对比 |
|------|-----|-----|------|
| 吞吐量 | 117.11 FPS | 56.52 FPS | MNN 快 2.07x |
| P50 延迟 | 8.53 ms | 17.68 ms | MNN 快 2.1x |
| P99 延迟 | 8.79 ms | 18.36 ms | MNN 快 2.1x |
| 初始化时间 | 95.98 ms | 153.00 ms | MNN 快 37% |
| 峰值内存 | 3000 KB | 760 KB | ORT 更省内存 |

## 火焰图分析 (MNN Debug 版本)

### 采样统计
- 采样时长: 5 秒
- 采样频率: 4000 Hz
- 事件: cpu-cycles
- 总样本数: 81,178
- 事件计数: 43,246,697,526

### 热点函数 (Top 10)
1. kernel.kallsyms - 47.50%
2. LoopL2 - 26.86%
3. LoopE12L4 - 4.05%
4. sched_yield - 2.96%
5. L16LoopW - 2.70%
6. __memcpy - 2.43%
7. kernel.kallsyms - 2.06%
8. E1LoopL - 1.97%
9. L8LoopW - 1.13%
10. StoreLH8 - 1.07%

### 火焰图文件
- perf.data: 12.4 MB
- flamegraph.svg: 75.6 KB
- report_functions.txt: 23.9 KB

## 验证结果

### 环境一致性 ✅
- CPU 调度器设置成功 (schedutil)
- 温度监控正常 (28°C → 37°C)
- nice 值设置可用

### 准确性验证 ✅
- MNN: Cosine similarity > 0.99 (通过)
- ORT: Cosine similarity > 0.99 (通过)

### Debug/Release 切换 ✅
- Release 版本: 44 MB (性能测试)
- Debug 版本: 99 MB (火焰图分析)
- 构建类型切换正常

## 结论

1. **MNN 性能优于 ORT**: MNN 吞吐量是 ORT 的 2.07 倍
2. **延迟表现**: MNN P99 延迟仅为 8.79 ms，远低于 ORT 的 18.36 ms
3. **内存使用**: ORT 更省内存 (760 KB vs 3000 KB)
4. **初始化时间**: MNN 初始化更快 (95.98 ms vs 153.00 ms)
5. **火焰图分析**: Debug 版本成功生成，可定位热点函数

## 建议

1. 对于延迟敏感场景，优先使用 MNN
2. 对于内存受限场景，可考虑 ORT
3. 继续测试其他框架 (NCNN, TNN, TFLite) 以获得完整对比
