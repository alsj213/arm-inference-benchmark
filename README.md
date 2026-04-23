# ARM Inference Benchmark

端侧深度学习推理框架性能基准测试项目，针对 ARM 架构手机芯片进行全面的性能对比。

![GitHub](https://img.shields.io/github/license/your-org/arm-inference-benchmark)
![Platform](https://img.shields.io/badge/platform-Android-lightgrey)
![Architecture](https://img.shields.io/badge/architecture-ARM64-brightgreen)

## ✨ 特性

- 🚀 **多框架支持**: ncnn, MNN, TFLite, ONNX Runtime, TVM, TNN, QNN
- 📊 **完整测试指标**: 延迟（P50/P90/P99）、吞吐量、初始化时间、内存占用
- 🤖 **真实模型测试**: MobileNetV2, ResNet50, YOLOv8n, BERT
- ⚡ **ARM 优化**: 原生 ARM64 编译，支持 NEON 优化
- 📱 **端侧友好**: 专为手机端侧推理设计的基准测试

## 📋 已支持框架

| 框架 | 状态 | 说明 |
|------|------|------|
| **ncnn** | ✅ | 腾讯 ncnn，手工 ARM 汇编优化 |
| **MNN** | ✅ | 阿里 MNN，移动端深度优化 |
| **TFLite** | ✅ | Google TensorFlow Lite |
| **ONNX Runtime** | ✅ | Microsoft ONNX Runtime Mobile |
| **TNN** | ✅ | 字节跳动 TNN，ARM NEON 汇编优化 |
| **TVM** | ⚙️ | Apache TVM - 可选（默认关闭） |
| **QNN** | ⚙️ | Qualcomm QNN SDK - 可选（默认关闭） |

## 📱 测试平台

### 骁龙 865 (SM8250)
- **CPU**: 1×A77@2.84GHz + 3×A77@2.42GHz + 4×A55@1.8GHz
- **GPU**: Adreno 650
- **ISA**: ARMv8.2-A, FP16

## 📊 基准测试结果 (MobileNetV2 FP32, 1 线程, 骁龙 865)

| 框架 | Init (ms) | P50 (ms) | P90 (ms) | Mean (ms) | FPS |
|------|-----------|----------|----------|-----------|-----|
| **TNN** | 151.29 | **13.14** | 13.22 | **13.15** | **76.07** |
| **MNN** | 26.82 | 18.54 | 18.68 | 18.55 | 53.90 |
| **ncnn** | 15.55 | 19.39 | 19.68 | 19.38 | 51.61 |
| **TFLite** | 15.07 | 22.70 | 22.80 | 22.67 | 44.11 |
| **ONNX Runtime** | 47.82 | 29.27 | 29.40 | 29.26 | 34.18 |

> TNN 使用 ARM NEON 汇编优化，性能显著领先；TVM 和 QNN 为可选框架，默认关闭
> 完整测试结果参见 [docs/results_sm8250.md](docs/results_sm8250.md)

## 🚀 快速开始

### 前置条件

- Android NDK (建议 r25c 或更高版本)
- Python 3.8+ (用于模型转换)
- Android 设备 (Android 10+, ARM64)
- CMake 3.18+

### 1. 克隆项目

```bash
git clone https://github.com/your-org/arm-inference-benchmark.git
cd arm-inference-benchmark
git submodule update --init --recursive
```

### 2. 环境配置

```bash
export ANDROID_NDK=/path/to/your/android-ndk-r25c
```

### 3. 编译

```bash
# 编译 Android 版本
./scripts/build_android.sh
```

### 4. 运行测试

```bash
# 推送并运行测试
./scripts/adb_run.sh --backend all --model mobilenetv2 --precision fp32 --threads 1 --runs 100
```

## 📁 项目结构

```
arm-inference-benchmark/
├── 📄 LICENSE                      # MIT 许可证
├── 📄 README.md                    # 项目说明
├── 📄 CMakeLists.txt              # 顶层 CMake 配置
├── 📄 .gitignore                  # Git 忽略配置
├── 📄 requirements.txt            # Python 依赖
├── 📁 cmake/                      # CMake 工具链
│   └── android.toolchain.cmake    # Android 交叉编译工具链
├── 📁 docs/                       # 文档
│   ├── results_sm8250.md          # 骁龙 865 测试结果
│   └── figures/                   # 性能图表
├── 📁 models/                     # 模型
│   ├── classification/            # 分类模型 (MobileNetV2, ResNet50)
│   ├── detection/                 # 检测模型 (YOLOv8n)
│   └── nlp/                       # NLP 模型 (BERT)
├── 📁 scripts/                    # 脚本
│   ├── build_android.sh           # Android 编译脚本
│   ├── build_host_tools.sh        # 主机构建转换工具脚本
│   ├── download_pretrained.py     # 下载预训练 ONNX 模型
│   ├── convert_models.sh          # 模型转换脚本 (所有框架)
│   ├── convert_tflite.py          # TFLite 转换脚本
│   ├── adb_run.sh                 # ADB 运行脚本
│   └── run_benchmark.sh           # 自动测试脚本
├── 📁 tools/                      # 转换工具
│   ├── bin/                       # 编译好的转换工具 (onnx2ncnn, onnx2mnn, onnx2tnn)
│   └── README.md                  # 工具使用说明
├── 📁 src/                        # 源代码
│   ├── main.cpp                   # 主程序入口
│   ├── CMakeLists.txt             # 源代码 CMake 配置
│   ├── common/                    # 公共模块
│   │   ├── benchmark.h/cpp        # 基准测试基类
│   │   ├── utils.h/cpp            # 工具函数（计时、内存统计）
│   │   └── config.h/cpp           # 配置定义
│   ├── backends/                  # 后端实现
│   │   ├── ncnn_backend.h/cpp     # ncnn 后端
│   │   ├── mnn_backend.h/cpp      # MNN 后端
│   │   ├── tflite_backend.h/cpp   # TFLite 后端
│   │   ├── ort_backend.h/cpp      # ONNX Runtime 后端
│   │   └── tvm_backend.h/cpp      # TVM 后端
│   └── models/                    # 模型信息
├── 📁 results/                    # 测试结果输出
└── 📁 third_party/                # 第三方依赖（git submodule）
    ├── ncnn/
    ├── MNN/
    ├── TNN/
    ├── tensorflow/
    ├── onnxruntime/
    └── tvm/
```

## 🔧 使用方式

### 命令行参数

```bash
./benchmark_inference [OPTIONS]

Options:
  --backend <backend>     指定后端: ncnn|mnn|tnn|tflite|qnn|onnxrt|tvm|all
  --model <model>         指定模型: mobilenetv2|resnet50|yolov8n|bert|all
  --precision <prec>      指定精度: fp32|fp16|int8
  --threads <num>         线程数 (默认: 1)
  --warmup <num>          warmup 次数 (默认: 10)
  --runs <num>            测试次数 (默认: 100)
  --gpu                   使用 GPU (如果支持)
  --help                  显示帮助
```

### 示例

```bash
# 测试所有框架在 MobileNetV2 上的性能
./benchmark_inference --backend all --model mobilenetv2 --runs 100

# 只测试 MNN
./benchmark_inference --backend mnn --model mobilenetv2 --precision fp32 --threads 4

# 测试多线程性能对比
for t in 1 2 4 8; do
    ./benchmark_inference --backend mnn --model mobilenetv2 --threads $t
done
```

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献指南

- 代码风格：遵循 Google C++ Style Guide
- 提交信息：使用清晰的描述
- 新增框架：请参考现有后端的实现方式

## 📝 开发计划

- [ ] 集成 Qualcomm QNN SDK
- [ ] TVM AutoTVM 自动调优
- [ ] GPU delegate 支持测试
- [ ] INT8 量化测试对比
- [ ] 更多模型支持 (YOLOv8n, BERT)
- [ ] 功耗测试功能
- [ ] 自动生成性能图表

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢以下开源项目：
- [ncnn](https://github.com/Tencent/ncnn) - 腾讯的高性能推理框架
- [MNN](https://github.com/alibaba/MNN) - 阿里的端侧推理引擎
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Google 的移动推理框架
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Microsoft 的跨平台推理引擎
- [Apache TVM](https://tvm.apache.org/) - 深度学习编译器栈
- [TNN](https://github.com/Tencent/TNN) - 字节跳动的推理框架

## 📧 联系方式

如有问题或建议，欢迎通过 Issue 反馈。

---

*本项目旨在为端侧推理框架的性能评估提供一个统一、可复现的基准测试平台。*
