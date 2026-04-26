# ARM Inference Benchmark

端侧深度学习推理框架性能基准测试项目，针对 ARM 架构手机芯片进行全面的性能对比。

![GitHub](https://img.shields.io/github/license/your-org/arm-inference-benchmark)
![Platform](https://img.shields.io/badge/platform-Android-lightgrey)
![Architecture](https://img.shields.io/badge/architecture-ARM64-brightgreen)

## ✨ 特性

- 🚀 **多框架支持**: ncnn, MNN, TFLite, ONNX Runtime, TVM, TNN, QNN
- 📊 **完整测试指标**: 延迟（P50/P90/P99）、吞吐量、初始化时间、内存占用
- 🎯 **精度对比**: 以 ONNX Runtime 为标杆，自动计算余弦相似度、绝对误差、相对误差
- 🤖 **真实模型测试**: MobileNetV2, ResNet50, MobileViT-S, ShuffleNetV2 x0.5, YOLOv8n
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

### 已验证设备

| 设备型号 | 芯片 | 说明 |
|---------|------|------|
| **红米 K30 Pro** | 骁龙 865 (SM8250) | ✅ 完全验证 |

### 骁龙 865 (SM8250) - 红米 K30 Pro
- **CPU**: 1×A77@2.84GHz + 3×A77@2.42GHz + 4×A55@1.8GHz
- **GPU**: Adreno 650
- **ISA**: ARMv8.2-A, FP16
- **推荐线程数**: 4 (使用大核)

## 📊 基准测试结果 (骁龙 865, FP32)

### 4 线程性能测试 (2026-04-26 更新)

| 模型 | 框架 | Init(ms) | P50(ms) | P90(ms) | Mean(ms) | FPS | MNN vs ORT 加速比 |
|------|------|----------|---------|---------|----------|-----|-------------------|
| **MobileNetV2** | ONNX Runtime | 49.05 | 18.16 | 18.36 | 18.17 | 55.0 | - |
| **MobileNetV2** | MNN | 29.53 | **8.70** | 8.96 | 8.70 | **114.9** | **2.09x** |
| **ShuffleNetV2 x0.5** | ONNX Runtime | 183.76 | 2.89 | 2.92 | 2.94 | 339.8 | - |
| **ShuffleNetV2 x0.5** | MNN | 89.19 | **1.69** | 1.92 | 1.73 | 577.6 | **1.71x** |
| **ResNet50** | ONNX Runtime | 388.38 | 84.48 | 86.94 | 84.94 | 11.77 | - |
| **ResNet50** | MNN | 497.68 | **82.54** | 85.27 | 83.66 | **11.95** | **1.02x** |
| **MobileViT-S** | ONNX Runtime | 162.55 | 71.31 | 76.63 | 72.49 | 13.79 | - |
| **MobileViT-S** | MNN | 95.16 | **59.65** | **61.84** | **60.16** | **16.62** | **1.20x** |
| **YOLOv8n** | ONNX Runtime | 73.27 | 106.55 | 108.68 | 106.62 | 9.38 | - |
| **YOLOv8n** | MNN | 105.73 | **79.60** | 86.18 | 82.02 | **12.19** | **1.34x** |

### 4 线程精度对比 (MNN vs ORT 标杆)

| 模型 | Cosine Similarity | Mean Abs Error | Max Abs Error | Mean Rel Error | 状态 |
|------|-------------------|----------------|---------------|----------------|------|
| **MobileNetV2** | 1.000000 | 0.000002 | 0.000009 | 0.0005% | ✅ Excellent |
| **ResNet50** | 1.000000 | 0.000070 | 0.000359 | 0.0227% | ✅ Excellent |
| **MobileViT-S** | 1.000000 | 0.000173 | 0.000595 | 0.0002% | ✅ Excellent |
| **YOLOv8n** | 1.000000 | 0.000082 | 0.046753 | 0.8522% | ✅ Excellent |

### 1 线程测试 (历史数据)

| 框架 | Init (ms) | P50 (ms) | P90 (ms) | Mean (ms) | FPS |
|------|-----------|----------|----------|-----------|-----|
| **TNN** | 151.29 | **13.14** | 13.22 | **13.15** | **76.07** |
| **MNN** | 26.82 | 18.54 | 18.68 | 18.55 | 53.90 |
| **ncnn** | 15.55 | 19.39 | 19.68 | 19.38 | 51.61 |
| **TFLite** | 15.07 | 22.70 | 22.80 | 22.67 | 44.11 |
| **ONNX Runtime** | 47.82 | 29.27 | 29.40 | 29.26 | 34.18 |

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

---

## 📱 红米手机完整测试指南

本指南详细介绍如何在 **红米 K30 Pro**（骁龙 865）等红米系列手机上进行性能测试。

### 📋 前置准备

#### 1. 手机端设置

```
1. 打开 "设置" → "我的设备" → "全部参数"
2. 连续点击 "MIUI 版本" 7 次，开启开发者模式
3. 返回 "设置" → "更多设置" → "开发者选项"
4. 开启以下选项：
   ✅ USB 调试
   ✅ USB 安装
   ✅ USB 调试（安全设置）
5. 将手机连接电脑，选择 "文件传输" 模式
6. 手机弹窗点击 "允许 USB 调试"
```

#### 2. 电脑端 ADB 配置 (WSL2)

```bash
# 配置 Windows ADB 路径
export PATH=$PATH:/mnt/e/andorid/adb/
alias adb='/mnt/e/andorid/adb/adb.exe'

# 验证设备连接
adb devices
# 应显示类似：
# List of devices attached
# b08dee23        device
```

#### 3. 安装 Android NDK

```bash
# 下载 NDK r25c (推荐版本)
wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip
unzip android-ndk-r25c-linux.zip

# 配置环境变量
export ANDROID_NDK=/path/to/android-ndk-r25c
```

---

### 🚀 快速开始（5 分钟完成）

#### 步骤 1: 编译 Android 版本

```bash
cd arm-inference-benchmark

# 配置 NDK 路径（替换为你的实际路径）
export ANDROID_NDK=/home/liu/android-ndk-r25c

# 执行编译脚本（自动编译 MNN 和 ONNX Runtime）
./scripts/build_android.sh

# 编译成功后，可执行文件位于：
ls -lh build_android/src/benchmark_inference
```

#### 步骤 2: 推送文件到手机

```bash
# 推送二进制文件
adb push build_android/src/benchmark_inference /data/local/tmp/benchmark/

# 推送模型文件（所有分类模型）
adb push models/classification /data/local/tmp/benchmark/models/classification

# 推送 ONNX Runtime 库（如果需要）
adb push third_party/onnxruntime/lib-android/aarch64/libonnxruntime.so /data/local/tmp/benchmark/
```

#### 步骤 3: 运行基准测试

**方式一：使用自动化脚本（推荐）**

```bash
# 测试 MobileNetV2 - 所有框架对比
./scripts/adb_run.sh --backend all --model mobilenetv2 --precision fp32 --threads 4 --runs 50

# 测试 MobileViT-S - 单框架测试
./scripts/adb_run.sh --backend mnn --model mobilevit_s --precision fp32 --threads 4 --runs 50

# 完整性能测试 - 所有模型 + 所有框架
./scripts/adb_run.sh --backend all --model all --precision fp32 --threads 4 --runs 50
```

**方式二：手动执行（灵活）**

```bash
# 进入手机 shell
adb shell

# 设置库路径并运行
cd /data/local/tmp/benchmark
LD_LIBRARY_PATH=/data/local/tmp/benchmark ./benchmark_inference \
  --backend mnn \
  --model shufflenet_v2_x0_5 \
  --precision fp32 \
  --threads 4 \
  --warmup 10 \
  --runs 100
```

**输出说明：**

运行后会显示精度校验和性能测试结果：
- **Accuracy Verification**: 精度校验（检查输出是否包含 NaN、全 0、无穷大等异常）
- **Performance Results**: 性能测试结果（延迟 P50/P90、FPS、内存占用等）

---

### 📊 常用测试命令

#### 单模型多框架对比
```bash
# MobileNetV2 - 4线程
./scripts/adb_run.sh --backend all --model mobilenetv2 --threads 4 --runs 50

# ResNet50 - 1线程
./scripts/adb_run.sh --backend all --model resnet50 --threads 1 --runs 50

# MobileViT-S - 8线程
./scripts/adb_run.sh --backend all --model mobilevit_s --threads 8 --runs 30
```

#### 多线程性能扫描
```bash
# 测试 MNN 在不同线程数下的性能
for t in 1 2 4 6 8; do
  echo "=== Testing $t threads ==="
  adb shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=. ./benchmark_inference --backend mnn --model mobilenetv2 --threads $t --runs 50"
done
```

#### 批量测试所有模型
```bash
# 测试所有分类模型
for model in mobilenetv2 resnet50 shufflenet_v2_x0_5 mobilevit_s; do
  for backend in mnn onnxrt; do
    echo "=== $model - $backend ==="
    ./scripts/adb_run.sh --backend $backend --model $model --threads 4 --runs 50
  done
done
```

---

### 🔍 结果解读

**典型输出示例：**
```
========================================================
Model: mobilevit_s
Input shape: 1 3 256 256 
========================================================

--- [Step 1] Getting reference output from ONNX Runtime ---
ONNXRT: loading model: ./models/classification/mobilevit_s/mobilevit_s.onnx
ONNXRT: input 0 = input
ONNXRT: output 0 = output
  ✅ Reference output obtained (1000 elements)
      Reference stats - Min: -207.7330, Max: 13.3748, Mean: -107.6129

--- [Step 2] Running benchmarks ---

>> Testing mnn on mobilevit_s...

--- Accuracy Comparison ---
  ✅ [Accuracy] MNN: PASSED
      Cosine Similarity:  1.000000 (Excellent)
      Mean Absolute Error: 0.000173
      Max Absolute Error:  0.000595
      Mean Relative Error: 0.0002%
      Output range: [-207.7325, 13.3747], Mean: -107.6127

--- Performance Results ---
  Init time:  93.22 ms
  Min:    59.05 ms
  P50:    59.46 ms
  P90:    60.05 ms
  Mean:   59.48 ms
  Throughput: 16.81 FPS
  Peak mem:  0 KB
```

**关键指标说明：**

#### 🎯 精度对比 (Accuracy Comparison)
以 ONNX Runtime 输出为标杆，与其他框架进行精度对比：
- **Cosine Similarity**: 余弦相似度，衡量输出向量方向一致性
  - >0.999: Excellent（优秀）
  - >0.99: Good（良好）
  - >0.95: Acceptable（可接受）
  - <0.95: Poor（较差）
- **Mean Absolute Error**: 平均绝对误差，元素级别差异
- **Max Absolute Error**: 最大绝对误差
- **Mean Relative Error**: 平均相对误差（考虑数值大小）
- **PASSED**: 余弦相似度 > 0.99

#### ⚡ 性能测试 (Performance Results)
- **Init time**: 模型初始化/加载时间（包括内存分配、算子优化）
- **P50/P90/P99**: 延迟百分位数，P50 为中位数延迟
- **Throughput (FPS)**: 每秒可处理的图片数
- **Std**: 延迟标准差，越小表示性能越稳定

---

### ❗ 常见问题排查

#### 问题 1: ADB 找不到设备
```bash
# 检查 Windows 下的设备状态
/mnt/e/andorid/adb/adb.exe kill-server
/mnt/e/andorid/adb/adb.exe start-server
/mnt/e/andorid/adb/adb.exe devices

# 确保手机授权
# 手机上勾选 "一律允许这台计算机进行调试"
```

#### 问题 2: 编译失败 - NDK 路径错误
```bash
# 确认 NDK 路径正确
ls $ANDROID_NDK/build/cmake/android.toolchain.cmake

# 重新配置编译
rm -rf build_android
mkdir build_android && cd build_android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DBENCHMARK_MNN=ON \
  -DBENCHMARK_ORT=ON
make -j$(nproc)
```

#### 问题 3: 运行时找不到模型文件
```bash
# 检查手机上的模型路径
adb shell ls /data/local/tmp/benchmark/models/classification/

# 重新推送
adb push models/classification/mobilevit_s /data/local/tmp/benchmark/models/classification/
```

#### 问题 4: 性能异常低
```
可能原因：
1. 手机发热降频 → 冷却手机后重新测试
2. 线程数设置不当 → 推荐 4 线程使用大核
3. 后台应用占用资源 → 关闭后台应用，开启飞行模式
4. 精度设置 → FP16 比 FP32 快约 2 倍
```

#### 问题 5: 精度校验失败 (Accuracy Check FAILED)
```
可能原因：
1. 模型文件损坏 → 重新转换或下载模型
2. 输入数据异常 → 使用随机输入或正确的预处理
3. 框架版本不兼容 → 检查框架版本和模型格式
4. 模型训练问题 → 检查模型是否正确训练和导出

建议：
- 检查输出统计是否包含 NaN、无穷大或全零
- 对比不同框架的输出范围是否一致
- 使用 ONNX Runtime 作为参考基准进行验证
```

---

### 💡 红米手机优化建议

| 优化项 | 建议值 | 说明 |
|-------|-------|------|
| **线程数** | 4 | 骁龙 865 有 4 个大核（A77） |
| **warmup 次数** | 10-20 | CPU 频率稳定后测试更准确 |
| **测试次数** | 50-100 | 统计结果更稳定 |
| **测试前准备** | 飞行模式 + 清后台 | 减少干扰 |
| **手机温度** | < 40°C | 避免降频影响结果 |

---

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
│   ├── classification/            # 分类模型
│   │   ├── mobilenetv2/           # MobileNetV2 (1.4MB)
│   │   ├── resnet50/              # ResNet50 (98MB)
│   │   ├── shufflenet_v2/         # ShuffleNetV2 x0.5 (5MB)
│   │   └── mobilevit_s/           # MobileViT-S (22MB)
│   ├── detection/                 # 检测模型 (YOLOv8n)
│   └── nlp/                       # NLP 模型
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
  --model <model>         指定模型: mobilenetv2|resnet50|shufflenet_v2_x0_5|mobilevit_s|yolov8n|all
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

### ✅ 已完成
- [x] MobileViT-S 模型支持 (ONNX Runtime + MNN)
- [x] ShuffleNetV2 x0.5 模型支持
- [x] 红米 K30 Pro (骁龙 865) 完整测试验证

### 🚧 进行中
- [ ] TVM AutoTVM 自动调优
- [ ] GPU delegate 支持测试
- [ ] INT8 量化测试对比
- [ ] 更多模型支持 (YOLOv8n)
- [ ] 功耗测试功能
- [ ] 自动生成性能图表
- [ ] LLM 推理 benchmark (llama.cpp)

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
