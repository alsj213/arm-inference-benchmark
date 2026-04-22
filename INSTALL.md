# 安装构建指南

本指南将帮助您在本地环境中构建和运行此项目。

## 📋 系统要求

### Linux / WSL2 (推荐)
- Ubuntu 20.04+ / Debian 11+
- GCC 9+ 或 Clang 12+
- CMake 3.18+
- Android NDK r25c+ (用于 Android 构建)
- Python 3.8+ (用于模型转换)

### macOS
- macOS 12+
- Xcode Command Line Tools
- Homebrew

## 🔧 安装依赖

### Ubuntu / WSL2

```bash
# 安装基础依赖
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip

# 安装 Android NDK (推荐 r25c)
wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip
unzip android-ndk-r25c-linux.zip
export ANDROID_NDK=$(pwd)/android-ndk-r25c
```

### macOS

```bash
# 使用 Homebrew 安装
brew install cmake git python3

# 安装 Android NDK
wget https://dl.google.com/android/repository/android-ndk-r25c-darwin.zip
unzip android-ndk-r25c-darwin.zip
export ANDROID_NDK=$(pwd)/android-ndk-r25c
```

## 📦 获取源代码

```bash
# 克隆项目
git clone https://github.com/your-org/arm-inference-benchmark.git
cd arm-inference-benchmark
```

## ⚙️ 设置第三方依赖

### 方式一：一键设置 (推荐)

```bash
# 安装所有依赖
./scripts/setup_deps.sh --all

# 或者最小化安装（仅 ncnn + MNN）
./scripts/setup_deps.sh --minimal
```

### 方式二：手动设置

```bash
# 1. 初始化 Git Submodules
git submodule update --init --recursive

# 2. 下载预编译库
# ONNX Runtime
./scripts/download_onnxruntime.sh

# TensorFlow Lite
./scripts/download_tflite.sh
```

### 方式三：选择特定框架

```bash
# 只安装 ncnn
./scripts/setup_deps.sh --ncnn

# 只安装 ONNX Runtime
./scripts/setup_deps.sh --ort
```

> **注意**: Apache TVM 体积较大 (~1GB)，如果不需要可以跳过。

详细说明请参考 [third_party/README.md](third_party/README.md)

## 🏗️ 构建

### Android ARM64 构建 (推荐)

```bash
# 设置 NDK 路径
export ANDROID_NDK=/path/to/your/android-ndk-r25c

# 使用脚本构建
./scripts/build_android.sh

# 或者手动构建
mkdir -p build_android && cd build_android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBENCHMARK_NCNN=ON \
    -DBENCHMARK_MNN=ON \
    -DBENCHMARK_TFLITE=ON \
    -DBENCHMARK_ORT=ON \
    -DBENCHMARK_TVM=ON

make -j$(nproc)
```

### 本机构建 (用于开发调试)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CMake 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BENCHMARK_NCNN` | ON | 启用 ncnn 后端 |
| `BENCHMARK_MNN` | ON | 启用 MNN 后端 |
| `BENCHMARK_TNN` | OFF | 启用 TNN 后端 |
| `BENCHMARK_TFLITE` | ON | 启用 TFLite 后端 |
| `BENCHMARK_QNN` | OFF | 启用 QNN 后端 |
| `BENCHMARK_ORT` | ON | 启用 ONNX Runtime 后端 |
| `BENCHMARK_TVM` | ON | 启用 TVM 后端 |

## 📱 准备模型

```bash
# 下载预训练模型
cd models
python3 download_pretrained.py

# 转换模型（需要）
../scripts/convert_models.sh
```

## 🚀 运行测试

### 连接 Android 设备

```bash
# 检查设备连接
adb devices

# 确保设备已连接
# 输出应该显示设备序列号
```

### 推送并运行

```bash
# 使用脚本自动推送和运行
./scripts/adb_run.sh --backend all --model mobilenetv2 --runs 100

# 或者手动操作
cd build_android
adb push src/benchmark_inference /data/local/tmp/
adb push ../models/classification /data/local/tmp/models/classification
adb shell "cd /data/local/tmp && chmod +x benchmark_inference && ./benchmark_inference --backend mnn --model mobilenetv2"
```

### 常用运行参数

```bash
# 测试单个框架
./benchmark_inference --backend mnn --model mobilenetv2

# 测试所有框架
./benchmark_inference --backend all --model mobilenetv2

# 测试不同线程数
for t in 1 2 4 8; do
    ./benchmark_inference --backend mnn --model mobilenetv2 --threads $t
done

# 更长时间的测试
./benchmark_inference --backend all --model mobilenetv2 --warmup 20 --runs 500
```

## 🔍 故障排除

### CMake 找不到 NDK

```bash
# 确保设置了正确的环境变量
export ANDROID_NDK=/absolute/path/to/android-ndk-r25c
echo $ANDROID_NDK

# 确保路径有效
ls $ANDROID_NDK/build/cmake/android.toolchain.cmake
```

### adb 找不到设备

```bash
# 检查 adb 版本
adb version

# 重启 adb 服务
adb kill-server && adb start-server

# 检查手机是否开启了 USB 调试
# 尝试更换 USB 线或 USB 口
```

### 编译错误

```bash
# 清理并重新构建
rm -rf build_android
./scripts/build_android.sh

# 检查 submodule 是否完整
git submodule update --init --recursive
```

### 运行时找不到动态库

```bash
# 确保推送了所有 .so 文件
LD_LIBRARY_PATH=/data/local/tmp ./benchmark_inference
```

## 📞 获取帮助

如果遇到问题：

1. 查看 [FAQ](docs/FAQ.md) (如果有)
2. 搜索 [Issues](https://github.com/your-org/arm-inference-benchmark/issues)
3. 提交新 Issue
