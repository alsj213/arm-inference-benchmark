# 第三方依赖库说明

本项目使用以下开源推理框架，作为 Git Submodules 管理或提供下载脚本。

## 📦 框架列表

| 框架 | 来源 | 方式 | 版本 | 大小 |
|------|------|------|------|------|
| **ncnn** | Tencent/ncnn | Git Submodule | latest | ~230MB |
| **MNN** | alibaba/MNN | Git Submodule | latest | ~540MB |
| **TNN** | Tencent/TNN | Git Submodule | latest | ~700MB |
| **ONNX Runtime** | microsoft/onnxruntime | 预编译下载 | 1.16.x | ~50MB |
| **TensorFlow Lite** | tensorflow/tensorflow | AAR 下载 | 2.15.0 | ~20MB |
| **Apache TVM** | apache/tvm | Git Submodule | latest | ~1GB |
| **QNN** | Qualcomm | 官网下载 | 2.2x+ | - |

## 🚀 快速设置

### 方式一：一键设置所有依赖（推荐）

```bash
# 在项目根目录执行
./scripts/setup_deps.sh
```

### 方式二：手动设置

#### 1. 初始化所有 Git Submodules

```bash
# 初始化子模块
git submodule update --init --recursive

# 或者只初始化需要的模块
git submodule update --init third_party/ncnn
git submodule update --init third_party/MNN
git submodule update --init third_party/TNN
git submodule update --init third_party/tvm
```

#### 2. 下载预编译库

```bash
# ONNX Runtime
./scripts/download_onnxruntime.sh

# TensorFlow Lite
./scripts/download_tflite.sh
```

#### 3. QNN SDK（可选）

需要从 Qualcomm Developer Network 下载：
1. 访问 https://qpm.qualcomm.com/
2. 搜索 "Qualcomm AI Engine Direct"
3. 下载并解压到 `third_party/QNN/`

## 🔧 子模块管理

### 添加新子模块

```bash
git submodule add https://github.com/Tencent/ncnn.git third_party/ncnn
git submodule add https://github.com/alibaba/MNN.git third_party/MNN
git submodule add https://github.com/Tencent/TNN.git third_party/TNN
git submodule add https://github.com/apache/tvm.git third_party/tvm
```

### 更新子模块

```bash
# 更新所有子模块
git submodule update --remote --merge

# 更新特定子模块
cd third_party/ncnn && git pull origin master
```

### 克隆仓库时获取子模块

```bash
git clone --recursive https://github.com/your-org/arm-inference-benchmark.git

# 或者克隆后再初始化
git clone https://github.com/your-org/arm-inference-benchmark.git
cd arm-inference-benchmark
git submodule update --init --recursive
```

## 📝 各框架说明

### ncnn

腾讯开源的高性能神经网络推理框架，专为手机端优化。

- 仓库: https://github.com/Tencent/ncnn
- 特性: NEON 汇编优化、Vulkan GPU 支持、量化支持

### MNN

阿里巴巴开源的端侧推理引擎，支持多种硬件和模型格式。

- 仓库: https://github.com/alibaba/MNN
- 特性: 轻量级、跨平台、支持 GPU、多种后端

### TNN

字节跳动开源的推理框架，由 Rapidnet 发展而来。

- 仓库: https://github.com/Tencent/TNN
- 特性: 多平台、高性能、模型压缩

### ONNX Runtime

Microsoft 开源的跨平台推理引擎。

- 官网: https://onnxruntime.ai/
- 下载: https://github.com/microsoft/onnxruntime/releases

### TensorFlow Lite

Google 的移动端推理框架。

- 官网: https://www.tensorflow.org/lite
- 直接从 Maven 下载 AAR 包提取

### Apache TVM

深度学习编译器栈，自动生成优化的内核代码。

- 官网: https://tvm.apache.org/
- 仓库: https://github.com/apache/tvm
- 需要从源码编译 Runtime

## ⚠️ 注意事项

1. **不要提交第三方库源码到主仓库**，使用 Git Submodule 管理
2. **不要提交构建产物**，所有 build*/ 目录都已在 .gitignore 中
3. **大文件不要提交**，使用下载脚本在构建时获取
4. **定期更新子模块**，使用稳定版本

## 🔍 故障排除

### Submodule 连接问题

```bash
# 如果 submodule 连接失败，可以手动克隆
rm -rf third_party/ncnn
git clone https://github.com/Tencent/ncnn.git third_party/ncnn
```

### 子模块更新失败

```bash
# 重置子模块
git submodule deinit -f third_party/ncnn
rm -rf .git/modules/third_party/ncnn
git submodule update --init third_party/ncnn
```
