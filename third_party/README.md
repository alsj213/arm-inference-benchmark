# 第三方依赖库说明

本项目使用以下开源推理框架，作为 Git Submodules 管理或提供下载脚本。

## 📦 框架列表

| 框架 | 来源 | 方式 | 版本 | 大小 |
|------|------|------|------|------|
| **ncnn** | Tencent/ncnn | Git Submodule | latest | ~230MB |
| **MNN** | alibaba/MNN | Git Submodule | latest | ~540MB |
| **TNN** | Tencent/TNN | Git Submodule | latest | ~700MB |
| **ONNX Runtime** | microsoft/onnxruntime | Git Submodule | v1.21.0 | ~1.3GB |
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
git submodule update --init third_party/onnxruntime
```

#### 2. 编译 ONNX Runtime

ORT 需要从源码编译（参考上面的编译命令），编译完成后才能编译 benchmark 项目。

#### 3. 下载预编译库（TFLite）

```bash
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

Microsoft 开源的跨平台推理引擎，通过 Git Submodule 管理源码，需单独编译。

- 仓库: https://github.com/microsoft/onnxruntime
- 官网: https://onnxruntime.ai/
- 版本: v1.21.0

#### 编译 ONNX Runtime (Android ARM64)

```bash
cd third_party/onnxruntime

# 清理 conda 环境变量（避免交叉编译冲突）
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  -u LD_LIBRARY_PATH -u LDFLAGS -u PKG_CONFIG_PATH \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  ./build.sh \
  --android \
  --android_abi arm64-v8a \
  --android_api 21 \
  --android_sdk_path /home/$USER/Android/Sdk \
  --android_ndk_path /home/$USER/android-ndk \
  --build_shared_lib \
  --config Release \
  --use_nnapi \
  --skip_tests \
  --parallel \
  --skip_submodule_sync
```

编译产物: `build/Android/Release/libonnxruntime.so`

#### 编译参数说明

| 参数 | 说明 |
|------|------|
| `--build_shared_lib` | 构建共享库 libonnxruntime.so |
| `--use_nnapi` | 启用 Android NNAPI 硬件加速 |
| `--skip_tests` | 跳过测试编译（节省时间和内存） |
| `--skip_submodule_sync` | 跳过 ORT 自身的 submodule 同步 |
| `--config Release` | Release 编译（可用 Debug 替换） |

#### 编译 Debug 版本

```bash
./build.sh --android --android_abi arm64-v8a --android_api 21 \
  --android_sdk_path /path/to/sdk --android_ndk_path /path/to/ndk \
  --build_shared_lib --config Debug --use_nnapi --skip_tests --parallel --skip_submodule_sync
```

产物: `build/Android/Debug/libonnxruntime.so`

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
