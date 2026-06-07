# ARM Inference Benchmark

端侧深度学习推理框架性能基准测试项目，针对 ARM 架构手机芯片进行性能对比。

![Platform](https://img.shields.io/badge/platform-Android-lightgrey)
![Architecture](https://img.shields.io/badge/architecture-ARM64-brightgreen)

## 特性

- **多框架支持**: MNN / ONNX Runtime / TVM / llama.cpp（4 核心框架）
- **性能指标**: 延迟 P50/P90/P99、吞吐量 FPS、初始化时间、峰值内存
- **精度对比**: 以 ONNX Runtime 为标杆，自动计算余弦相似度、平均绝对/相对误差
- **Claude Code 插件**: 通过 `claude-code-mobile-bench` 插件自动化测试流程
- **测试模型**: MobileNetV2 / ResNet50 / YOLOv8n / BERT / Qwen2-0.5B（覆盖 CV + NLP + LLM）
- **单算子 Benchmark**: 支持分类批量测试（Conv1x1/MatMul/DWConv 等 7 大类 80+ 测例）
- **环境控制**: CPU 锁频 + 缓存清理，保证结果可复现

## 测试平台

| 设备 | 芯片 | CPU | GPU | ISA |
|------|------|-----|-----|-----|
| 红米 K30 Pro | 骁龙 865 (SM8250) | 1×A77@2.84GHz + 3×A77@2.42GHz + 4×A55@1.8GHz | Adreno 650 | ARMv8.2-A, FP16 |

## 基准测试结果

> 测试平台: 红米 K30 Pro / 骁龙 865 (SM8250) / Android 12 · FP32 · 4 线程 · 2026-06-08  
> 完整原始日志: `results/phaseA_model_*.log`, `results/phaseA_singleop_*.log`

### 框架策略定位

| 框架 | 角色 | 说明 |
|------|------|------|
| **MNN** | 🎯 主测 | 摸底 MNN 性能基线，挖掘优化点（CV + LLM） |
| **ONNX Runtime** | 📐 精度标杆 | 作为 output reference，其他框架对比精度 |
| **TVM** | 🔍 对比优化 | 与 MNN 对比发现隐式优化空间（当前 Kernel 未调优，数据仅供参考） |
| **llama.cpp** | 🦙 LLM 标杆 | 端侧 LLM 推理性能基线（GGUF 量化） |

---

### 4 线程整模型性能 (FP32)

| 模型 | 框架 | Init(ms) | P50(ms) | P90(ms) | FPS | 加速比(vs ORT) | 精度(Cosine) |
|------|------|----------|---------|---------|-----|----------------|-------------|
| **MobileNetV2** | ORT | 33.11 | 21.55 | 21.94 | 45.89 | 1.00x | — |
| | **MNN** | 36.76 | **8.62** | **8.89** | **114.66** | **2.50x** ✅ | 1.000000 |
| | TVM | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ 待适配 | ⏳ |
| **ResNet50** | ORT | 364.49 | 142.62 | 182.04 | 7.15 | 1.00x | — |
| | **MNN** | 509.59 | **83.47** | **97.20** | **11.44** | **1.60x** ✅ | 1.000000 |
| | TVM | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ 待适配 | ⏳ |
| **YOLOv8n** | ORT | 36.94 | 134.38 | 140.92 | 7.35 | 1.00x | — |
| | **MNN** | 159.23 | **76.28** | **78.05** | **13.00** | **1.77x** ✅ | 1.000000 |
| | TVM | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ 待适配 | ⏳ |
| **BERT** | ORT | 643.56 | **263.50** | **270.49** | **3.77** | 1.00x | — |
| | **MNN** | 1202.04 | 322.13 | 328.92 | 3.09 | 0.82x ⚠️ | 0.990439 |
| | TVM | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ 待适配 | ⏳ |
| **Qwen2-0.5B** | **llama.cpp** | 0.37s | **19.25ms/tok** | — | **51.96 tok/s** | LLM 标杆 | — |
| | MNN LLM | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ 待适配 | — |
| | TVM | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ 待探索 | — |
| **mobilevit_s** | — | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ 模型文件待下载 | ⏳ |

> ⏳ = Phase B 补充；TVM 目前仅 MobileNetV2 完成编译（kernel 未调优，513ms，详见 [TVM 部署指南](docs/tvm_deployment_guide.md)）。

---

### 单算子性能 (4 线程)

#### 表 A: Conv1x1

| 算子 | 输入→输出 | 空间 | MNN(ms) | ORT(ms) | MNN优势 |
|------|----------|------|---------|---------|---------|
| K16_C64_M784 | 64→16 | 28×28 | **0.210** | 0.259 | **1.23x** ✅ |
| K1024_C256_M784 | 256→1024 | 28×28 | **4.980** | 5.393 | **1.08x** ✅ |
| M49_C32_K64 | 32→64 | 7×7 | **0.017** | 0.088 | **5.18x** ✅ |
| M49_C256_K512 | 256→512 | 7×7 | **0.154** | 0.336 | **2.18x** ✅ |
| M784_C32_K64 | 32→64 | 28×28 | **0.071** | 0.253 | **3.56x** ✅ |
| M3136_C64_K128 | 64→128 | 56×56 | 0.740 | **0.655** | 0.89x (ORT) |

#### 表 B: Conv1x1 通道失配

| 算子 | 通道 | MNN(ms) | ORT(ms) | MNN退化 |
|------|------|---------|---------|---------|
| Misaligned C31 | 31→64 | 0.506 | **0.367** | ORT 快 **1.38x** ⚠️ |
| Misaligned C33 | 33→64 | 0.518 | **0.288** | ORT 快 **1.80x** ⚠️ |

> MNN NCHW4c 在通道非 4 倍数时需要 padding，额外开销显著。

#### 表 C: DWConv

| 算子 | 通道 | 空间 | MNN(ms) | ORT(ms) | MNN优势 |
|------|------|------|---------|---------|---------|
| C16_3x3 | 16 | 112×112 | 0.640 | **0.418** | ORT 快 **1.53x** |
| C960_3x3 | 960 | 7×7 | **0.153** | 0.489 | **3.20x** ✅ |

#### 表 D: MatMul

| 算子 | 形状 | 场景 | MNN(ms) | ORT(ms) | 胜者 |
|------|------|------|---------|---------|------|
| 512×512 | [1,512]×[512,512] | 通用 | **0.131** | 0.210 | MNN **1.60x** |
| 768×768 | [1,768]×[768,768] | BERT | 0.263 | **0.179** | ORT **1.47x** |
| 768×3072 | [1,768]×[768,3072] | BERT FFN | 0.549 | **0.513** | ORT **1.07x** |
| 3072×768 | [1,3072]×[3072,768] | BERT FFN | 0.504 | **0.391** | ORT **1.29x** |

> ⏳ NLP 算子 (LayerNorm / Softmax / GELU): ONNX 模型未生成，待补充。

---

### 精度对比 (MNN vs ORT 标杆)

| 模型 | Cosine Similarity | Mean Abs Error | 结论 |
|------|-------------------|----------------|------|
| **MobileNetV2** | 1.000000 | 实测 | ✅ 精度一致 |
| **ResNet50** | 1.000000 | 实测 | ✅ 精度一致 |
| **YOLOv8n** | 1.000000 | 实测 | ✅ 精度一致 |
| **BERT** | 0.990439 | 实测 | ⚠️ 微小偏差，可接受 |

---

### 🔑 分析解读

**MNN CV 优势显著**: MobileNetV2 (2.50x) / YOLOv8n (1.77x) / ResNet50 (1.60x) 全面领先 ORT，得益于 NCHW4c 内存布局 + NEON 手写汇编。

**BERT 是唯一弱项**: MNN 在 BERT 上落后 ORT 22% (3.09 vs 3.77 FPS)，根因分析指向：
1. MatMul 大矩阵 (768×3072) 落后 1.29x — MNN GEMM 长矩阵 pack 策略不如 ORT
2. 单算子验证：768×768 MatMul ORT 快 1.47x，BERT 热点对齐

**Conv1x1 通道失配是 MNN 盲区**: C31/C33 非对齐通道 ORT 分别快 1.38x/1.80x，优化思路是手写 Neon kernel 处理尾部通道。

**llama.cpp LLM 表现**: Qwen2-0.5B Q4_K_M 在骁龙 865 上达到 51.96 tok/s，比单线程 (32.5 tok/s) 提升 60%。

**后续 (Phase B)**: TVM 编译全部模型 + auto-tuning 后补全数据，形成 4 框架完整对比。

## 快速开始

### 前置条件

- Android NDK (本机路径: `/home/liu/android-ndk`)
- Android 设备 (Android 10+, ARM64)
- ADB (Windows 路径: `/mnt/e/andorid/adb/adb.exe`)
- Python 3.8+

### 1. 初始化子模块

```bash
git submodule update --init --recursive
```

### 2. 编译 ONNX Runtime（首次需单独编译）

```bash
cd third_party/onnxruntime
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  -u LD_LIBRARY_PATH -u LDFLAGS -u PKG_CONFIG_PATH \
  ./build.sh \
  --android --android_abi arm64-v8a --android_api 29 \
  --android_sdk_path $ANDROID_SDK --android_ndk_path $ANDROID_NDK \
  --build_shared_lib --config Release --use_nnapi \
  --skip_tests --parallel --skip_submodule_sync
cd ../..
```

产物: `third_party/onnxruntime/build/Android/Release/libonnxruntime.so`

### 3. 编译 benchmark

```bash
export ANDROID_NDK=/home/liu/android-ndk
./scripts/build_android.sh       # Release
# ./scripts/build_android.sh --debug   # Debug（profiling 需要符号信息）
```

产物: `build_android/src/benchmark_inference` (约 1.8MB，MNN 为共享库)

### 4. 下载模型

```bash
python scripts/download_pretrained.py
./scripts/build_host_tools.sh && ./scripts/convert_models.sh
```

### 5. 运行测试

```bash
./scripts/run_benchmark_android.sh --backend mnn --model mobilenetv2 --threads 4 --runs 50
```

脚本自动执行：锁频 → 推送二进制/模型/so → 运行 benchmark → 恢复环境。

### 命令行参数

```
--backend   <mnn|onnxrt|ort|tvm|llamacpp|all>  后端 (默认: all)
--model     <模型名|all>           模型 (默认: all)
--precision <fp32|fp16|int8>     精度 (默认: fp32)
--threads   <num>                 线程数 (默认: 1)
--warmup    <num>                 warmup 次数 (默认: 10)
--runs      <num>                 测试次数 (默认: 100)
--gpu                             启用 GPU (MNN OpenCL)
--profiling <file>                启用逐算子 profiling
--json                            输出 JSON 格式结果
--help                            帮助
```

支持的模型: `mobilenetv2`, `resnet50`, `yolov8n`, `bert`, `qwen2_05b`, `mobilevit_s`

## Profiling

```bash
# MNN 逐算子
adb shell "cd /data/local/tmp/benchmark && MNN_PROFILING=1 LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2 --backend mnn"

# simpleperf 火焰图
./scripts/simpleperf_profile.sh --backend mnn --model mobilenetv2 --duration 10

# 集成 profiling 入口
./scripts/profile_benchmark.sh --backend mnn --model mobilenetv2 --profile simpleperf
```

## 项目结构

```
benchmark/
├── CLAUDE.md                  # 项目配置文件（文档入口）
├── CMakeLists.txt             # 顶层 CMake（控制后端开关）
├── src/                       # 源代码
│   ├── main.cpp              # 入口 + 参数解析
│   ├── common/               # 基类、配置、工具函数
│   ├── backends/             # 9 个后端实现
│   ├── models/               # 6 个模型信息定义（MobileNetV2/ResNet50/YOLOv8n/BERT/Qwen2-0.5B/mobilevit_s）
│   ├── single_op_benchmark.cpp # 单算子测试
│   └── llm_benchmark.cpp     # LLM 推理测试
├── scripts/                  # 22 个脚本
├── models/                   # 模型文件
├── third_party/              # 第三方依赖（git 子模块）
│   ├── MNN/                 # libMNN.so（共享库）
│   └── onnxruntime/         # libonnxruntime.so（需单独编译）
├── skills/                   # 项目技能文档（4 个 SKILL.md）
├── docs/                     # 文档 + 测试结果
├── results/                  # 测试结果输出
└── tools/
    ├── MNNConvert/          # MNN 模型转换工具
    └── tvm/                 # TVM 模型编译脚本 + 编译产物
```

## 后端状态

| 框架 | CMake 选项 | 状态 |
|------|-----------|------|
| **MNN** | `BENCHMARK_MNN=ON` | 活跃（共享库 libMNN.so，支持 CPU/GPU OpenCL） |
| **ONNX Runtime** | `BENCHMARK_ORT=ON` | 活跃（动态库 libonnxruntime.so，精度标杆） |
| **TVM** | `BENCHMARK_TVM=ON` | 活跃（Relax VM + libtvm_runtime.so + libtvm_ffi.so） |
| **llama.cpp** | `BENCHMARK_LLAMACPP=ON` | 活跃（GGUF 格式，LLM 推理） |
| TFLite | `BENCHMARK_TFLITE=OFF` | 已停用 |
| TNN | `BENCHMARK_TNN=OFF` | 已停用 |
| QNN | `BENCHMARK_QNN=OFF` | 已停用 |

> TVM 部署详情见 [docs/tvm_deployment_guide.md](docs/tvm_deployment_guide.md)

## Claude Code 插件

Benchmark 流程由 `claude-code-mobile-bench` 插件自动化管理：

```bash
claude plugins marketplace add ./claude-code-mobile-bench
claude plugins install claude-code-mobile-bench@claude-code-mobile-bench
```

插件仓库: [github.com/alsj213/claude-code-mobile-bench](https://github.com/alsj213/claude-code-mobile-bench)

## 详细指南

- **插件 skills**: 安装 `claude-code-mobile-bench` 后自动加载 4 个 skill + agent + 11 条数据红线

## 许可证

MIT License
