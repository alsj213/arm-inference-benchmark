# ARM Inference Benchmark

端侧深度学习推理框架性能基准测试项目，针对 ARM 架构手机芯片进行性能对比。

![Platform](https://img.shields.io/badge/platform-Android-lightgrey)
![Architecture](https://img.shields.io/badge/architecture-ARM64-brightgreen)

## 特性

- **多框架支持**: MNN / ONNX Runtime / TVM / llama.cpp（4 核心框架）
- **性能指标**: 延迟 P50/P90/P99、吞吐量 FPS、初始化时间、峰值内存
- **精度对比**: 以 ONNX Runtime 为标杆，自动计算余弦相似度、平均绝对/相对误差
- **Claude Code 插件**: 通过 `claude-code-mobile-bench` 插件自动化测试流程
- **测试模型**: MobileNetV2 / ResNet50 / YOLOv8n / BERT / Qwen2-0.5B / mobilevit_s（覆盖 CV + NLP + LLM）
- **单算子 Benchmark**: 支持分类批量测试（Conv1x1/MatMul/DWConv 等 7 大类 80+ 测例）
- **环境控制**: CPU 锁频 + 缓存清理，保证结果可复现

## 测试平台

| 设备 | 芯片 | CPU | GPU | ISA |
|------|------|-----|-----|-----|
| 红米 K30 Pro | 骁龙 865 (SM8250) | 1×A77@2.84GHz + 3×A77@2.42GHz + 4×A55@1.8GHz | Adreno 650 | ARMv8.2-A, FP16 |

## 基准测试结果

> 测试平台: 红米 K30 Pro / 骁龙 865 (SM8250) / Android 12 · FP32 · 4 线程 · 2026-06-09  
> TVM: v0.15.0 Relay + GraphExecutor（v0.24 Relax VM → 降级）  
> 完整原始日志: `results/phaseA_model_*.log`, `results/phaseA_singleop_*.log`

### 框架策略定位

| 框架 | 角色 | 说明 |
|------|------|------|
| **MNN** | 🎯 主测 | 摸底 MNN 性能基线，挖掘优化点（CV + LLM） |
| **ONNX Runtime** | 📐 精度标杆 | 作为 output reference，其他框架对比精度 |
| **TVM** | 🔍 对比优化 | v0.15 Relay + GraphExecutor（降级自 v0.24 Relax VM）；AutoTVM/AutoScheduler RPC 调优就绪 |
| **llama.cpp** | 🦙 LLM 标杆 | 端侧 LLM 推理性能基线（GGUF 量化） |

---

### 4 线程整模型性能 (FP32)

| 模型 | 框架 | Init(ms) | P50(ms) | P90(ms) | FPS | 加速比(vs ORT) | 精度(Cosine) |
|------|------|----------|---------|---------|-----|----------------|-------------|
| **MobileNetV2** | ORT | 33.11 | 21.55 | 21.94 | 45.89 | 1.00x | — |
| | **MNN** | 36.76 | **8.62** | **8.89** | **114.66** | **2.50x** ✅ | 1.000000 |
| | TVM | 17.64 | 10.01 | 10.59 | 97.91 | **2.15x** ✅ | 1.000000 |
| **ResNet50** | ORT | 364.49 | 142.62 | 182.04 | 7.15 | 1.00x | — |
| | **MNN** | 509.59 | **83.47** | **97.20** | **11.44** | **1.60x** ✅ | 1.000000 |
| | TVM | 250.94 | **64.38** | **65.64** | **15.30** | **2.22x** ✅🏆 | 1.000000 |
| **YOLOv8n** | ORT | 36.94 | 134.38 | 140.92 | 7.35 | 1.00x | — |
| | **MNN** | 159.23 | **76.28** | **78.05** | **13.00** | **1.77x** ✅ | 1.000000 |
| | TVM | 20.92 | 126.63 | 143.45 | 7.67 | 1.06x | 1.000000 |
| **BERT** | ORT | 643.56 | **263.50** | **270.49** | **3.77** | 1.00x | — |
| | **MNN** | 1202.04 | 322.13 | 328.92 | 3.09 | 0.82x ⚠️ | 0.990439 |
| | TVM | — | — | — | — | ⏳ 已编译(417MB), 待 bench | — |
| **Qwen2-0.5B** | **llama.cpp** | 0.35 | 70.07 | 42.54 | **42.54 tok/s** | LLM 标杆 | — |
| | **MNN LLM** | 0.61 | 267.09 | **60.84** | **60.84 tok/s** | **1.43x** 🏆 | — |
| | TVM | — | — | — | — | ⸺ 后期评估 | — |
| **mobilevit_s** | ORT | 98.57 | 92.64 | 97.01 | 10.67 | 1.00x | — |
| | **MNN** | 92.77 | **59.95** | **61.83** | **16.50** | **1.55x** ✅ | 1.000000 |
| | TVM | — | — | — | — | ⸺ v0.15 ONNX frontend 不兼容 | — |

> **TVM v0.15 Relay 对比 v0.24 Relax VM**: MobileNetV2 514ms→10ms (28x 加速), YOLOv8n 4406ms→127ms (35x)。Relax VM 的运行时类型分发/分支/memcpy 对小 CV 模型致命。ResNet50 TVM (64ms) 以 2.2x 超越 ORT (143ms) 🏆，为三框架中最快。
>
> **LLM 行列说明**: Qwen2-0.5B 的 Init = 模型加载时间(s), P50 = prefill pp128 (tok/s), P90 = decode tg128 (tok/s), FPS = decode 速度 (tok/s)。

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

#### 表 E: NLP 算子 (BERT 热点)

| 算子 | 形状 | 场景 | MNN(ms) | ORT(ms) | 胜者 |
|------|------|------|---------|---------|------|
| LayerNorm | [1,128,768] | BERT hidden | **0.182** | 0.305 | MNN **1.68x** |
| Softmax | [1,128,128] | BERT attention | **0.085** | 0.217 | MNN **2.55x** |
| GELU | [1,128,3072] | BERT FFN | **1.748** | 2.144 | MNN **1.23x** |

> NLP 单算子全部生成自 PyTorch → ONNX (opset 18)，BERT 真实形状。MNN 在逐元素/规约类算子全面领先 ORT。

#### TVM 单算子 (1 线程, v0.15 Relay, 未调优)

| 类别 | 算子 | ORT(ms) | TVM(ms) | TVM 加速比 | 精度 |
|------|------|---------|---------|-----------|------|
| Conv1x1 | K16_C64_M784 | 0.07 | 0.10 | 0.70x | 1.000 |
| | K1024_C256_M784 | 13.23 | **5.77** | **2.29x** ✅ | 1.000 |
| | M49_C32_K64 | 0.09 | — | — | 待测 |
| | M49_C256_K512 | 0.34 | — | — | 待测 |
| | M784_C32_K64 | 0.25 | — | — | 待测 |
| | M3136_C64_K128 | 0.66 | — | — | 待测 |
| Misaligned | C31→64 | 0.37 | — | — | 待测 |
| | C33→64 | 0.29 | — | — | 待测 |
| DWConv | C16_3x3 | 0.42 | — | — | 待测 |
| | C960_3x3 | 0.39 | **0.18** | **2.17x** ✅ | 1.000 |
| MatMul | 512×512 | 0.04 | **0.03** | **1.33x** ✅ | 1.000 |
| | 768×768 | 0.18 | — | — | 待测 |
| | 768×3072 | 0.43 | **0.33** | **1.30x** ✅ | 1.000 |
| | 3072×768 | 0.39 | — | — | 待测 |
| NLP | LayerNorm | 0.31 | — | — | 待测 |
| | Softmax | 0.22 | — | — | 待测 |
| | GELU | 2.14 | — | — | 待测 |

> **最新三框架完整对比**: 41 算子 ORT/MNN/TVM 全量数据 → [`results/single_op_benchmark_2026-06-09.md`](results/single_op_benchmark_2026-06-09.md)<br>
> 几何平均: MNN/ORT=1.61x, TVM/ORT=1.32x, TVM/MNN=0.82x。TVM 16/41 最快, MNN 22/41 最快。

---

### 精度对比

| 模型 | TVM vs ORT | MNN vs ORT | 结论 |
|------|-----------|-----------|------|
| **MobileNetV2** | 1.000000 | 1.000000 | ✅ 三方完美一致 |
| **ResNet50** | 1.000000 | 1.000000 | ✅ 三方完美一致 |
| **YOLOv8n** | 1.000000 | 1.000000 | ✅ 三方完美一致 |
| **BERT** | 1.000000 | 0.990439 | ⚠️ MNN vs ORT 微小偏差 |

---

### 🔑 分析解读

**MNN CV 优势显著**: MobileNetV2 (2.50x) / YOLOv8n (1.77x) / ResNet50 (1.60x) 全面领先 ORT，得益于 NCHW4c 内存布局 + NEON 手写汇编。

**MobileViT-S (CV+Transformer 混合)**: MNN 仍以 1.55x 领先 ORT。mobilevit_s 含 LayerNorm/MatMul/Transpose 等 Transformer 算子 + Conv 混合，是 MNN 在混合架构上的首次验证。MNN 对 Conv 的 NCHW4c 优化仍发挥作用，但 Transformer 算子是纯 MatMul/Self-Attention，收益有限。

**BERT 弱项根因已定位**: MNN 在 BERT 上落后 ORT 22% (3.09 vs 3.77 FPS)。NLP 单算子拆解揭示：
1. **逐元素/激活算子 MNN 全面领先**: LayerNorm 1.68x / Softmax 2.55x / GELU 1.23x
2. **MatMul 是唯一短板**: 768×768 落后 1.47x / 768×3072 落后 1.07x / 3072×768 落后 1.29x
3. **结论**: BERT 的 MatMul 热点消耗 ~90% 总时间，MNN 的 GEMM pack 策略不如 ORT，逐元素优势被淹没

**Conv1x1 通道失配是 MNN 盲区**: C31/C33 非对齐通道 ORT 分别快 1.38x/1.80x，优化思路是手写 Neon kernel 处理尾部通道。

**LLM 三框架实测**: 以 llama.cpp 为 LLM 标杆 (Q4_K_M, decode 42.54 tok/s)。MNN LLM (HQQ 4-bit) 实测 prefill 267.09 / decode 60.84 tok/s，相对标杆 decode 加速 1.43x。MNN LLM 的 prefill 快 3.8x（可能受益于 HQQ 量化格式在 prompt 批处理上的优化），decode 快 1.43x。MNN LLM 体积 295 MB vs llama.cpp 374 MB (省 21%)。TVM Qwen2 → MLC-LLM 独立工具链，与项目 TVM 子模块冲突，后期评估。

**TVM v0.15 Relay 性能质变**: 降级到 v0.15 后，MobileNetV2 从 514ms→10ms (28x)，ResNet50 从 4738ms→64ms (74x)，YOLOv8n 从 4406ms→127ms (35x)。根因: v0.24 Relax VM 的运行时类型分发/分支/memcpy 对小 CV 模型造成致命开销，v0.15 Relay GraphExecutor 生成静态形状 AOT 专用代码消除此瓶颈。ResNet50 TVM (64ms) 超越 ORT (143ms) 成为三框架最快。

**TVM AutoTVM/AutoScheduler RPC 调优就绪**: RPC tracker + device server + ADB 双通道已验证可用。AutoTVM 单算子实测: Conv1x1 0.07→0.06ms (1.65x FPS)。AutoScheduler 集成完成。全模型调优建议在原生 Linux 上过夜运行（WSL2 NDK 编译瓶颈 ~10h）。

**mobilevit_s**: v0.15 ONNX frontend 的 Reshape 类型推导与 mobilevit_s 不兼容，待 TVM 更新或模型修改。BERT 已编译(417MB)，待 bench。

**分支策略**: `feat/tvm-integration` (v0.24 Relax VM) 保留不动，`feat/tvm-stable-relay` (v0.15 Relay + GraphExecutor) 为主力分支。

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
./scripts/build/build-android.sh       # Release
# ./scripts/build/build-android.sh --debug   # Debug（profiling 需要符号信息）
```

产物: `build_android/src/benchmark_inference` (约 1.8MB，MNN 为共享库)

### 4. 下载模型

```bash
python scripts/convert/download-pretrained.py
./scripts/build/build-host-tools.sh && ./scripts/convert/convert-models.sh
```

### 5. 运行测试

```bash
./scripts/benchmark/run-android.sh --backend mnn --model mobilenetv2 --threads 4 --runs 50
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
./scripts/profile/simpleperf-profile.sh --backend mnn --model mobilenetv2 --duration 10

# 集成 profiling 入口
./scripts/profile/profile-benchmark.sh --backend mnn --model mobilenetv2 --profile simpleperf
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
├── scripts/                  # 脚本（按功能分组到 7 个子目录）
├── models/                   # 模型文件（source + exported + single_ops + llm）
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

> TVM 部署详情见 [docs/01-guides/tvm-deployment-guide.md](docs/01-guides/tvm-deployment-guide.md)

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
