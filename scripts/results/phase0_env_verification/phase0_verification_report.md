# Phase 0: 环境就绪验证报告

> **日期:** 2026-06-06 | **分支:** feat/benchmark-full-plan-2026 | **Commit:** af90753

## 1. 设备状态

| 项目 | 值 | 状态 |
|------|-----|------|
| 型号 | M2007J3SC (红米 K30S) | ✅ |
| 芯片 | Qualcomm SM8250 (骁龙 865) | ✅ |
| CPU | 8核 (1×A77@2.84 + 3×A77@2.42 + 4×A55@1.8) | ✅ |
| SDK | 31 (Android 12) | ✅ |
| 温度 | 30.2°C | ✅ 正常 |
| Root | ❌ 无 | ⚠️ 无法锁频/系统级 simpleperf |
| OpenCL | `/system/vendor/lib64/libOpenCL.so` | ✅ 可用 |
| GPU | Adreno 650 (驱动存在) | ✅ 可用 |
| ADB | b08dee23 device | ✅ 已连接 |

## 2. 代码就绪度

### 需要修改的文件（优先级排序）

| 优先级 | 文件 | 修改内容 |
|--------|------|---------|
| P0 | `src/common/config.h` | 添加 `BackendType::MNN_GPU`、`LLAMACPP` |
| P0 | `src/models/model_info.h` | 添加 MobileViT-S、Qwen2-0.5B 模型 |
| P1 | `src/backends/mnn_backend.cpp` | 实现真正的 Callback 式 profiling + OpenCL GPU 路径 |
| P1 | `scripts/run_benchmark_android.sh` | 环境变量透传修复 |
| P1 | `third_party/CMakeLists.txt` | 启用 `MNN_OPENCL=ON` |
| P2 | `CMakeLists.txt` | 启用 `BENCHMARK_LLAMACPP=ON` |
| P2 | `scripts/build_android.sh` | llama.cpp 编译开关 |

### 当前 BackendType 枚举缺失

```cpp
// 当前 (config.h)
enum class BackendType {
    NCNN, MNN, TNN, TFLITE, QNN, ONNXRT, TVM, MINDSPORE_LITE
};

// 需要添加
// MNN_GPU, LLAMACPP
```

## 3. 模型资产

### 已就绪（可直接测试）

| 模型 | MNN | ONNX | 大小 |
|------|-----|------|------|
| MobileNetV2 | ✅ mobilenetv2_MNN.mnn | ✅ mobilenetv2.onnx | 14MB |
| ResNet50 | ✅ resnet50_MNN.mnn | ✅ resnet50.onnx | 98MB |
| YOLOv8n | ✅ yolov8n_MNN.mnn | ✅ yolov8n.onnx | 13MB |
| BERT | ✅ bert_MNN.mnn (418MB) | ✅ bert.onnx (1.1MB) | 见备注 |
| ShuffleNetV2 | ✅ | ✅ | 5.3MB/320KB |

### BERT ONNX 说明

- `bert.onnx` (1.1MB): 小型权重版本（214初始器，494节点）— 可用于端到端测试
- `bert_patched.onnx` (417MB): 完整 FP32 权重版本（212初始器，503节点）
- `bert_MNN.mnn` (418MB): MNN 转换后的完整模型
- 测试时需确认哪个 ONNX 文件被 `model_info` 加载

### 缺失模型

| 模型 | 需要格式 | 用途 |
|------|---------|------|
| MobileViT-S | .onnx → .mnn | Phase 2-3 新计算模式测试 |
| Qwen2-0.5B | .gguf | Phase 5 LLM 推理 |

## 4. 编译系统

| 项目 | 状态 | 操作 |
|------|------|------|
| MNN_OPENCL | OFF (编译时) | Phase 3 启用 |
| BENCHMARK_LLAMACPP | OFF | Phase 5 启用 |
| llama.cpp 子模块 | ✅ 已检出 (9725a31) | — |
| MNN 版本 | 3.5.0 | — |
| `model_registry.h` | ✅ 编译自动生成 | `build_android/src/generated/` |

## 5. 关键发现：MNN Profiling 机制

**⚠️ 重要：计划文档中的 `MNN_PROFILING=1` 环境变量在 MNN 3.5.0 中不存在。**

MNN 的实际 profiling 方式：

```cpp
// 使用 Callback API（TensorCallBackWithInfo）
net_->runSessionWithCallBackInfo(session_, before_callback, after_callback);
```

当前 `mnn_backend.cpp` 中只写了注释和 stub，没有真正实现回调。Phase 1 需要：
1. 在 `mnn_backend.cpp` 中实现 `runSessionWithCallBackInfo` 回调
2. 在回调中记录每个算子的名称和耗时
3. 输出格式：算子名, 耗时(ms), 占比(%)

同时，`scripts/run_benchmark_android.sh` 需要修复环境变量透传：
- 当前 host 设置的 `MNN_PROFILING=1` 不会传递到 `adb shell` 内
- 需要在 adb shell 命令内设置环境变量

### 替代方案

| 方案 | 优势 | 劣势 |
|------|------|------|
| A. MNN Callback API | 精确到算子级 | 需修改 C++ 代码+重新编译 |
| B. simpleperf stat | 无需修改代码 | PMU 级别，不能直接映射算子 |
| C. simpleperf record + 火焰图 | 可视化函数调用 | 需 root（本设备不可用） |
| **D. 优先方案: A+B** | 互补 | Callback 耗时 + PMU 验证 |

## 6. 冒烟测试结果

### MobileNetV2 端到端

| 后端 | 平均耗时 | FPS | vs ORT |
|------|---------|-----|--------|
| MNN CPU | 18.53 ms | 53.98 | **1.57×** |
| ORT CPU | 29.12 ms | 34.34 | 基线 |

**精度验证:** Cosine Similarity = 1.000000 ✅

### 关键问题

| 问题 | 严重度 | 说明 |
|------|--------|------|
| 无 Root 无法锁频 | 🟡 中 | benchmark 结果有波动风险，增加 warmup 次数缓解 |
| 无 Root simpleperf 受限 | 🟢 低 | 应用级 simpleperf 仍可用 |
| BERT ONNX 路径二义性 | 🟢 低 | 需确认测试使用的文件 |
| `MACTHINKING_PROFILING` 不存在 | 🟡 中 | Phase 1 需实现真正的 Callback 式 profiling |

## 7. Phase 0 准入检查清单

- [x] 设备在线、温度正常 (30.2°C)
- [x] ADB 连接稳定 (b08dee23)
- [x] NDK/CMake 版本已验证
- [x] MNN + ORT 冒烟测试通过
- [x] 模型清单已建立
- [x] `model_registry.h` 编译自动生成
- [ ] ~~MNN_PROFILING~~ → 不存在的宏，需改用 Callback API
- [x] simpleperf arm64 二进制存在于 NDK
- [x] Phase 0 环境验证报告完成

**Phase 0 完成 → 可以进入 Phase 1**
