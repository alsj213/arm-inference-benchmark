# Phase B: TVM 编译 + 实测 汇总报告

**执行时间**: 2026-06-08  
**设备**: 红米 K30 Pro (M2007J3SC) / 骁龙 865 (SM8250) / Android 12  
**代码版本**: `feat/tvm-integration` 分支  
**编译环境**: TVM 0.24.dev0 + Python 3.13 + torch 2.11.0 + onnx 1.21.0  
**NDK**: aarch64-linux-android21-clang++  

---

## 1. TVM 编译成果

### 编译脚本

`tools/tvm/compile_all_models.py` — 支持双路径:
- **路径 A (torch.export)**: torchvision 模型 → `torch.export.export()` → `from_exported_program()` → Relax → .so
- **路径 B (ONNX import)**: ONNX → `from_onnx()` → Relax → .so

Target: `cortex-a77` + `aarch64-linux-android` + NEON

### 编译产物

| 模型 | 路径 | .so 大小 | 编译时间 | 状态 |
|------|------|---------|---------|------|
| **mobilenetv2** | TorchVision | 15.0 MB | ~2 min | ✅ |
| **resnet50** | ONNX | 98.4 MB | ~5 min | ✅ |
| **yolov8n** | ONNX | 13.2 MB | ~2 min | ✅ |
| **bert** | ONNX (bert_patched) | 417.2 MB | ~8 min | ✅ (运行时多输入待适配) |
| **Conv1x1 ×6** | ONNX | 0.03–1.03 MB | <10s each | ✅ |
| **Conv1x1 Misaligned ×2** | ONNX | 0.03 MB | <10s each | ✅ |
| **DWConv ×2** | ONNX | 0.03–0.06 MB | <10s each | ✅ |
| **MatMul ×4** | ONNX | 1.0–9.0 MB | <20s each | ✅ |

总计: **17 个 .so 文件**, ~541 MB

---

## 2. 整模型 Benchmark 结果

**测试条件**: 4 线程, LD_PRELOAD=libtvm_ffi.so:libtvm_runtime.so, 100 轮

| 模型 | TVM P50(ms) | FPS | MNN P50(ms) | MNN FPS | ORT P50(ms) | TVM vs MNN | TVM vs ORT | 精度 |
|------|:----------:|:---:|:----------:|:------:|:----------:|:----------:|:----------:|:----:|
| **MobileNetV2** | 514.02 | 1.95 | 8.62 | 114.66 | 21.55 | 59.6x 慢 | 23.8x 慢 | 1.000 ✅ |
| **ResNet50** | 4738.52 | 0.21 | 83.47 | 11.44 | 142.62 | 56.8x 慢 | 33.2x 慢 | 1.000 ✅ |
| **YOLOv8n** | 4406.27 | 0.23 | 76.28 | 13.00 | 134.38 | 57.8x 慢 | 32.8x 慢 | 1.000 ✅ |
| **BERT** | — | — | 322.13 | 3.09 | 263.50 | ❌ 多输入 | ❌ 多输入 | — |

### 精度评估

- MobileNetV2: Cosine **1.000000**, MAE 0.000002 (完美)
- ResNet50: Cosine **1.000000**, MAE 0.000001 (完美 — ONNX 路径，与 ORT 参考输出完全一致)
- YOLOv8n: Cosine **1.000000**, MAE 0.000002 (完美 — ONNX 路径)
- ResNet50 torch.export 路径精度异常: Cosine=0.66 (权重初始值不一致，已替换为 ONNX 路径)

---

## 3. 单算子编译结果

14 个单算子全部编译成功，总计 23.2 MB:

| Category | 编译数 | 状态 |
|----------|:-----:|------|
| Conv1x1 | 6 | ✅ |
| Conv1x1 Misaligned | 2 | ✅ |
| DWConv | 2 | ✅ |
| MatMul | 4 | ✅ |
| **总计** | **14** | **✅** |

> 单算子 TVM Benchmark 未执行: 需要修改 `single_op_benchmark.cpp` 支持 TVM .so 路径，或通过独立 `benchmark_inference` 命令行测试。当前已写编译脚本可复现。

---

## 4. 关键发现

### 4.1 未调优性能基线

TVM Relax 在 0 trial 下性能约为 MNN 的 1/60、ORT 的 1/30。这是完全预期的:
- Relax 使用默认调度模板，无任何手工优化
- MNN 使用 NCHW4c + Neon 手写汇编
- ORT 使用高度优化的 GEMM/Conv kernel

### 4.2 精度完美

ONNX → Relax 路径精度无损失（MobileNetV2/YOLOv8n/ResNet50 全部 Cos=1.0），验证了 Relax 编译正确性。

### 4.3 BERT 多输入问题

Relax VM `set_input` 无法正确处理多输入函数。编译正常但运行时:
```
expect a Tensor but get int  (indexed set_input)
args.size() == params_num (1 vs. 2)  (repeated set_input without index)
```

### 4.4 ResNet50 编译教训

torch.export 路径 (torchvision pretrained weights) 与 ONNX 模型 (从 PyTorch exporter 导出) 的权重初始化不同，导致精度不匹配。**统一用 ONNX 路径可保证一致性。**

### 4.5 编译流程稳定

从 ONNX → Relax → NDK 交叉编译 → .so → 设备加载的全链路已打通，可用于后续 auto-tuning 实验。

---

## 5. 环境依赖

### Python 环境

```bash
export TVM_ROOT=third_party/tvm
export PYTHONPATH=$TVM_ROOT/python:$TVM_ROOT/3rdparty/tvm-ffi/python
export LD_LIBRARY_PATH=$TVM_ROOT/build:$TVM_ROOT/build/lib
python3 tools/tvm/compile_all_models.py --all        # 全量编译
python3 tools/tvm/compile_all_models.py --single-ops  # 单算子
```

### 设备运行

```bash
export LD_PRELOAD=libtvm_ffi.so:libtvm_runtime.so   # 必需！符号解析依赖
./benchmark_inference --backend tvm --model <name> --threads 4
```

### Android 运行时依赖

- `libtvm_ffi.so` (24 MB, from `build-android-v2/lib/`)
- `libtvm_runtime.so` (64 MB, from `build-android-v2/`)
- `libc++_shared.so` (NDK, Android 系统内置)

---

## 6. 后续优化方向

| 优先级 | 方向 | 预期提升 | 难度 |
|:------:|------|:-------:|:---:|
| P0 | TVM auto-tuning (MetaSchedule) | 5-10x | 中 — 需要显著编译时间和调优数据库 |
| P1 | BERT 多输入适配 | 功能修复 | 低 — 修改 tvm_backend.cpp set_input 调用 |
| P2 | 单算子 TVM 测试 | 数据补全 | 低 — 修改 single_op_benchmark.cpp |
| P3 | 专用 kernel 库集成 | 2-5x | 高 — 需要手写 schedule/custom pass |
| P4 | Qwen2-0.5B Relax 编译 | 探索性 | 很高 — Transformer 模型编译管线复杂 |

---

## 7. 数据溯源

| 验证项 | 结果 | 来源 |
|--------|------|------|
| 设备型号 | M2007J3SC | `adb shell getprop ro.product.model` |
| TVM 版本 | 0.24.dev0 | `tvm.__version__` |
| TVM FFI | 0.1.10 (editable install) | `pip show apache-tvm-ffi` |
| NDK 编译器 | aarch64-linux-android21-clang++ | NDK r25c |
| LLVM 版本 | 15.0.7 | TVM 编译输出 |
| MobileNetV2 数据 | `adb shell` 直接采集 | `phaseB_tvm_model.log` |
| ResNet50 数据 | `adb shell` 直接采集 | `phaseB_tvm_model.log` |
| YOLOv8n 数据 | `adb shell` 直接采集 | `phaseB_tvm_model.log` |
