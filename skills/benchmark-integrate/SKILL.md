---
name: benchmark-integrate
description: Use when adding a new inference framework backend or re-enabling a disabled one — implement backend class, integrate CMake, convert models, validate precision against ORT baseline
---

# Benchmark Integrate

## Overview

将新的推理框架集成到 benchmark 项目中，或恢复已停用的后端。

关键原则：
- 每个框架部署前参考官方教程
- 必须有真实的模型转换流程
- 必须有在设备上的真实测试数据
- 一次只部署一个框架，解决完再部署下一个

## 后端架构

所有后端继承 `BenchmarkBackend`（`src/common/benchmark.h`）：

```
BenchmarkBackend（抽象基类）
├── init(config)              ← 加载模型，初始化推理会话
├── infer(input)              ← 单次推理
├── infer_with_output(input, output)  ← 推理并获取输出（精度对比）
├── deinit()                  ← 释放资源
└── name() const              ← 后端名称
```

工厂函数 `create_backend(BackendType)` 在 `benchmark.cpp` 中，由编译宏控制哪些后端可用。

## 精度对比

以 ONNX Runtime 输出为标杆，计算余弦相似度和平均绝对误差。`AccuracyResult` 结构体包含：

| 字段 | 说明 |
|------|------|
| `passed` | 精度是否通过 |
| `cosine_similarity` | 余弦相似度（接近 1.0 为最佳） |
| `mean_absolute_error` | 平均绝对误差 |
| `max_absolute_error` | 最大绝对误差 |
| `mean_relative_error` | 平均相对误差 |
| `output_min / output_max / output_mean` | 输出统计值 |

**精度合格标准**：余弦相似度 > 0.99

所有后端的输入数据通过 `fill_random_float`（固定种子 42）保证一致。

## 集成步骤

### Step 1: 创建后端文件

在 `src/backends/` 下创建 `xxx_backend.h` 和 `xxx_backend.cpp`。

关键实现要点：
- `init()` 中加载模型、创建推理会话、应用配置（线程数、精度、GPU）
- `infer()` 执行单次推理
- `infer_with_output()` 获取输出张量用于精度对比（若不实现则精度结果为 N/A）
- `deinit()` 释放所有资源
- `name()` 返回后端名称字符串

参考 MNN 后端 (`mnn_backend.h` / `mnn_backend.cpp`) 的实现模式。

### Step 2: CMake 集成

在根 `CMakeLists.txt` 中添加 option：

```cmake
option(BENCHMARK_NEW_FW "Enable NewFW backend" OFF)
```

在 `third_party/CMakeLists.txt` 中添加依赖查找逻辑。

在 `src/CMakeLists.txt` 中添加条件编译和源文件。

### Step 3: 模型转换

```bash
# 将 ONNX 模型转换为目标框架格式
./scripts/convert_models.sh
```

或参考各框架官方工具（如 MNNConvert、onnx2ncnn 等）。

### Step 4: 精度验证

编译并运行，与 ORT 基准对比：

```bash
# 编译
./scripts/build_android.sh

# 推送并运行，检查精度输出
./scripts/run_benchmark_android.sh --backend <new_fw> --model mobilenetv2
```

### Step 5: 恢复已停用后端

恢复步骤：
1. 确保 `third_party/` 中依赖就绪（子模块已检出、SDK 已下载）
2. 根 `CMakeLists.txt` 中将对应 option 改为 ON
3. 重新编译
4. 转换模型
5. 运行测试验证精度

完整指南见 `BACKEND_REENABLE_GUIDE.md`。

## 关键约束

1. 每个框架部署前参考官方教程
2. 每个框架都有真实的模型转换和在设备上真实的测试数据
3. 完全部署好一个框架后再部署下一个，不要交叉部署
4. 停用的后端源代码完整保留，恢复见 `BACKEND_REENABLE_GUIDE.md`
5. 停用后端的旧完整配置备份在 `backup/all-backends` 分支

## 常见问题

- **精度对比为 N/A**: 后端未实现 `infer_with_output()` 或 ORT 标杆输出不可用
- **编译失败**: 检查 CMake 选项和依赖路径
- **模型加载失败**: 确认模型格式与后端兼容
