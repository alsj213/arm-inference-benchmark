# Changelog

本项目的所有重要变更将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### ✨ 新增
- 5 个推理框架完整集成：ncnn, MNN, TFLite, ONNX Runtime, TVM
- 支持 MobileNetV2 分类模型测试
- 完整的性能指标统计：延迟（P50/P90/P99）、吞吐量、初始化时间、内存占用
- 统一的后端抽象接口，易于扩展
- Android ARM64 交叉编译支持
- 自动化测试脚本

### 🔧 改进
- **Debug/Release 构建类型切换支持**
  - `build_android.sh`: 支持 `--debug` / `--release` 标志
  - `run_benchmark_android.sh`: 支持 `--build-type` 参数
  - `simpleperf_profile.sh`: 默认使用 Debug 版本生成火焰图
  - `profile_benchmark.sh`: 默认使用 Release 版本性能测试
- **测试基础设施增强**
  - `setup_test_environment.sh`: 预测试环境设置（CPU 调度器、nice 值）
  - `restore_test_environment.sh`: 测试后环境恢复
  - `run_comprehensive_benchmark.sh`: 自动化综合测试脚本
  - `generate_report.py`: 自动生成 Markdown 测试报告
  - `validate_results.py`: 结果验证脚本
- **脚本重命名**: `adb_run.sh` → `run_benchmark_android.sh`（更清晰的命名）
- **统计鲁棒性**: 新增异常值移除和置信区间计算功能

### 📊 已完成测试
- **骁龙 865 (SM8250)** 平台 MobileNetV2 FP32 性能对比
- 单线程推理性能基准

### 🚀 性能结果

| 框架 | Mean (ms) | FPS |
|------|-----------|-----|
| MNN | 18.64 | 53.65 |
| ncnn | 19.57 | 51.11 |
| TFLite | 22.69 | 44.07 |
| ONNX Runtime | 29.41 | 34.00 |
| TVM (基线) | 39.40 | 25.38 |

### 📝 计划中
- [ ] TNN 框架集成
- [ ] Qualcomm QNN SDK 集成
- [ ] TVM AutoTVM 自动调优
- [ ] GPU delegate 测试
- [ ] INT8 量化支持
- [ ] YOLOv8n 检测模型
- [ ] BERT NLP 模型
- [ ] 功耗测试功能

---

## [0.1.0] - 2026-04-22

### ✨ 新增
- 项目初始版本发布
- 基准测试框架核心实现
- 5 个推理框架后端
- MobileNetV2 模型支持
- 骁龙 865 完整测试结果

---

## 版本说明

### 版本号格式
`主版本号.次版本号.修订号`

- **主版本号**: 不兼容的 API 变更
- **次版本号**: 向后兼容的功能性新增
- **修订号**: 向后兼容的问题修正

### 版本类型
- `Unreleased`: 开发中的版本
- 已发布的版本: 带版本号和发布日期
