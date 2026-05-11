---
name: result-processor
description: Use when benchmarks or profiling are complete — parse logs, compare precision with cosine similarity, generate Markdown reports, validate results
---

# Result Processor

## Overview

处理 benchmark 结果：解析日志、精度对比（以 ONNX Runtime 为标杆）、生成报告。

## 精度对比机制

- **标杆**: ONNX Runtime 输出
- **指标**: 余弦相似度（方向一致性，1.0=完全一致）、平均绝对误差（MAE）
- **输入一致**: `fill_random_float` 使用固定种子 42，保证所有后端输入相同

## 快速参考

| 操作 | 命令 |
|------|------|
| 生成报告 | `python scripts/generate_report.py -i <log> -o <output>` |
| 验证结果 | `python scripts/validate_results.py -i <results>` |
| 分析单算子 | `python scripts/analyze_single_op_results.py -i <csv>` |

## 输出指标

| 指标 | 说明 |
|------|------|
| P50 | 中位数延迟（ms），核心指标 |
| P90 | 90% 尾部延迟 |
| P99 | 99% 尾部延迟 |
| FPS | 每秒推理帧数 |
| 内存峰值 | 推理过程最大内存占用 |
| 余弦相似度 | 与 ORT 标杆的精度一致性 |

## 工作流

```bash
python scripts/generate_report.py \
  -i results/sm8250/benchmark_20260501.log \
  -o results/final_report.md

python scripts/validate_results.py -i results/final_report.md
```

## 常见问题

- **精度对比为 N/A**: 后端未实现 `infer_with_output()` 或 ORT 标杆输出不可用
- **数据不一致**: 检查是否使用固定种子 42
- **报告为空**: 检查日志文件路径是否正确