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
| 生成报告 | `python3 scripts/generate_report.py <results_dir>` |
| 验证结果 | `python3 scripts/validate_results.py <results_dir>` |
| 分析单算子 | `python3 scripts/analyze_single_op_results.py <csv_file>` |

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

## 生成 HTML 报告

所有分析结果最终应生成 HTML 格式报告，便于可视化查阅。

```bash
# 将 profiling 分析 Markdown 报告转换为 HTML
pip3 install markdown
python3 -c "
import markdown
md = open('report.md').read()
body = markdown.markdown(md, extensions=['tables'])
html = f'''<!DOCTYPE html>
<html lang=\"zh-CN\">
<head><meta charset=\"utf-8\">
<style>
body {{ max-width: 960px; margin: 0 auto; padding: 24px; font-family: sans-serif; }}
h2 {{ border-left: 4px solid #4a90d9; padding-left: 12px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 12px 16px; border: 1px solid #e0e0e0; text-align: center; }}
th {{ background: #4a90d9; color: #fff; }}
</style></head><body>{body}</body></html>'''
open('report.html', 'w').write(html)
"

# 本地预览（WSL2）
cd results/profiling/<结果目录>
python3 -m http.server 8899
# 浏览器打开 http://localhost:8899/report.html
```

## 数据真实性验证

报告中必须包含以下验证表，确保数据可追溯：

```markdown
## 数据真实性验证

| 验证项 | 结果 | 来源命令 |
|--------|------|---------|
| 设备型号 | 红米 K30 Pro | adb shell getprop |
| 代码版本 | XXXXXXX | git log |
| 二进制 MD5 | XXXXXXX | md5sum |
| 模型文件 | name (XX MB) | ls -lh |
| 原始日志 | XXX lines | wc -l |
| 设备温度 | XX°C | thermal_zone |
| 执行时间 | YYYY-MM-DD HH:MM | date |
```

## 红线

- 不得编造性能数字 — 所有指标必须从 adb/tee 日志提取
- 不得使用旧数据代替新跑
- 不得跳过温度检查
- 温度 > 45°C 时必须在报告中标注降频风险

## 常见问题

- **精度对比为 N/A**: 后端未实现 `infer_with_output()` 或 ORT 标杆输出不可用
- **数据不一致**: 检查是否使用固定种子 42
- **报告为空**: 检查日志文件路径是否正确