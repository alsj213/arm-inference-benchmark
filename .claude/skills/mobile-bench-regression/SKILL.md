---
name: mobile-bench-regression
description: Use when detecting performance regressions across code, model, or framework changes — baseline snapshot, cross-commit comparison with statistical significance testing, and regression threshold alerts
---

# Mobile Bench Regression

## Overview

检测代码、模型或框架版本变更导致的**性能回归**。通过基线快照(baseline.json)、跨 commit 对比与显著性检验,让性能退化被自动发现并告警,而不是靠人工盯数字。

配置通过项目根目录的 `.benchmarkrc.yml` 读取。依赖 mobile-bench-methodology 的多轮统计与 env_snapshot 环境字段。

## 完整流程

```text
基线测量(多轮) → 保存 baseline.json → 变更后复测(多轮)
→ 对比(baseline vs 当前) → 显著性检验 → 回归告警
```

## 前置条件

- 基线与当前测量均 ≥ 5 轮(`--runs` + 多次,或单日志多轮),否则无法做显著性检验
- 建议安装 scipy(显著性检验用),不可用时脚本自动回退描述性对比并提示
- 环境一致(同一设备、相近温度),环境不一致时结果仅供参考

## 保存基线

基线在**代码/模型变更前**测量:

```bash
python3 scripts/analyze/mobilebench/regression_compare.py save \
  --results results/base/*.log \
  --env results/env_snapshot_*.json \
  --out results/baseline.json
```

baseline.json 记录:创建时间、git commit、环境字段(设备/OS/commit)、每个配置的多轮 P50 样本。

## 变更后对比

代码或模型变更后,同配置复测:

```bash
python3 scripts/analyze/mobilebench/regression_compare.py compare \
  --baseline results/baseline.json \
  --results results/current/*.log \
  --env results/env_snapshot_*.json
```

### 输出判定

| 判定 | 条件 |
|------|------|
| **REGRESSION DETECTED** | P50 恶化 > 5% 且显著(或无法检验时仅描述性超过阈值) |
| 显著 | p < 0.05(Mann-Whitney U,或 `--paired` 配对 t-test) |
| 不显著 | 差异未达统计显著 |
| NEW | 基线中不存在的配置 |

### 环境差异标注

对比时自动比对基线/当前的设备型号、OS、commit;不一致时输出 `⚠️ 环境差异`,提示结果可能失真(如设备或 OS 变了,Δ 不能归因于代码变更)。

## 参数

```bash
# 配对 t-test(逐轮对应)而非 Mann-Whitney U
python3 scripts/analyze/mobilebench/regression_compare.py compare --baseline results/baseline.json \
  --results results/current/*.log --paired

# JSON 输出(供 CI 或后续分析)
python3 scripts/analyze/mobilebench/regression_compare.py compare --baseline results/baseline.json \
  --results results/current/*.log --json
```

## CI 集成

`--compare` 发现回归时以退出码 2 结束,便于接入 CI:

```bash
python3 scripts/analyze/mobilebench/regression_compare.py compare --baseline baseline.json --results logs/ || echo "回归告警"
```

## 数据真实性规则

- 基线 MUST 在变更前测量,当前 MUST 在变更后复测,不得复用旧会话数据(R11 延伸)
- 两测 MUST 均为 ≥ 5 轮的真实测量,来源标注完整
- 显著性检验结果 MUST 如实报告(scipy 不可用时标注"未检验",不得伪装成已检验)
- 环境不一致 MUST 标注,不得归因于代码变更
- 回归阈值(默认 5%)可按需调整,但 MUST 在报告中注明

## 常见问题

- **所有配置都是 NEW**: 当前日志解析不到基线里的配置键,检查 `--results` 是否覆盖全部
- **提示 scipy 未安装**: `pip install scipy` 后重跑,才有显著性检验
- **无法做检验**: 增加轮次到 ≥ 5,再对比
- **环境差异告警**: 确认用同一设备、相近温度与 OS 版本

## 关联 Skill

- [mobile-bench-methodology](../mobile-bench-methodology/SKILL.md) — 多轮统计、env_snapshot、环境控制是本 skill 的基础
- [mobile-bench-run](../mobile-bench-run/SKILL.md) — 完整基准测试流程
- [mobile-bench-thermal](../mobile-bench-thermal/SKILL.md) — 温度差异会导致回归误报,必要时先做热稳定性评估
