---
name: mobile-bench-thermal
description: Use when assessing thermal stability and throttling on Android devices — soak test (15-30 min), time-vs-performance throttling curve, and stability index, similar to AI Benchmark's throttling test
---

# Mobile Bench Thermal

## Overview

评估设备在持续负载下的**热稳定性**,量化降频对性能的影响。仿 AI Benchmark 的 throttling 测试思路:持续压力负载 + 周期重测,输出降频曲线与稳定性指数,而不是只检查单点温度阈值。

配置通过项目根目录的 `.benchmarkrc.yml` 读取。Soak test 前先做 mobile-bench-methodology 的环境控制(锁频、电量、充电检查)。

## 指标说明

| 指标 | 含义 |
|------|------|
| 降频曲线 | 时间-性能序列:每轮 FPS/P50 与温度随时间的演化 |
| 降频点 | 性能较首轮下降超过 10% 的时间点与对应温度 |
| 稳定性指数 | 末尾区间平均性能 / 起始区间平均性能(%) |

## Soak Test 流程

### Step 1: 环境控制(复用 methodology)

```bash
./scripts/setup/setup-test-env.sh   # 锁频、清缓存
python3 scripts/analyze/mobilebench/env_snapshot.py       # 记录初始环境快照(含初始温度)
```

### Step 2: 持续压力 + 周期采集

**一次 Soak 采集循环同时记录性能、温度、功耗**,不得为 thermal 和 power 分别跑两遍 Soak(重复加热设备会污染数据):

```bash
# 每轮: 记录温度 → 跑一轮 benchmark → 记录性能/功耗
# 每 5 分钟一轮,共 15–30 分钟(4–6 轮)
```

每轮采集内容,输出为结构化时间序列 `results/soak_<ts>.json`:

```json
{
  "timestamp": "2026-08-09 23:10:00",
  "elapsed_min": 0,
  "temp_c": 38,
  "p50_ms": 8.4,
  "fps": 114.9,
  "power_w": 2.0
}
```

### Step 3: 计算稳定性指数与降频点

```bash
python3 scripts/analyze/mobilebench/stats_summary.py --json results/soak_*.log > results/soak_stats.json
```

稳定性指数 = 末尾两轮平均 FPS / 起始两轮平均 FPS × 100%。

| 指数 | 等级 | 判定 |
|------|------|------|
| ≥ 0.95 | 正常 | 散热良好,数据可信 |
| 0.90–0.95 | 轻微降频 | 部分负载下性能下降 |
| < 0.90 | **降频显著** | MUST 标注,建议冷却后重测或换散热方案 |

## 报告模板

```markdown
## 热稳定性报告

### Soak Test 记录

| 轮次 | 用时(min) | 温度(°C) | P50(ms) | FPS | 功耗(W) |
|------|-----------|----------|---------|-----|---------|
| 1 | 0 | 38 | 8.4 | 114.9 | 2.0 |
| 2 | 5 | 42 | 8.6 | 112.0 | 2.1 |
| ... | ... | ... | ... | ... | ... |

### 结论

- 稳定性指数: xx%(等级: 正常 / 轻微降频 / 降频显著)
- 降频点: 第 N 轮, 温度 xx°C(如无则标注"未观察到明显降频")
- 散热等级: 优 / 良 / 差
```

## 数据真实性规则

- 温度、性能 MUST 来自真实采集(adb shell / tee 日志),每项标注来源
- 降频点判定 MUST 基于首轮对比,不得凭主观
- 稳定性指数 MUST 明确标注计算区间(起始/末尾各几轮)
- 若设备在 Soak 中途过热崩溃或断电,MUST 如实报告,不得用中断数据冒充完整 Soak

## 关联 Skill

- [mobile-bench-methodology](../mobile-bench-methodology/SKILL.md) — 环境控制与多轮统计是 Soak 的基础
- [mobile-bench-power](../mobile-bench-power/SKILL.md) — Soak 期间可同步采集功耗,分析功耗-升温-降频链路
- [mobile-bench-run](../mobile-bench-run/SKILL.md) — 完整基准测试流程
