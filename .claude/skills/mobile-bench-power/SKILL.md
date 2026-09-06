---
name: mobile-bench-power
description: Use when measuring energy efficiency of inference on Android devices — energy per inference (J), perf/Watt, and power measurement via Perfetto power rails / dumpsys batterystats / battery-delta fallback
---

# Mobile Bench Power

## Overview

测量手机端推理的**功耗与能效**指标:每推理能耗(J)、perf/Watt、平均功率。对齐 MLPerf Power 思路,让结果不止看性能,也看能效。

配置通过项目根目录的 `.benchmarkrc.yml` 读取;正式测量前建议先做 mobile-bench-methodology 的环境控制(电量、充电、温度),否则功耗数据失真。

## 指标说明

| 指标 | 含义 | 计算公式 |
|------|------|---------|
| 每推理能耗 | 单次推理消耗能量 | 总能耗(J) / 推理次数 |
| 平均功率 | 测试窗口内平均功耗 | 总能耗(J) / 时长(s) |
| perf/Watt | 能效(每瓦吞吐) | 吞吐(FPS) / 平均功率(W) |

## 三级测量路径

按优先级尝试,**只有真实测量可用才报告数字**;全部不可用输出 `Power data unavailable`,禁止伪造或估算冒充实测。

### 路径 1:Perfetto power rails(优先)

Android 10+ 内置,可读取各电源轨(CPU/GPU/内存)能耗。

```bash
# 采集 perfetto trace(需较新 Android)
adb shell perfetto --time 60s -o /data/local/tmp/power.pb -c - 2>&1 <<'EOF' &
write_interval_ms: 1000
data_sources { config { name: "android.power" } }
EOF
# 基准测试期间采集;结束后拉回并导出能耗 JSON(如 energy_j / power_rails)
adb pull /data/local/tmp/power.pb .
```

将能耗数据导出为 JSON(含 `energy_j` 或 `power_rails` 数组),再分析:

```bash
python3 scripts/analyze/mobilebench/power_analysis.py --source perfetto --input power.json \
  --inference-count 100 --duration-s 60 --throughput-fps 114.94
```

### 路径 2:dumpsys batterystats(中)

```bash
# 测试前重置统计
adb shell dumpsys batterystats --reset
# ... 执行 benchmark ...
# 测试后导出统计
adb shell dumpsys batterystats > results/batterystats.txt
```

```bash
python3 scripts/analyze/mobilebench/power_analysis.py --source batterystats --input results/batterystats.txt \
  --inference-count 100 --duration-s 60 --throughput-fps 114.94 --voltage 3.85
```

batterystats 输出的是「预估功耗(mAh)」,脚本按电压换算为能量(J)。

### 路径 3:电量差估算(降级,误差最大)

无 root、无 batterystats 时,用测试前后电量差估算:

```bash
# 记录测试前后电量百分比与电池容量(如 4500 mAh)
adb shell dumpsys battery | grep level
```

```bash
python3 scripts/analyze/mobilebench/power_analysis.py --source battery-delta \
  --start-pct 80 --end-pct 79 --battery-capacity-mah 4500 \
  --inference-count 100 --duration-s 60 --throughput-fps 114.94 --voltage 3.85
```

> 电量差法误差较大(受温度、屏幕、后台影响),报告中 MUST 标注使用路径 3 及误差风险。

## 数据真实性规则

- 功耗数字 MUST 来自真实测量,每项标注来源(路径 1/2/3 + 命令)
- 路径不可用时 MUST 输出 `Power data unavailable`,不得编造
- 电量差估算 MUST 标注为估算级,不得与实测混为一谈
- 测试时设备 MUST 处于充电控制状态(见 methodology:充电会严重干扰功耗测量)

## 关联 Skill

- [mobile-bench-methodology](../mobile-bench-methodology/SKILL.md) — 环境控制(电量/充电/温度)是功耗测量的前提
- [mobile-bench-run](../mobile-bench-run/SKILL.md) — 完整基准测试流程,功耗测量与性能测量共用同一测试窗口
- [mobile-bench-thermal](../mobile-bench-thermal/SKILL.md) — 热稳定性与功耗密切相关(高功耗 → 升温 → 降频)
