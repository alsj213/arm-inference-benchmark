---
name: mobile-bench-methodology
description: Use when running benchmark experiments that need statistical rigor — warmup handling, multi-run statistics (mean/CV/P50), full environment control (battery/charging/airplane/screen/WiFi), fair cross-framework comparison, and reproducibility snapshots
---

# Mobile Bench Methodology

## Overview

规范手机端推理 benchmark 的实验设计,保证结果**统计可信、环境可复现、对比公平**。是所有 benchmark 测量(尤其 power/thermal/regression)的环境与统计基础。

配置通过项目根目录的 `.benchmarkrc.yml` 读取,所有路径和参数从配置提取,不硬编码。

## 完整流程

```text
预热 → 环境检查与记录 → 公平配置确认 → 多轮执行(tee日志) → 统计汇总 → 环境快照 → 报告标注
```

## 读取配置

与 run skill 一致,从 `.benchmarkrc.yml` 读取设备与项目配置:

```bash
DEVICE_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['id'])")
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")
```

## 分步流程

### Step 1: 预热与冷启动排除

正式测量前必须预热,首轮冷启动不计入统计。

- **预热次数 ≥ 10 次**(`--warmup 10`),通过 benchmark 二进制参数或 `.benchmarkrc.yml` 配置
- 首轮(含模型加载、会话初始化)耗时 MUST NOT 计入 P50/P90/P99
- 预热不足(< 10)时在报告中标注 `Warmup insufficient (< 10)`

```bash
./scripts/benchmark/benchctl.sh run cnn <model> -f <backend> -w 10 -r 100 2>&1 | tee results/benchmark_$(date +%Y%m%d_%H%M%S).log
```

### Step 2: 环境控制补全

除 run skill 的锁频外,补充以下环境状态检查与记录:

| 项 | 检查命令 | 记录/动作 |
|----|---------|-----------|
| 电量 | `adb shell dumpsys battery \| grep level` | <80% 时提示 `Battery < 80%` |
| 充电状态 | `adb shell dumpsys battery \| grep status` | 充电中禁止采集或标注 `Charging throttling risk` |
| 飞行模式 | `adb shell settings get global airplane_mode_on` | 建议开启以排除网络干扰 |
| 屏幕 | `adb shell dumpsys power \| grep -i screen` | 记录亮/灭状态(影响温控与 GPU) |
| WiFi | `adb shell dumpsys wifi \| grep mNetworkInfo` | 记录状态 |
| 锁频 | `./scripts/setup/setup-test-env.sh` | 沿用 run skill 的锁频流程 |

充电状态对性能影响显著(触发充电限频),是结果失真的常见隐藏因素,必须检查。

### Step 3: 公平对比协议

跨框架对比时,保证以下配置对齐并记录:

1. **线程数对齐** — 所有后端使用相同 `--threads`
2. **精度对齐** — 所有后端使用相同 `--precision`(或各自最优精度时明确标注)
3. **输入一致** — 固定种子输入(`fill_random_float` seed 42),确保输入张量相同
4. **版本锁定** — 记录各框架 commit/版本、模型哈希、二进制 MD5
5. **加速器审计** — 确认真实使用了声明加速器(CPU/GPU/NPU),若配置声明 NPU 但日志显示未启用,标注 `Accelerator mismatch`

```bash
# 记录框架版本与加速器
git submodule status
adb shell dumpsys gfxinfo <pkg> 2>/dev/null  # 或根据日志确认执行后端
```

### Step 4: 多轮执行与统计汇总

- **正式测量 ≥ 5 轮**(可多次调用 benchctl,或单日志含多轮),每轮 tee 保存日志
- 用 `stats_summary.py` 汇总多轮统计

```bash
python3 scripts/analyze/mobilebench/stats_summary.py results/*.log
# 可选:调整方差告警阈值
python3 scripts/analyze/mobilebench/stats_summary.py --cv-threshold 10 results/*.log
# 结构化输出(供回归分析)
python3 scripts/analyze/mobilebench/stats_summary.py --json results/*.log > results/stats.json
```

**统计输出**:mean、std、CV、P50/P90/P99(线性插值百分位)。

**方差判定**:
- CV > 5% → `High variance (CV=xx%)` 告警,提示检查降频/后台进程/环境控制
- 样本 < 5 → `only N run(s); spec requires >= 5` 提示,数据仅供参考

### Step 5: 可复现性清单(环境快照)

生成结构化环境快照,每字段带来源命令,供复现与回归对比:

```bash
python3 scripts/analyze/mobilebench/env_snapshot.py --config .benchmarkrc.yml
# 输出 results/env_snapshot_<ts>.json
```

快照字段:设备型号/OS/kernel、git commit/branch、框架 submodule 版本、二进制 MD5 与大小、模型文件 MD5 与大小、温度。

**失败语义**:任何字段采集失败记录为 `{value: null, source: <cmd>, error: <原因>}`,不得静默吞掉,也不得用估计值冒充实测值。

## 报告模板

报告中必须包含统计与可复现性信息:

```markdown
## 统计结果

| 配置 | n | P50 mean(ms) | CV% | 判定 |
|------|---|--------------|-----|------|
| mnn/mobilenetv2/fp32/4t | 5 | 8.40 | 1.0 | ok |

## 环境快照

| 字段 | 值 | 来源 |
|------|----|------|
| 设备型号 | xxx | adb shell getprop |
| 代码版本 | xxxxx | git log |
| 二进制 MD5 | xxxxx | md5sum |
| 温度 | 38°C | thermal_zone |

## 方差说明

- CV > 5%: 结果方差高,数据可能受环境因素影响,建议检查后重跑
```

## 红线

- 首轮冷启动 MUST NOT 计入统计
- 充电状态下 MUST 标注 `Charging throttling risk`,不得静默采集
- 样本不足 5 轮 MUST 标注,不得冒充正式统计
- 环境字段采集失败 MUST 记录 error,不得伪造或估算
- 跨框架对比 MUST 对齐线程/精度/输入,版本 MUST 锁定
- 声明使用 NPU/GPU 但未真实启用 MUST 标注 `Accelerator mismatch`

## 常见问题

- **CV 居高不下**: 检查充电状态、锁频是否生效、后台进程、温度是否 >40°C
- **env_snapshot 设备字段全 error**: 确认 `adb devices` 在线、`.benchmarkrc.yml` 的 device.id 正确
- **统计只有 1 个样本**: 检查日志是否包含多轮,或需要多次调用 benchmark 再汇总
- **框架版本为空**: 确认 benchmark 项目是 git 仓库且框架为 submodule 管理

## 关联 Skill

- [mobile-bench-run](../mobile-bench-run/SKILL.md) — 完整基准测试流程(锁频、tee、报告),本 skill 是其统计与公平性增强
- [mobile-bench-power](../mobile-bench-power/SKILL.md) — 功耗与能效测量,依赖本 skill 的环境控制
- [mobile-bench-thermal](../mobile-bench-thermal/SKILL.md) — 热稳定性/Soak test,依赖本 skill 的环境控制与多轮统计
- [mobile-bench-regression](../mobile-bench-regression/SKILL.md) — 性能回归检测,复用本 skill 的 env_snapshot 与多轮统计
