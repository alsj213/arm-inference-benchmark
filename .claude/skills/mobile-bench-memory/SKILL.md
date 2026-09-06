---
name: mobile-bench-memory
description: Use when analyzing inference memory usage on Android — RSS/PSS reporting, native heap growth over time, and memory leak detection (sustained growth that never returns to baseline)
---

# Mobile Bench Memory

## Overview

深入分析推理过程的内存占用,提供 RSS/PSS、native heap 增长曲线与泄漏检测,超越单一「峰值内存」指标,帮助定位内存泄漏与高水位。

配置通过项目根目录的 `.benchmarkrc.yml` 读取。配合 mobile-bench-run 的完整流程使用。

## 内存指标

| 指标 | 含义 | 来源 |
|------|------|------|
| RSS | 常驻内存(Resident Set Size) | `/proc/<pid>/status` 或 `dumpsys meminfo` |
| PSS | 按共享页均摊的驻留内存 | `dumpsys meminfo` |
| Native heap | native 堆占用 | `dumpsys meminfo` 的 Native Heap 段,或 `mallinfo` |
| 峰值/稳态 | 推理期间最大 / 稳定后内存 | 多次采样取极值与收敛值 |

## 完整流程

```text
定位进程 → 记录初始内存 → 多次推理(每轮采样) → 分析增长曲线 → 泄漏判定 → 报告
```

## 采集方法

### 方法 1:dumpsys meminfo(推荐)

```bash
# 推理前
adb shell dumpsys meminfo <pid> > results/mem_before.txt
# 推理后
adb shell dumpsys meminfo <pid> > results/mem_after.txt
```

包含 RSS、PSS、Native Heap 分区。

### 方法 2:/proc 采样(多次推理,观察增长)

```bash
# 定位 benchmark 进程
PID=$(adb shell pidof benchmark_inference)

# 每轮推理后采样 RSS + VmRSS
for i in $(seq 1 10); do
  adb shell "cat /proc/$PID/status | grep -E 'VmRSS|VmSize'" >> results/mem_curve.txt
  # ... 触发一轮推理 ...
done
```

输出为时间序列,观察 native heap 是否随推理次数增长。

## 泄漏检测

判定规则:连续 N 次推理后 RSS 持续增长且不回落,累计增量 > 10 MB → 标注疑似泄漏。

```text
第1次推理: RSS 500 MB
第5次推理: RSS 560 MB
第10次推理: RSS 620 MB   ← 持续增长不回落,Δ=120MB > 10MB → Possible memory leak
```

**判定标准**:
- 增长后回落 → 正常(峰值波动)
- 持续增长不回落 + 累计 > 10 MB → `Possible memory leak`
- 少量增长但稳定在新水平 → 可能为缓存,需结合 `dumpsys meminfo` 区分(如 Dalvik Heap vs Native Heap)

## 报告模板

```markdown
## 内存分析报告

| 指标 | 峰值 | 稳态 | 来源 |
|------|------|------|------|
| RSS | 620 MB | 500 MB | /proc/<pid>/status |
| PSS | 480 MB | 400 MB | dumpsys meminfo |
| Native heap | 250 MB | 180 MB | dumpsys meminfo |

### 增长曲线

| 推理次数 | RSS(MB) |
|---------|---------|
| 1 | 500 |
| 5 | 560 |
| 10 | 620 |

### 结论

- 泄漏判定: ✅ 疑似泄漏 / ❌ 未发现
- 累计增长: 120 MB(> 10 MB 阈值)
```

## 数据真实性规则

- RSS/PSS/native heap MUST 来自真实采样(`/proc`、`dumpsys meminfo`),标注来源命令
- 峰值/稳态 MUST 基于多次采样,不得单点冒充
- 泄漏判定 MUST 基于「增长 + 不回落」双重条件,不得仅凭峰值高判定泄漏
- 内存增长需区分缓存与泄漏(结合 heap 分区),不得误报

## 常见问题

- **pidof 找不到进程**: benchmark 进程可能已退出(LLM 类进程执行完即退出),改用 profiling 的命令行方式采样,或采集成批内存日志
- **RSS 波动大**: 检查是否与其他进程共享页、malloc 缓存行为
- **无法区分缓存/泄漏**: 看 Native Heap 段是否随轮次单调增长,而非总 RSS

## 关联 Skill

- [mobile-bench-run](../mobile-bench-run/SKILL.md) — 完整基准测试流程,内存采集可嵌入其执行阶段
- [mobile-bench-llm](../mobile-bench-llm/SKILL.md) — LLM 的 KV cache 内存与长上下文内存趋势
- [mobile-bench-methodology](../mobile-bench-methodology/SKILL.md) — 多轮测量与环境控制,内存曲线依赖多次重复测量
- [mobile-bench-profiling](../mobile-bench-profiling/SKILL.md) — 内存泄漏可与 perfetto 内存 trace 结合深入分析
