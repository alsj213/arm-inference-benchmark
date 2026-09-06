---
name: mobile-bench-run
description: Use when running inference performance tests on Android devices — full workflow from device handshake to result report, including environment control, ADB operations, and data validation
---

# Mobile Bench Run

## Overview

手机端推理 benchmark 完整流程：设备握手 → 版本检查 → 二进制验证 → 模型检查 → 锁频 → 执行(tee日志) → 恢复环境 → 结果解析 → 报告生成。

配置通过项目根目录的 `.benchmarkrc.yml` 读取（设备 ID、ADB 路径等），所有路径和参数均从该配置文件中提取，无需硬编码。

## 完整流程

```text
Step 1: 设备握手 → Step 2: 代码版本 → Step 3: 二进制验证 → Step 4: 模型检查
→ Step 5: 环境控制(锁频) → Step 6: 执行(tee日志) → Step 7: 恢复环境
→ 结果解析 → 报告生成
```

## 读取配置

所有设备相关的配置（ADB 路径、设备 ID、Android NDK 路径等）统一从项目根目录的 `.benchmarkrc.yml` 读取：

```bash
# 从 .benchmarkrc.yml 读取设备配置
python3 -c "
import yaml
with open('.benchmarkrc.yml') as f:
    cfg = yaml.safe_load(f)
print('Device:', cfg['device']['id'])
print('ADB:',   cfg['device']['adb'])
"
```

常用配置读取快捷方式：

```bash
DEVICE_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['id'])")
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")
ANDROID_NDK=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['ndk']['path'])")
```

## 分步流程

### Step 1: 设备握手

验证设备在线并记录设备信息：

```bash
DEVICE_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['id'])")
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")

# 检查连接
$ADB devices | grep "$DEVICE_ID" || exit 1

# 记录设备信息
$ADB shell getprop ro.product.model
$ADB shell getprop ro.board.platform

# 检测温度
$ADB shell cat /sys/class/thermal/thermal_zone*/temp
```
> 温度 > 45°C 时需标注降频风险。

### Step 2: 检查代码版本

```bash
git log --oneline -1
```

### Step 3: 检查/编译二进制

```bash
ls -lh build_android/src/benchmark_inference  # 验证产物
md5sum build_android/src/benchmark_inference   # 记录 MD5

# 如需重新编译，使用编译脚本
./scripts/build/build-android.sh
```

### Step 4: 检查模型文件

```bash
ls -lh models/classification/mobilenetv2/mobilenetv2.onnx
# 缺失则执行模型准备 flow，详见 mobile-bench-model-prep skill
```

### Step 5: 环境控制

锁频、清缓存，保证测试环境一致性。

```bash
# 自动检测 root 权限，有 root 则设置 performance governor 并清缓存
./scripts/setup/setup-test-env.sh
```

**有 root 时：** CPU performance governor、清理缓存（`echo 3 > /proc/sys/vm/drop_caches`）、记录初始频率温度、停止 zygote

**无 root 时：** 跳过硬件控制，仅记录状态，设置进程优先级（nice）

### Step 6: 执行测试

推送并运行，使用 `tee` 保留原始日志：

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")

# 推送二进制、动态库、模型
$ADB push build_android/src/benchmark_inference /data/local/tmp/benchmark/
$ADB push build_android/third_party/*.so /data/local/tmp/benchmark/
$ADB push models/ /data/local/tmp/benchmark/

# 执行测试（必须使用 tee 保存日志）
./scripts/benchmark/benchctl.sh run cnn mobilenetv2 -f mnn,ort,tvm,llamacpp 2>&1 | tee results/latest_benchmark.log
```

**重要：** 记录二进制 md5、模型文件大小和设备温度。

#### 命令行参数 (benchmark_inference)

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--model` | 模型名 | mobilenetv2, resnet50, yolov8n, bert, qwen2_0.5b |
| `--backend` | 后端类型 | mnn, onnxrt, ncnn, tvm, mslite, llamacpp |
| `--precision` | 推理精度 | fp32, fp16, int8, q8_0, q4_k_m |
| `--threads` | 线程数 | 1, 2, 4 |
| `--runs` | 运行次数 | 100 |
| `--warmup` | 预热次数 | 10 |
| `--gpu` | 使用 GPU | 无参数 |
| `--profiling <file>` | 启用逐算子 profiling | profiling.json |

### Step 7: 恢复环境

```bash
./scripts/setup/restore-test-env.sh
```

有 root 时恢复 schedutil governor 并重启 zygote。无 root 时仅记录状态。

## 设备操作参考

### ADB 常用命令

| 操作 | 命令 |
|------|------|
| 检查连接 | `adb devices` |
| 验证设备信息 | `adb shell getprop ro.product.model && adb shell getprop ro.board.platform` |
| 检测温度 | `adb shell cat /sys/class/thermal/thermal_zone*/temp` |
| 推送文件 | `adb push <local> <remote>` |
| 拉取结果 | `adb pull <remote> <local>` |
| 远程执行 | `adb shell "<command>"` |
| 设置权限 | `adb shell chmod +x <file>` |

> ADB 路径和设备 ID 从 `.benchmarkrc.yml` 读取，使用 `python3 -c "import yaml; ..."` 获取。

### 温度说明

- **< 40°C**: 正常，数据可信
- **40-45°C**: 可能轻微降频
- **> 45°C**: 降频风险高，建议冷却后重跑

## 结果处理

### 精度对比机制

- **标杆**: ONNX Runtime (ORT) 输出
- **指标**: 余弦相似度（方向一致性，1.0=完全一致）、平均绝对误差（MAE）
- **输入一致**: `fill_random_float` 使用固定种子 42，保证所有后端输入相同

### 生成报告

两种方式：

**方式 A：使用 generate_report.py（快速，自动解析 results/ 下所有日志）**
```bash
python3 scripts/analyze/mobilebench/generate_report.py results/
```

**方式 B：使用 HTML 报告模板（含跨后端对比 + 排名 + 精度验证）**

执行 benchmark 后，用以下脚本从最新日志生成完整的 HTML 报告。该脚本自动解析日志，生成含跨后端对比表、性能排名、精度验证的 HTML：

```python
#!/usr/bin/env python3
"""从 benchmark 日志生成完整 HTML 报告。"""
import re, os, sys, glob, json
from datetime import datetime

def parse_results(log_path):
    """解析 benchmark 日志，返回结构化结果列表。"""
    results = []
    with open(log_path) as f:
        content = f.read()
    # 每段测试以 "Running: backend=X model=Y" 分隔
    blocks = re.split(r'={2,}\nRunning:\s*', content)
    for block in blocks:
        if not block.strip():
            continue
        r = {'backend': None, 'model': None, 'precision': 'fp32', 'threads': 4,
             'init_ms': 0, 'p50': 0, 'p90': 0, 'p99': 0, 'fps': 0, 'cosine': None,
             'mae': None, 'max_err': None, 'status': 'PASS'}
        m = re.search(r'backend=(\S+)', block)
        if m: r['backend'] = m.group(1)
        m = re.search(r'model=(\S+)', block)
        if m: r['model'] = m.group(1)

        def extract(pattern, text, group=1, cast=float):
            m = re.search(pattern, text)
            return cast(m.group(group)) if m else None

        r['init_ms'] = extract(r'Init time:\s*([\d.]+)\s*ms', block) or 0
        r['p50'] = extract(r'P50:\s*([\d.]+)\s*ms', block) or 0
        r['p90'] = extract(r'P90:\s*([\d.]+)\s*ms', block) or 0
        r['p99'] = extract(r'P99:\s*([\d.]+)\s*ms', block) or 0
        r['fps'] = extract(r'Throughput:\s*([\d.]+)\s*FPS', block) or 0
        r['cosine'] = extract(r'Cosine Similarity:\s*([\d.]+)', block)
        r['mae'] = extract(r'Mean Absolute Error:\s*([\d.]+)', block)
        r['max_err'] = extract(r'Max Absolute Error:\s*([\d.]+)', block)
        if 'Failed to initialize' in block or 'Build failed' in block:
            r['status'] = 'FAIL'
        if r['backend'] and r['model']:
            results.append(r)
    return results

def gen_html(results, log_path, output_path):
    """生成完整 HTML 报告。"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    passed = sum(1 for r in results if r['status'] == 'PASS' and r['fps'] > 0)
    total = len(results)
    models = sorted(set(r['model'] for r in results))
    backends = sorted(set(r['backend'] for r in results))

    # 跨后端对比表
    cross_rows = ''
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        best_fps = max((r['fps'] for r in model_results), default=0)
        for r in model_results:
            is_best = r['fps'] >= best_fps and r['fps'] > 0
            fps_display = f'{r["fps"]:.2f}' if r['fps'] > 0 else '0.00'
            cls = ' class="best"' if is_best else ''
            status_tag = '<span class="pass">PASS</span>' if r['status'] == 'PASS' else '<span class="fail">FAIL</span>'
            cosine = f'{r["cosine"]:.6f}' if r['cosine'] else '-'
            cross_rows += f'''    <tr>
      <td{cls}>{r["backend"].upper()}</td>
      <td class="left">{model}</td>
      <td class="num">{r["init_ms"]:.2f}</td>
      <td class="num{cls}">{r["p50"]:.2f}</td>
      <td class="num{cls}">{r["p90"]:.2f}</td>
      <td class="num{cls}">{r["p99"]:.2f}</td>
      <td class="num{cls}">{fps_display}</td>
      <td class="num">{cosine}</td>
      <td>{status_tag}</td>
    </tr>
'''
        cross_rows += '    <tr><td colspan="9" style="background:#f5f7fa;padding:4px;"></td></tr>\n'

    # 排名表
    rank_rows = ''
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        sorted_r = sorted([r for r in model_results if r['fps'] > 0], key=lambda x: -x['fps'])
        medals = ['gold', 'silver', 'bronze']
        cells = ''
        for i, r in enumerate(sorted_r[:3]):
            label = f'{r["backend"].upper()}: {r["fps"]:.2f}'
            cells += f'      <td><span class="badge badge-{medals[i]}">{label}</span></td>\n'
        for _ in range(3 - len(sorted_r)):
            cells += '      <td><span class="fail">FAIL</span></td>\n'
        rank_rows += f'''    <tr>
      <td><strong>{model}</strong></td>
{cells}    </tr>
'''

    # 后端详情表
    backend_tables = ''
    for backend in backends:
        backend_results = [r for r in results if r['backend'] == backend]
        rows = ''
        for r in backend_results:
            cosine = f'{r["cosine"]:.6f}' if r['cosine'] else '-'
            status_tag = '<span class="pass">PASS</span>' if r['status'] == 'PASS' else '<span class="fail">FAIL</span>'
            rows += f'    <tr><td>{r["model"]}</td><td class="num">{r["init_ms"]:.2f}</td><td class="num">{r["p50"]:.2f}</td><td class="num">{r["p90"]:.2f}</td><td class="num">{r["p99"]:.2f}</td><td class="num">{r["fps"]:.2f}</td><td class="num">{cosine}</td><td>{status_tag}</td></tr>\n'
        backend_tables += f'''
<h3>{backend.upper()}</h3>
<table>
  <thead><tr><th>模型</th><th>Init(ms)</th><th>P50(ms)</th><th>P90(ms)</th><th>P99(ms)</th><th>FPS</th><th>Cosine Sim</th><th>Status</th></tr></thead>
  <tbody>{rows}</tbody>
</table>'''

    # 精度表
    acc_rows = ''
    for r in results:
        if r['cosine'] and r['status'] == 'PASS':
            mae = f'{r["mae"]:.6f}' if r['mae'] else '-'
            max_err = f'{r["max_err"]:.6f}' if r['max_err'] else '-'
            grade = 'Excellent' if r['cosine'] >= 0.99 else 'Good'
            acc_rows += f'    <tr><td>{r["backend"].upper()}</td><td>{r["model"]}</td><td class="num">{r["cosine"]:.6f}</td><td class="num">{mae}</td><td class="num">{max_err}</td><td><span class="pass">{grade}</span></td></tr>\n'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>端侧推理框架性能基准测试报告</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#333;background:#f5f7fa;padding:24px;}}
  .container{{max-width:1280px;margin:0 auto;}}
  h1{{color:#1a1a2e;margin-bottom:8px;font-size:28px;}}
  h2{{color:#16213e;margin:32px 0 16px;padding-bottom:8px;border-bottom:2px solid #e0e0e0;}}
  table{{width:100%;border-collapse:collapse;margin-bottom:24px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
  th,td{{padding:10px 14px;text-align:center;border-bottom:1px solid #eee;}}
  th{{background:#1a1a2e;color:#fff;font-weight:600;font-size:13px;text-transform:uppercase;white-space:nowrap;}}
  tr:hover{{background:#f0f4ff;}}
  td.num{{text-align:right;font-family:'SF Mono','Fira Code',monospace;font-size:13px;}}
  td.left{{text-align:left;}}
  .pass{{color:#155724;background:#d4edda;font-weight:600;border-radius:4px;padding:2px 8px;}}
  .fail{{color:#721c24;background:#f8d7da;font-weight:600;border-radius:4px;padding:2px 8px;}}
  .best{{font-weight:700;color:#1a1a2e;}}
  .section{{background:#fff;border-radius:8px;padding:20px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;}}
  .badge-gold{{background:#fff3cd;color:#856404;}}
  .badge-silver{{background:#e9ecef;color:#495057;}}
  .badge-bronze{{background:#fde8d0;color:#8b4513;}}
  .code{{font-family:'SF Mono','Fira Code',monospace;font-size:12px;background:#f4f4f4;padding:1px 4px;border-radius:3px;}}
  .footnote{{font-size:12px;color:#888;margin-top:8px;}}
  .summary-bar{{display:flex;gap:16px;margin:16px 0;flex-wrap:wrap;}}
  .summary-card{{flex:1;min-width:140px;background:#fff;padding:20px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);text-align:center;}}
  .summary-card .value{{font-size:28px;font-weight:700;color:#1a1a2e;}}
  .summary-card .label{{font-size:12px;color:#888;text-transform:uppercase;margin-top:4px;}}
</style>
</head>
<body>
<div class="container">
<h1>端侧推理框架性能基准测试报告</h1>
<p class="meta">{now} &mdash; 测试: {len(backends)} 后端 x {len(models)} 模型 = {total} 组</p>
<div class="summary-bar">
  <div class="summary-card"><div class="value">{len(backends)}</div><div class="label">后端</div></div>
  <div class="summary-card"><div class="value">{total}</div><div class="label">测试总数</div></div>
  <div class="summary-card"><div class="value" style="color:#155724;">{passed}</div><div class="label">通过</div></div>
  <div class="summary-card"><div class="value" style="color:#721c24;">{total - passed}</div><div class="label">失败/跳过</div></div>
  <div class="summary-card"><div class="value">{passed / total * 100:.1f}%</div><div class="label">通过率</div></div>
</div>

<h2>跨后端性能对比</h2>
<div class="section">
<table>
  <thead><tr><th>后端</th><th>模型</th><th>Init(ms)</th><th>P50(ms)</th><th>P90(ms)</th><th>P99(ms)</th><th>FPS</th><th>Cosine Sim</th><th>状态</th></tr></thead>
  <tbody>
{cross_rows}  </tbody>
</table>
  <p class="footnote">加粗行 = 每组模型 FPS 最优。ONNX Runtime 为精度参考基准，不自比。</p>
</div>

<h2>性能排名</h2>
<div class="section">
<table>
  <thead><tr><th>模型</th><th>第1名 (FPS)</th><th>第2名 (FPS)</th><th>第3名 (FPS)</th></tr></thead>
  <tbody>{rank_rows}</tbody>
</table>
</div>

<h2>各后端详情</h2>
<div class="section">{backend_tables}</div>

<h2>精度验证</h2>
<div class="section">
<table>
  <thead><tr><th>后端</th><th>模型</th><th>Cosine Similarity</th><th>MAE</th><th>Max Error</th><th>判定</th></tr></thead>
  <tbody>{acc_rows}</tbody>
</table>
  <p class="footnote">以 ONNX Runtime 输出为参考基准。余弦相似度 0.99+ 为通过。</p>
</div>
</div>
</body>
</html>'''
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML report: {output_path} ({len(html)} bytes)')

if __name__ == '__main__':
    log_files = sorted(glob.glob('results/*.log'), key=os.path.getmtime, reverse=True)
    if not log_files:
        print('Error: no log files found in results/')
        sys.exit(1)
    log_path = log_files[0]  # 使用最新的日志
    results = parse_results(log_path)
    if not results:
        print('Error: no valid results found in log')
        sys.exit(1)
    gen_html(results, log_path, 'results/benchmark_report.html')
```

### 输出指标

| 指标 | 说明 |
|------|------|
| P50 | 中位数延迟（ms），核心指标 |
| P90 | 90% 尾部延迟 |
| P99 | 99% 尾部延迟 |
| FPS | 每秒推理帧数 |
| 内存峰值 | 推理过程最大内存占用 |
| 余弦相似度 | 与 ORT 标杆的精度一致性 |

### 输出示例

```
--- Performance Results ---
  Init time:  7.23 ms
  P50:    8.70 ms
  P90:    9.12 ms
  P99:    9.45 ms
  Throughput: 114.94 FPS
  Peak mem:   24576 KB
```

## 数据真实性验证表

报告中必须包含以下验证表，确保数据可追溯：

```markdown
## 数据真实性验证

| 验证项 | 结果 | 来源命令 |
|--------|------|---------|
| 设备型号 | (设备型号) | adb shell getprop |
| 代码版本 | XXXXXXX | git log |
| 二进制 MD5 | XXXXXXX | md5sum |
| 模型文件 | name (XX MB) | ls -lh |
| 原始日志 | XXX lines | wc -l |
| 设备温度 | XX°C | thermal_zone |
| 执行时间 | YYYY-MM-DD HH:MM | date |
```

## 红线

- 不得伪造 adb 输出、模型数据、性能数字
- 不得跳过设备握手步骤
- 不得使用旧数据代替新跑 — 每次 request 必须重新运行
- 不得静默忽略命令失败 — exit code != 0 必须报告
- 不得跳过 HTML 报告生成
- 每行数据必须标注来源命令
- 温度 > 45°C 时必须在报告中标注降频风险

## 常见问题

- **结果波动大**: 检查是否已执行 `scripts/setup/setup-test-env.sh`
- **模型找不到**: `adb shell ls /data/local/tmp/benchmark/models/`
- **动态库找不到**: 推送后检查 `LD_LIBRARY_PATH` 是否正确设置
- **device not found**: 检查 `adb devices`，确认设备 ID 在列表中
- **permission denied**: 推送后 `chmod +x`
- **空间不足**: `adb shell rm -rf /data/local/tmp/*`
- **精度对比为 N/A**: 后端未实现 `infer_with_output()` 或 ORT 标杆输出不可用
- **报告为空**: 检查日志文件路径是否正确

## 关联 Skill

- [mobile-bench-model-prep](../mobile-bench-model-prep/SKILL.md) — 模型下载、转换、二进制编译
- [mobile-bench-profiling](../mobile-bench-profiling/SKILL.md) — 逐算子 profiling、火焰图、系统 trace
- [mobile-bench-integrate](../mobile-bench-integrate/SKILL.md) — 集成新推理框架后端
- [mobile-bench-methodology](../mobile-bench-methodology/SKILL.md) — 多轮统计、环境控制补全(电量/充电)、公平对比与可复现性清单,正式测量建议配合使用
