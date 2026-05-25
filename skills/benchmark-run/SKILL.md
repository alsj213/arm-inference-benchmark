---
name: benchmark-run
description: Use when running inference performance tests on Android devices — full workflow from device handshake to result report, including environment control, ADB operations, and data validation
---

# Benchmark Run

## Overview

手机端推理 benchmark 完整流程：设备握手 → 版本检查 → 二进制验证 → 模型检查 → 锁频 → 执行(tee日志) → 恢复环境 → 结果解析 → 报告生成。

配置通过 `.benchmarkrc.yml` 读取（设备 ID、ADB 路径等），无需硬编码。

## 完整流程

```text
Step 1: 设备握手 → Step 2: 代码版本 → Step 3: 二进制验证 → Step 4: 模型检查
→ Step 5: 环境控制(锁频) → Step 6: 执行(tee日志) → Step 7: 恢复环境
→ 结果解析 → 报告生成
```

## 读取配置

```bash
# 从 .benchmarkrc.yml 读取设备配置（如设备ID、ADB路径等）
python3 -c "
import yaml
with open('.benchmarkrc.yml') as f:
    cfg = yaml.safe_load(f)
print('Device:', cfg['device']['id'])
print('ADB:',   cfg['device']['adb_path'])
"
```

## 分步流程

### Step 1: 设备握手

验证设备在线并记录设备信息：

```bash
DEVICE_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['id'])")
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb_path'])")

# 检查连接
$ADB devices | grep "$DEVICE_ID" || exit 1

# 记录设备信息
$ADB shell getprop ro.product.model
$ADB shell cat /proc/cpuinfo | grep "A77"

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

# 如需重新编译
./scripts/build_android.sh
```

### Step 4: 检查模型文件

```bash
ls -lh models/classification/mobilenetv2/mobilenetv2.onnx
# 缺失则执行下载和转换
```

### Step 5: 环境控制

锁频、清缓存，保证测试环境一致性。

```bash
# 自动检测 root 权限，有 root 则设置 performance governor 并清缓存
./scripts/setup_test_environment.sh
```

**有 root 时：** CPU performance governor、清理缓存（`echo 3 > /proc/sys/vm/drop_caches`）、记录初始频率温度、停止 zygote

**无 root 时：** 跳过硬件控制，仅记录状态，设置进程优先级（nice）

### Step 6: 执行测试

推送并运行，使用 `tee` 保留原始日志：

```bash
# 推送二进制、动态库、模型
$ADB push build_android/src/benchmark_inference /data/local/tmp/benchmark/
$ADB push build_android/third_party/libonnxruntime.so /data/local/tmp/benchmark/
$ADB push build_android/third_party/libMNN.so /data/local/tmp/benchmark/
$ADB push models/ /data/local/tmp/benchmark/

# 执行测试（必须使用 tee 保存日志）
ANDROID_NDK=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['env']['android_ndk'])")
./scripts/run_benchmark_android.sh --model mobilenetv2 --backend all | tee results/latest_benchmark.log
```

**重要：** 记录二进制 md5、模型文件大小和设备温度。

#### 命令行参数 (benchmark_inference)

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--model` | 模型名 | mobilenetv2, resnet50, yolov8n, bert |
| `--backend` | 后端类型 | mnn, onnxrt, ort |
| `--precision` | 推理精度 | fp32, fp16, int8 |
| `--threads` | 线程数 | 1, 2, 4 |
| `--runs` | 运行次数 | 100 |
| `--warmup` | 预热次数 | 10 |
| `--gpu` | 使用 GPU | 无参数 |
| `--profiling <file>` | 启用逐算子 profiling | profiling.json |

### Step 7: 恢复环境

```bash
./scripts/restore_test_environment.sh
```

有 root 时恢复 schedutil governor 并重启 zygote。无 root 时仅记录状态。

## 设备操作参考

### ADB 常用命令

| 操作 | 命令 |
|------|------|
| 检查连接 | `adb devices` |
| 验证设备信息 | `adb shell getprop ro.product.model && adb shell cat /proc/cpuinfo \| grep "A77"` |
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

- **标杆**: ONNX Runtime 输出
- **指标**: 余弦相似度（方向一致性，1.0=完全一致）、平均绝对误差（MAE）
- **输入一致**: `fill_random_float` 使用固定种子 42，保证所有后端输入相同

### 生成报告

```bash
python3 scripts/generate_report.py results/sm8250
```

### HTML 报告生成

```bash
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
| 设备型号 | 红米 K30 Pro | adb shell getprop |
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

- **结果波动大**: 检查是否已执行 `setup_test_environment.sh`
- **模型找不到**: `adb shell ls /data/local/tmp/benchmark/models/`
- **libonnxruntime.so 找不到**: 脚本自动推送至同目录，`LD_LIBRARY_PATH` 已设置
- **device not found**: 检查 `adb devices`，确认设备 ID 在列表中
- **permission denied**: 推送后 `chmod +x`
- **空间不足**: `adb shell rm -rf /data/local/tmp/*`
- **精度对比为 N/A**: 后端未实现 `infer_with_output()` 或 ORT 标杆输出不可用
- **报告为空**: 检查日志文件路径是否正确
