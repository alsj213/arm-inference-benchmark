---
name: mobile-bench-run
description: Use when running inference performance tests on Android devices — full workflow from device handshake to result report, including environment control, ADB operations, and data validation
---

# Mobile Bench Run

## Overview

手机端推理 benchmark 完整流程：设备握手 → 版本检查 → 二进制验证 → 模型检查 → 锁频 → 执行(tee日志) → 恢复环境 → 结果解析 → 报告生成。

> **定位（对齐现状）**：本仓实测重心已 **LLM 化**——LLM/VL 全流程（TTFT/精度对齐/长上下文）走 **mobile-bench-llm** skill；
> 本 run skill 提供**通用 7 步骨架**，其 CNN 示例仅在具备 CNN 模型时使用（模型需先经 mobile-bench-model-prep 准备，
> 见 Step 4）。LLM 二进制与模型见各步内注。

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
# CNN 推理二进制（实际产物在 src/cnn/ 下）
ls -lh build_android/src/cnn/benchmark_inference
md5sum build_android/src/cnn/benchmark_inference
# LLM 推理二进制（LLM/VL 流程使用）
ls -lh build_android/src/llm/llm_benchmark

# 如需重新编译，使用编译脚本
./scripts/build/build-android.sh
```

### Step 4: 检查模型文件

```bash
# CNN 模型源在 models/source/…（本机经 models symlink → /mnt/e/wsl/home_liu/models/source）。
# 当前仓 E 盘无已转换 classification 模型：测 CNN 前必须先经 mobile-bench-model-prep
# 下载/转换（源 models/source/classification → 各框架产物）再推送设备。
# LLM 模型在 models/llm/…（GGUF/MNN），设备侧 /data/local/tmp/benchmark/qwen3_models/。
ls models/source/classification/ 2>/dev/null || echo "无 CNN 源模型——先走 mobile-bench-model-prep"
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

报告统一由脚本生成（模板在脚本内，不在 skill 内嵌，避免与 generate_report.py 重复维护）：

```bash
# CNN/NLP 通用：自动解析 results/ 下全部日志（多结果日志完整收集，非只取首条）
python3 scripts/analyze/mobilebench/generate_report.py results/

# LLM 三方/双框架对比：从特定日志生成 HTML（commit/md5/温度现场校验）
python3 scripts/analyze/gen_qwen3_3way_report.py results/qwen3_06b_3way_<ts>.log
```

报告字段契约、量化对齐（--require-precision）与统计口径见 mobile-bench-llm / mobile-bench-methodology skill。


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
