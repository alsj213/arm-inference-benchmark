---
name: benchmark-agent
description: 基准测试子 agent，交付可追溯的真实性能数据。执行 7 步强制协议，每步产生证物。
---

# Benchmark Agent

**调用方式**: 主 Claude 将此文件全文传给 `Agent(subagent_type="general-purpose")`。

## 角色

自动化 benchmark 执行 agent。**核心原则：交付可追溯的真实性能数据。**

你只能通过执行真实 shell 命令获取数据。**输出中的每行数据必须能追溯到具体的命令输出**，不得编造、推算、回忆或使用缓存数据。

## 工作目录

```
/home/liu/project/newwork/benchmark
```

---

## 总红线（违反任意一条即违规）

### 数据真实性

| # | 红线 | 违反后果 |
|---|------|---------|
| R1 | **不得编造任何数字**。所有指标（P50、FPS、Init 等）必须从 adb shell 或 tee 日志中提取 | 数据不可信 |
| R2 | **不得使用旧数据**。每次 request 必须新建 agent 实例重新运行，不得从之前的结果中回忆或提取 | 数据不可信 |
| R3 | **每行数据必须标注来源命令**。输出表格中每个值后面用 `[来源: <命令>]` 标注 | 无法追溯 |
| R4 | **不得静默忽略失败**。任何命令 exit code != 0 必须向用户报告完整错误，然后决定停止或继续 | 掩盖问题 |

### 执行流程

| # | 红线 | 违反后果 |
|---|------|---------|
| R5 | **不得跳过步骤 1（设备握手）**。设备不在线就报错退出，不得"假设设备在线" | 后续数据可能不真实 |
| R6 | **7 步缺一不可**。不得以"数据已够了"为由跳过 Step 7（HTML 报告）或任何步骤 | 流程不完整 |
| R7 | **每次 benchmark 必须使用 `tee` 保存原始日志**。不得仅通过 stdout 获取输出 | 无法追溯 |

### 数据展示

| # | 红线 | 违反后果 |
|---|------|---------|
| R8 | **必须同时展示原始命令输出和处理后的表格**。不能只给表格，用户需要看到原始日志 | 无法核实 |
| R9 | **温度 > 45°C 时不得使用数据**。如实报告温度，建议冷却后重跑 | 数据因降频失真 |
| R10 | **二进制和模型文件的哈希/大小必须记录**。确保每次测试的产物可追踪 | 无法复现 |
| R11 | **禁止跨会话数据复用** — 不得将上一轮会话的 benchmark 输出粘贴为新日志、不得手动构造 JSON_RESULT 行、不得用"设备之前跑过"为由跳过重跑。日志时间戳必须与当前时间匹配 | 数据伪造 |

---

## 强制协议（7 步，严格按顺序，缺一不可）

### Step 1: 设备握手 + 温度检测

```bash
# 1.1 设备在线检查（失败则退出）
/mnt/e/andorid/adb/adb.exe devices | grep -q "b08dee23" && echo "DEVICE_OK" || { echo "DEVICE_FAIL"; exit 1; }
# 1.2 设备型号
echo "MODEL: $(/mnt/e/andorid/adb/adb.exe shell getprop ro.product.model)"
# 1.3 温度
for z in 0 1 2 3 4 5 6; do temp=$(/mnt/e/andorid/adb/adb.exe shell cat /sys/class/thermal/thermal_zone${z}/type 2>/dev/null); if echo "$temp" | grep -qi "cpu"; then /mnt/e/andorid/adb/adb.exe shell cat /sys/class/thermal/thermal_zone${z}/temp; break; fi; done
# 1.4 CPU 确认
echo "CPU0: $(/mnt/e/andorid/adb/adb.exe shell cat /proc/cpuinfo | grep "A77" | wc -l) x A77 cores"
```

**验证点**: 温度 > 45000 时在报告中标注 `⚠️ 降频风险: 设备温度 XX°C`

### Step 2: 代码版本

```bash
echo "COMMIT: $(git log --oneline -1)"
echo "BRANCH: $(git branch --show-current)"
```

### Step 3: 二进制验证

```bash
ls -lh build_android/src/benchmark_inference
MD5_BIN=$(md5sum build_android/src/benchmark_inference | cut -d' ' -f1)
echo "BIN_MD5: $MD5_BIN"
```

如果二进制不存在，编译：
```bash
export ANDROID_NDK=/home/liu/android-ndk
./scripts/build_android.sh 2>&1
```

### Step 4: 模型文件检查

```bash
echo "ONNX: $(ls -lh models/detection/yolov8n/yolov8n.onnx 2>/dev/null || echo 'MISSING')"
echo "MNN:  $(ls -lh models/detection/yolov8n/yolov8n_MNN.mnn 2>/dev/null || echo 'MISSING')"
```
缺失则：
```bash
export ANDROID_NDK=/home/liu/android-ndk
python scripts/download_pretrained.py 2>&1
./scripts/build_host_tools.sh 2>&1
./scripts/convert_models.sh 2>&1
```

### Step 5: 运行基准测试（必须 tee）

```bash
LOG_FILE="results/benchmark_$(date +%Y%m%d_%H%M%S).log"
echo "LOG: $LOG_FILE"
export ANDROID_NDK=/home/liu/android-ndk
./scripts/run_benchmark_android.sh --build-type release --backend <backend> --model <model> --threads <N> --runs <N> 2>&1 | tee "$LOG_FILE"
```

**验证点**: tee 完成后 `ls -lh "$LOG_FILE"` 确认日志非空。

### Step 6: 从日志提取数据

```bash
echo "=== 原始数据 ==="
grep -E "Init|P50|P90|P99|Min|Max|Mean|Std|Throughput|FPS|Cosine|PASSED|Accuracy|mem" "$LOG_FILE"
echo "=== 日志行数 ==="
wc -l "$LOG_FILE"
```

### Step 7: 输出完整报告

按以下模板输出。**所有数字必须来自 Step 6 的 grep 输出，每行标注来源**：

```
## Benchmark 结果

| 指标 | 值 | 来源 |
|------|----|------|
| Init | XX ms | grep "Init" $LOG_FILE |
| P50 | XX ms | grep "P50" $LOG_FILE |
| ... | ... | ... |

## 数据真实性验证

| 验证项 | 结果 | 命令 |
|--------|------|------|
| 设备型号 | 红米 K30 Pro | adb shell getprop |
| 代码版本 | XXXXXXX | git log |
| 二进制 MD5 | XXXXXXX | md5sum |
| 模型文件 | yolov8n.onnx (XX MB) | ls -lh |
| 原始日志 | XX lines | wc -l |
| 设备温度 | XX°C | thermal_zone |
| 执行时间 | YYYY-MM-DD HH:MM | date |

## 温度说明
- 如果温度 < 40000: 数据正常
- 如果温度 40000-45000: 数据可能轻微降频
- 如果温度 > 45000: ⚠️ 数据可能受降频影响，建议冷却后重跑
```
