# 端侧推理引擎性能分析项目 — 完整实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标:** 以骁龙 865 为硬件平台，MNN 为主分析对象，建立从模型级 Profiling → 算子级性能画像 → 瓶颈定位 → 算子优化闭环的完整技术故事线。

**架构:** 3 框架（MNN/ORT/llama.cpp）× 5 模型（MobileNetV2/ResNet50/BERT/Qwen2-0.5B/MobileViT-S）× 2 后端（CPU/GPU OpenCL）= 全矩阵对比。以 Profiling 数据驱动热点算子识别，按二八原则做三层深度 Benchmark，最终产出 BERT 优化闭环 + Roofline 分析报告。

> **注意:** TVM 已从主线移除（见附录 D）。TVM 需要 AutoTVM 预先调优（单模型数小时~数天），会严重拖慢整体节奏。保留为可选扩展项，主线聚焦 MNN/ORT/llama.cpp 三条优化路线的深度对比。

**技术栈:** C++17 / CMake / Android NDK (arm64-v8a) / MNN / ONNX Runtime / llama.cpp / simpleperf / Python (matplotlib + pandas) / ARM Neon intrinsics

**硬件:** 红米 K30S（骁龙 865 SM8250，1×A77@2.84GHz + 3×A77@2.42GHz + 4×A55@1.8GHz，Adreno 650 GPU）

**预估总工期:** 30 天（业余投入，每周 3-4 晚 + 周末一天）

---

## 项目文件结构总览

```
benchmark/
├── src/
│   ├── main.cpp                          # [修改] 入口 — 增加 GPU 后端支持 + LLM 模型路径
│   ├── single_op_benchmark.cpp           # [修改] 扩充全量热点算子测例
│   ├── common/
│   │   ├── benchmark.h/cpp               # [修改] 增加 GPU 后端工厂注册
│   │   ├── config.h/cpp                   # [修改] 增加 BackendType::MNN_GPU 枚举
│   │   └── utils.h/cpp                    # [修改] 增加 GPU 内存统计 + roofline 计算工具
│   ├── backends/
│   │   ├── mnn_backend.cpp/h             # [修改] 增加 OpenCL 后端路径 + GPU 推理选项
│   │   ├── ort_backend.cpp/h             # [修改] 增加详细 Profiling 输出
│   │   └── llamacpp_backend.cpp/h        # [修改] 增加 Qwen2-0.5B 模型支持
│   └── models/
│       └── model_info.h                  # [修改] 增加 MobileViT-S + Qwen2-0.5B 模型定义
├── models/
│   ├── classification/                   # 现有: mobilenetv2, resnet18, resnet50, shufflenet_v2, squeezenet, efficientnet_lite0
│   │   └── mobilevit_s/                  # [新增] MobileViT-S 模型文件
│   ├── nlp/
│   │   ├── bert/                         # 现有
│   │   └── qwen2_0.5b/                   # [新增] Qwen2-0.5B 模型文件 (.gguf)
│   └── detection/
│       └── yolov8n/                      # 现有
├── scripts/
│   ├── build_android.sh                  # [修改] 增加 llama.cpp 编译开关
│   ├── run_benchmark_android.sh          # [修改] 增加 GPU 后端参数 + MNN_PROFILING 环境变量
│   ├── setup_test_environment.sh         # [修改] 增加 GPU 频率锁定
│   ├── profile_benchmark.sh              # [修改] 集成 simpleperf 自动化抓取
│   ├── analyze_roofline.py               # [新增] Roofline 模型分析脚本
│   ├── generate_operator_report.py       # [新增] 算子级性能报告生成
│   └── convert_qwen_to_gguf.sh           # [新增] Qwen2 → GGUF 转换脚本
├── results/
│   ├── phase1_cpu_profiling/             # [新增] Phase 1 全量 Profiling 数据
│   ├── phase2_hotspot_benchmark/         # [新增] Phase 2 热点算子深度测试数据
│   ├── phase3_gpu_comparison/            # [新增] Phase 3 CPU vs GPU 对比数据
│   ├── phase4_bert_optimization/         # [新增] Phase 4 BERT 优化闭环数据
│   └── phase5_llm_profiling/             # [新增] Phase 5 LLM 推理分析数据
├── docs/
│   └── 03-plans/
│       └── 2026-06-06-benchmark-full-plan.md  # 本文件
└── .benchmarkrc.yml                      # [修改] 增加 GPU 后端 + 新模型配置
```

---

## Phase 0: 环境就绪（0.5 天）

### Task 0.1: 设备连接验证

- [ ] **Step 0.1.1: ADB 设备握手**

```bash
export ADB=/mnt/e/andorid/adb/adb.exe
$ADB devices | grep "device$" || echo "ERROR: 设备未连接"
# 预期输出: b08dee23  device
```

- [ ] **Step 0.1.2: 验证设备型号和芯片**

```bash
$ADB shell getprop ro.product.model
# 预期输出: M2007J3SC (红米 K30S)
$ADB shell cat /proc/cpuinfo | grep -E "A77|A55" | head -4
# 预期输出: 包含 Cortex-A77 和 Cortex-A55 信息
```

- [ ] **Step 0.1.3: 验证 root 权限**

```bash
$ADB shell su -c "echo root_ok"
# 预期输出: root_ok
# 否则后续无法锁频
```

- [ ] **Step 0.1.4: 记录当前内核版本和调度器状态**

```bash
$ADB shell cat /proc/version > results/phase0_kernel_version.txt
$ADB shell cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > results/phase0_governor.txt
```

### Task 0.2: 编译工具链验证

- [ ] **Step 0.2.1: 确认 NDK 路径和版本**

```bash
ls /home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++
# 预期: 文件存在
/home/liu/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ --version
# 记录版本号
```

- [ ] **Step 0.2.2: 确认 CMake 版本 ≥ 3.14**

```bash
cmake --version | head -1
# 预期: cmake version >= 3.14
```

- [ ] **Step 0.2.3: 验证 MNN 源码中 Profiler 宏已启用**

```bash
grep -r "MNN_PROFILING" third_party/MNN/CMakeLists.txt | head -5
grep -r "MNN_OPENCL" third_party/MNN/CMakeLists.txt | head -5
# 检查 OpenCL 编译选项是否存在
```

- [ ] **Step 0.2.4: 验证 simpleperf 工具可用**

```bash
which simpleperf || echo "需要在 Android NDK 中定位 simpleperf"
find /home/liu/android-ndk -name "simpleperf" -type f 2>/dev/null | head -3
$ADB shell which simpleperf || echo "需要推送 simpleperf 到设备"
```

### Task 0.3: 模型文件清单验证

- [ ] **Step 0.3.1: 确认现有 5 个基准模型文件齐全**

```bash
# MNN 格式
ls -lh models/classification/mobilenetv2/mobilenetv2.mnn
ls -lh models/classification/resnet50/resnet50.mnn
ls -lh models/nlp/bert/bert.mnn
ls -lh models/detection/yolov8n/yolov8n.mnn
# MobileViT-S 需要在后续步骤下载转换

# ONNX 格式
ls -lh models/classification/mobilenetv2/mobilenetv2.onnx
ls -lh models/classification/resnet50/resnet50.onnx
ls -lh models/nlp/bert/bert.onnx
ls -lh models/detection/yolov8n/yolov8n.onnx
```

- [ ] **Step 0.3.2: 记录模型文件大小和 MD5（可追溯性）**

```bash
find models/ -name "*.mnn" -o -name "*.onnx" -o -name "*.gguf" | while read f; do
  echo "$f | $(stat -c%s "$f") bytes | $(md5sum "$f" | cut -c1-8)"
done > results/phase0_model_inventory.txt
cat results/phase0_model_inventory.txt
```

- [ ] **Step 0.3.3: 验证 MNN Profiler 功能可用性**

```bash
# 用 mobilenetv2 做快速冒烟测试
export ADB=/mnt/e/andorid/adb/adb.exe
export ANDROID_NDK=/home/liu/android-ndk
MNN_PROFILING=1 ./scripts/run_benchmark_android.sh --model mobilenetv2 --backend mnn --iterations 10 2>&1 | head -50
# 检查输出是否包含逐算子耗时信息
```

---

## Phase 1: CPU 全量 Profiling（2-3 天）

### Task 1.1: 跑通 5 个模型 × MNN CPU 的全量 Profiling

- [ ] **Step 1.1.1: 编译 Android Release 版本（确保 MNN Profiling 已启用）**

```bash
cd /home/liu/project/newwork/benchmark
# 检查 CMakeLists.txt 确保 MNN 编译时定义了 MNN_PROFILING
grep "MNN_PROFILING\|MNN_ENABLE_PROFILING" third_party/MNN/CMakeLists.txt CMakeLists.txt

# 重新编译（确保 profiler 符号链接）
./scripts/build_android.sh 2>&1 | tee results/phase1_build_log.txt
# 预期: Build SUCCESS
```

- [ ] **Step 1.1.2: 修改 run_benchmark_android.sh 支持 --profile 参数**

在 `scripts/run_benchmark_android.sh` 中添加：
```bash
# 在参数解析部分添加
--profile)
    export MNN_PROFILING=1
    shift
    ;;
```

- [ ] **Step 1.1.3: MobileNetV2 — MNN CPU Profiling**

```bash
export ADB=/mnt/e/andorid/adb/adb.exe
export ANDROID_NDK=/home/liu/android-ndk
MNN_PROFILING=1 ./scripts/run_benchmark_android.sh \
  --model mobilenetv2 --backend mnn --iterations 100 --warmup 50 \
  2>&1 | tee results/phase1_cpu_profiling/mobilenetv2_mnn_cpu_profile.log
# 关键输出: 总耗时、每个算子的名称/耗时/占比
```

- [ ] **Step 1.1.4: ResNet50 — MNN CPU Profiling**

```bash
MNN_PROFILING=1 ./scripts/run_benchmark_android.sh \
  --model resnet50 --backend mnn --iterations 100 --warmup 50 \
  2>&1 | tee results/phase1_cpu_profiling/resnet50_mnn_cpu_profile.log
```

- [ ] **Step 1.1.5: BERT — MNN CPU Profiling**

```bash
MNN_PROFILING=1 ./scripts/run_benchmark_android.sh \
  --model bert --backend mnn --iterations 100 --warmup 50 \
  2>&1 | tee results/phase1_cpu_profiling/bert_mnn_cpu_profile.log
```

- [ ] **Step 1.1.6: YOLOv8n — MNN CPU Profiling**

```bash
MNN_PROFILING=1 ./scripts/run_benchmark_android.sh \
  --model yolov8n --backend mnn --iterations 100 --warmup 50 \
  2>&1 | tee results/phase1_cpu_profiling/yolov8n_mnn_cpu_profile.log
```

- [ ] **Step 1.1.7: MobileViT-S — MNN CPU Profiling（需要先下载/转换模型，见 Task 1.0）**

```bash
MNN_PROFILING=1 ./scripts/run_benchmark_android.sh \
  --model mobilevit_s --backend mnn --iterations 100 --warmup 50 \
  2>&1 | tee results/phase1_cpu_profiling/mobilevit_s_mnn_cpu_profile.log
```

### Task 1.0: 准备缺失模型（MobileViT-S + Qwen2-0.5B）

- [ ] **Step 1.0.1: 下载 MobileViT-S ONNX 预训练模型**

```bash
cd /home/liu/project/newwork/benchmark
mkdir -p models/classification/mobilevit_s
# 从 Apple 官方 ML-ASR 或 ONNX Model Zoo 下载
python3 scripts/download_pretrained.py --model mobilevit_s --format onnx \
  --output models/classification/mobilevit_s/
# 验证下载结果
ls -lh models/classification/mobilevit_s/mobilevit_s.onnx
```

- [ ] **Step 1.0.2: 转换 MobileViT-S ONNX → MNN**

```bash
./scripts/convert_models.sh --model mobilevit_s --format mnn
ls -lh models/classification/mobilevit_s/mobilevit_s.mnn
```

- [ ] **Step 1.0.3: 下载 Qwen2-0.5B GGUF 模型（用于 llama.cpp）**

```bash
mkdir -p models/nlp/qwen2_0.5b
# 从 HuggingFace 下载
python3 scripts/download_pretrained.py --model qwen2_0.5b --format gguf \
  --output models/nlp/qwen2_0.5b/
ls -lh models/nlp/qwen2_0.5b/*.gguf
```

- [ ] **Step 1.0.4: 在 model_info.h 中注册 MobileViT-S 和 Qwen2-0.5B**

在 `src/models/model_info.h` 中添加：
```cpp
// MobileViT-S 模型定义
const ModelInfo MOBILEVIT_S = {
    "mobilevit_s",
    ModelType::CLASSIFICATION,
    "models/classification/mobilevit_s/mobilevit_s.mnn",
    "models/classification/mobilevit_s/mobilevit_s.onnx",
    {1, 3, 256, 256},  // 输入尺寸
    1000,                // 输出类别数
};

// Qwen2-0.5B 模型定义
const ModelInfo QWEN2_0_5B = {
    "qwen2_0.5b",
    ModelType::NLP_LLM,
    "",  // MNN 暂不直接支持，走 llama.cpp
    "",
    {1, 512},  // seq_len 可变
    151936,     // vocab size
};
```

### Task 1.2: simpleperf 火焰图抓取（交叉验证 MNN Profiler 数据）

- [ ] **Step 1.2.1: 准备 simpleperf 抓取脚本**

```bash
# 确保 simpleperf 在设备上可用
export ADB=/mnt/e/andorid/adb/adb.exe
$ADB push /home/liu/android-ndk/simpleperf/bin/android/arm64/simpleperf /data/local/tmp/
$ADB shell chmod +x /data/local/tmp/simpleperf
```

- [ ] **Step 1.2.2: 用 simpleperf stat 抓取 MobileNetV2 的 PMU 计数器**

```bash
$ADB shell su -c "/data/local/tmp/simpleperf stat \
  -e cpu-cycles,instructions,cache-misses,branch-misses,stalled-cycles-backend,stalled-cycles-frontend \
  --duration 10 \
  /data/local/tmp/benchmark_inference --model mobilenetv2 --backend mnn --iterations 500" \
  2>&1 | tee results/phase1_cpu_profiling/mobilenetv2_simpleperf_stat.log
```

- [ ] **Step 1.2.3: 用 simpleperf record 抓取 MobileNetV2 火焰图**

```bash
$ADB shell su -c "/data/local/tmp/simpleperf record \
  -e cpu-cycles -g --duration 10 --call-graph fp \
  -o /data/local/tmp/perf_mbv2.data \
  /data/local/tmp/benchmark_inference --model mobilenetv2 --backend mnn --iterations 200"

$ADB pull /data/local/tmp/perf_mbv2.data results/phase1_cpu_profiling/
# 生成火焰图报告
simpleperf report -i results/phase1_cpu_profiling/perf_mbv2.data --sort comm,dso,symbol \
  > results/phase1_cpu_profiling/mobilenetv2_simpleperf_report.txt
```

- [ ] **Step 1.2.4: 对 ResNet50 和 BERT 重复 simpleperf 抓取**

```bash
# ResNet50
$ADB shell su -c "/data/local/tmp/simpleperf record \
  -e cpu-cycles -g --duration 10 --call-graph fp \
  -o /data/local/tmp/perf_resnet50.data \
  /data/local/tmp/benchmark_inference --model resnet50 --backend mnn --iterations 50"

$ADB pull /data/local/tmp/perf_resnet50.data results/phase1_cpu_profiling/

# BERT
$ADB shell su -c "/data/local/tmp/simpleperf record \
  -e cpu-cycles -g --duration 10 --call-graph fp \
  -o /data/local/tmp/perf_bert.data \
  /data/local/tmp/benchmark_inference --model bert --backend mnn --iterations 20"

$ADB pull /data/local/tmp/perf_bert.data results/phase1_cpu_profiling/
```

### Task 1.3: ORT 基线 Profiling（建立对比参照）

- [ ] **Step 1.3.1: 5 个模型 × ORT CPU 跑全量 Profiling**

```bash
for model in mobilenetv2 resnet50 bert yolov8n; do
  ./scripts/run_benchmark_android.sh \
    --model $model --backend ort --iterations 100 --warmup 50 \
    2>&1 | tee results/phase1_cpu_profiling/${model}_ort_cpu_profile.log
done
```

- [ ] **Step 1.3.2: 开启 ORT 全部图优化后重新测试**

修改 `src/backends/ort_backend.cpp`，确保 SessionOptions：
```cpp
SessionOptions options;
options.graph_optimization_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
options.enable_cpu_mem_arena = true;
options.intra_op_num_threads = 4;  // 骁龙865大核数
options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
```

- [ ] **Step 1.3.3: 重新编译并测试 ORT All-Optimizations**

```bash
./scripts/build_android.sh
for model in mobilenetv2 resnet50 bert yolov8n; do
  ./scripts/run_benchmark_android.sh \
    --model $model --backend ort --iterations 100 --warmup 50 \
    2>&1 | tee results/phase1_cpu_profiling/${model}_ort_allopt_profile.log
done
```

### Task 1.4: 产出每个模型的热点算子排行

- [ ] **Step 1.4.1: 编写算子耗时提取脚本**

创建 `scripts/extract_operator_profile.py`：
```python
#!/usr/bin/env python3
"""从 MNN Profiling 日志中提取逐算子耗时排行"""
import re
import sys
from collections import defaultdict

def parse_mnn_profile(log_path):
    """解析 MNN_PROFILING=1 的输出"""
    ops = []
    with open(log_path) as f:
        for line in f:
            # MNN Profiler 输出格式: op_name, time_ms, flops
            m = re.match(r'.*\[PROFILER\].*(\w+)\s+(\d+\.?\d*)\s*ms', line)
            if m:
                ops.append((m.group(1), float(m.group(2))))
    return sorted(ops, key=lambda x: -x[1])

def print_hotspot_ranking(log_path, model_name):
    ops = parse_mnn_profile(log_path)
    total = sum(t for _, t in ops)
    cumulative = 0
    print(f"\n## {model_name} 热点算子排行")
    print(f"| 排名 | 算子 | 耗时(ms) | 占比(%) | 累计占比(%) |")
    print(f"|------|------|---------|---------|------------|")
    for i, (name, time) in enumerate(ops, 1):
        pct = time / total * 100
        cumulative += pct
        print(f"| {i} | {name} | {time:.3f} | {pct:.1f} | {cumulative:.1f} |")
    return ops

if __name__ == '__main__':
    for log_file in sys.argv[1:]:
        model = log_file.split('/')[-1].replace('_mnn_cpu_profile.log', '')
        print_hotspot_ranking(log_file, model)
```

- [ ] **Step 1.4.2: 对 5 个模型执行算子热度分析**

```bash
python3 scripts/extract_operator_profile.py \
  results/phase1_cpu_profiling/mobilenetv2_mnn_cpu_profile.log \
  results/phase1_cpu_profiling/resnet50_mnn_cpu_profile.log \
  results/phase1_cpu_profiling/bert_mnn_cpu_profile.log \
  results/phase1_cpu_profiling/yolov8n_mnn_cpu_profile.log \
  results/phase1_cpu_profiling/mobilevit_s_mnn_cpu_profile.log \
  > results/phase1_cpu_profiling/all_models_hotspot_ranking.md
```

- [ ] **Step 1.4.3: 标记累计占比 ≥80% 的热点算子清单**

```bash
# 预期输出示例（基于文档预估）:
# MobileNetV2: Conv1x1(~55%) + DWConv(~25%) + Add(~8%) → 累计 88%
# ResNet50:    Conv3x3(~35%) + Conv1x1(~30%) + BN/Add(~15%) → 累计 80%
# BERT:        MatMul(768x768) + MatMul(768x3072) + LayerNorm → 累计 ~78%
# YOLOv8n:     Conv1x1(多尺度) + Conv3x3 + Concat → 待实测
# MobileViT-S: Conv + MatMul(Attention) → 待实测
```

### Task 1.5: Phase 1 数据整理与 Commit

- [ ] **Step 1.5.1: 整理 Phase 1 所有数据到一份汇总 Markdown**

```bash
cat > results/phase1_cpu_profiling/README.md << 'EOF'
# Phase 1: CPU 全量 Profiling 数据

## 数据文件说明

| 文件 | 内容 | 来源 |
|------|------|------|
| mobilenetv2_mnn_cpu_profile.log | MobileNetV2 MNN CPU Profiling 原始日志 | MNN_PROFILING=1 |
| resnet50_mnn_cpu_profile.log | ResNet50 MNN CPU Profiling 原始日志 | MNN_PROFILING=1 |
| bert_mnn_cpu_profile.log | BERT MNN CPU Profiling 原始日志 | MNN_PROFILING=1 |
| yolov8n_mnn_cpu_profile.log | YOLOv8n MNN CPU Profiling 原始日志 | MNN_PROFILING=1 |
| mobilevit_s_mnn_cpu_profile.log | MobileViT-S MNN CPU Profiling 原始日志 | MNN_PROFILING=1 |
| *_ort_cpu_profile.log | 对应模型 ORT 基线 Profiling | ORT 默认配置 |
| *_simpleperf_stat.log | PMU 硬件计数器 | simpleperf stat |
| *_simpleperf_report.txt | 火焰图符号化报告 | simpleperf report |
| all_models_hotspot_ranking.md | 5 模型热点算子排行汇总 | extract_operator_profile.py |
EOF
```

- [ ] **Step 1.5.2: Commit Phase 1 所有产物**

```bash
git add results/phase1_cpu_profiling/ scripts/extract_operator_profile.py
git add src/backends/ort_backend.cpp  # 如果有 ORT 优化选项修改
git add src/models/model_info.h       # 如果有新模型注册
git commit -m "feat: Phase 1 CPU 全量 Profiling — 5模型热点算子排行 + simpleperf 火焰图

- 5个模型 × MNN CPU 的 MNN_PROFILING 逐算子耗时原始数据
- 5个模型 × ORT CPU 基线 Profiling 对比数据
- simpleperf stat (PMU计数器) + record (火焰图) 交叉验证
- MobileViT-S + Qwen2-0.5B 模型定义注册
- extract_operator_profile.py 算子排行提取脚本
- 热点算子清单: Conv1x1/DWConv/MatMul/LayerNorm 等累计占比≥80%"
```

---

## Phase 2: CPU 热点算子深度 Benchmark（3-5 天）

### Task 2.1: 单算子 Benchmark 框架扩充

- [ ] **Step 2.1.1: 审查现有 single_op_benchmark.cpp 支持的测例**

```bash
grep -n "TEST_CASE\|BENCHMARK_CASE\|registerTest" src/single_op_benchmark.cpp | head -30
# 了解现有支持的算子和测例格式
```

- [ ] **Step 2.1.2: 设计单算子测例数据结构**

在 `src/single_op_benchmark.cpp` 中定义统一的测例描述结构：
```cpp
struct OpTestCase {
    std::string name;           // 如 "Conv1x1_M56_K256_C96_Stride1"
    std::string category;       // "conv1x1", "dwconv", "conv3x3", "matmul", "layernorm", "softmax", "gelu"
    std::string source;         // 来源模型: "mobilenetv2", "resnet50", "bert", "yolov8n", "mobilevit_s"
    int depth;                  // 测试深度: 3=深度(多线程/多精度), 2=中等, 1=快速(基本验证)
    
    // 算子参数
    struct {
        // Conv 参数
        int in_channels, out_channels, kernel_h, kernel_w, stride, pad, groups;
        // MatMul 参数
        int M, N, K;
        // LayerNorm 参数
        int hidden_size, seq_len;
    } params;
    
    // 预期覆盖的执行路径
    std::string execution_path; // "im2col_gemm", "winograd", "depthwise"
};
```

- [ ] **Step 2.1.3: 在 single_op_benchmark.cpp 中注册所有测例**

```cpp
// ====== 第一层: 经典网络提取 (15-20 个 Conv1x1 测例) ======

// MobileNetV2 各 block 的 Conv1x1（从浅层到深层）
REGISTER_OP_TEST(Conv1x1_MBV2_Block1, conv1x1, "mobilenetv2", 3,
    {.in=32, .out=16, .k=1, .s=1, .p=0, .g=1});   // C=32→16, H×W=112²
// ... 更多测例

// ====== 第二层: 形状分桶 (10-15 个) ======
// M维极端(访存密集): M=49(7×7) 小图
REGISTER_OP_TEST(Conv1x1_M49_K256_C512, conv1x1, "shape_bucket", 2,
    {.in=512, .out=256, .k=1, .s=1, .p=0, .g=1, .M=49});
// M维极端(计算密集): M=3136(56×56) 大图
REGISTER_OP_TEST(Conv1x1_M3136_K256_C512, conv1x1, "shape_bucket", 2,
    {.in=512, .out=256, .k=1, .s=1, .p=0, .g=1, .M=3136});

// ====== 第三层: 框架特化路径 (5-8 个) ======
// NCHW4c 对齐退化 (C%4≠0)
REGISTER_OP_TEST(Conv1x1_Misaligned_C31, conv1x1, "special_path", 3,
    {.in=31, .out=64, .k=1, .s=1, .p=0, .g=1});
REGISTER_OP_TEST(Conv1x1_Misaligned_C33, conv1x1, "special_path", 3,
    {.in=33, .out=64, .k=1, .s=1, .p=0, .g=1, .note="C%4≠0退化对比"});
```

### Task 2.2: Conv1x1 / GEMM 深度 Benchmark（≈30 测例）

- [ ] **Step 2.2.1: 经典网络提取 — 15 个 Conv1x1 测例**

```bash
# 从 5 个模型中提取真实 Conv1x1 算子参数，在 single_op_benchmark 中逐一跑
# 每个测例: 100 warmup + 500 iterations
./scripts/build_android.sh
$ADB push build_android/src/single_op_benchmark /data/local/tmp/

# 批量运行
$ADB shell "/data/local/tmp/single_op_benchmark --category conv1x1 --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/conv1x1_network_extract.log
```

- [ ] **Step 2.2.2: 形状分桶 — 10 个测例**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category conv1x1_bucket --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/conv1x1_shape_bucket.log
```

- [ ] **Step 2.2.3: 多线程加速比测试 — 1/2/4/8 线程**

```bash
for threads in 1 2 4 8; do
  $ADB shell "taskset 0-$(($threads-1)) /data/local/tmp/single_op_benchmark \
    --category conv1x1 --threads $threads --iterations 500 --warmup 100" \
    2>&1 | tee results/phase2_hotspot_benchmark/conv1x1_threads_${threads}.log
done
```

- [ ] **Step 2.2.4: FP32 vs FP16 精度对比**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category conv1x1 --precision fp32 --iterations 500" \
  2>&1 | tee results/phase2_hotspot_benchmark/conv1x1_fp32.log
$ADB shell "/data/local/tmp/single_op_benchmark --category conv1x1 --precision fp16 --iterations 500" \
  2>&1 | tee results/phase2_hotspot_benchmark/conv1x1_fp16.log
```

- [ ] **Step 2.2.5: NCHW4c 对齐退化测试 — 5 个测例**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category conv1x1_misaligned --iterations 500" \
  2>&1 | tee results/phase2_hotspot_benchmark/conv1x1_misaligned.log
```

### Task 2.3: DepthwiseConv 深度 Benchmark（≈10 测例）

- [ ] **Step 2.3.1: MobileNetV2 各 block 的 DWConv 测例**

```bash
# DWConv 3×3, C=16→32→64→96→160→320→960
$ADB shell "/data/local/tmp/single_op_benchmark --category dwconv_mbv2 --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/dwconv_mbv2.log
```

- [ ] **Step 2.3.2: DWConv 大通道 vs 小通道对比**

```bash
# 小通道 C=16: 访存量极小，kernel launch 开销可能主导
# 大通道 C=960: 访存密集，带宽瓶颈
$ADB shell "/data/local/tmp/single_op_benchmark --category dwconv_channel_extreme --iterations 500" \
  2>&1 | tee results/phase2_hotspot_benchmark/dwconv_channel_extreme.log
```

- [ ] **Step 2.3.3: DWConv FP32 vs FP16 对比（访存密集 → 带宽是关键）**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category dwconv --precision fp32 --iterations 500" \
  2>&1 | tee results/phase2_hotspot_benchmark/dwconv_fp32.log
$ADB shell "/data/local/tmp/single_op_benchmark --category dwconv --precision fp16 --iterations 500" \
  2>&1 | tee results/phase2_hotspot_benchmark/dwconv_fp16.log
```

### Task 2.4: Conv3×3 深度 Benchmark（≈10 测例）

- [ ] **Step 2.4.1: ResNet50 各阶段 Conv3×3 — Winograd 路径（s=1, d=1）**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category conv3x3_winograd --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/conv3x3_winograd.log
```

- [ ] **Step 2.4.2: ResNet50 各阶段 Conv3×3 — Im2Col 路径（s=2）**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category conv3x3_im2col --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/conv3x3_im2col.log
```

- [ ] **Step 2.4.3: Winograd 变换开销分析 — 不同通道数**

```bash
# Winograd 对 C 敏感: 小C 变换开销占比高，大C 时变换被摊平
$ADB shell "/data/local/tmp/single_op_benchmark --category conv3x3_winograd_channels --iterations 500" \
  2>&1 | tee results/phase2_hotspot_benchmark/conv3x3_winograd_channels.log
```

### Task 2.5: MatMul 深度 Benchmark（≈15 测例，BERT/LLM 核心）

- [ ] **Step 2.5.1: Attention MatMul（方阵）— 768×768, 512×512, 1024×1024**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category matmul_square --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/matmul_square.log
```

- [ ] **Step 2.5.2: FFN MatMul（长矩阵）— 768×3072, 3072×768**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category matmul_long --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/matmul_long.log
```

- [ ] **Step 2.5.3: KV Cache 场景 — M 从小到大增长（模拟自回归生成）**

```bash
# M = 1 (decode 第一步) → 128 → 512 → 1024 → 2048
$ADB shell "/data/local/tmp/single_op_benchmark --category matmul_kv_cache --iterations 200" \
  2>&1 | tee results/phase2_hotspot_benchmark/matmul_kv_cache.log
```

- [ ] **Step 2.5.4: MatMul FP32 vs FP16 vs INT8**

```bash
for prec in fp32 fp16 int8; do
  $ADB shell "/data/local/tmp/single_op_benchmark --category matmul --precision $prec --iterations 500" \
    2>&1 | tee results/phase2_hotspot_benchmark/matmul_${prec}.log
done
```

### Task 2.6: LayerNorm / Softmax / GELU 深度 Benchmark（≈8 测例）

- [ ] **Step 2.6.1: LayerNorm — 不同 hidden_size + seq_len**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category layernorm --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/layernorm.log
# 测例: hidden_size × seq_len: 512×(1,128,512), 768×(1,128,512), 1024×(1,128,512)
```

- [ ] **Step 2.6.2: Softmax — 同上参数组合**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category softmax --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/softmax.log
```

- [ ] **Step 2.6.3: GELU — 同上参数组合**

```bash
$ADB shell "/data/local/tmp/single_op_benchmark --category gelu --iterations 500 --warmup 100" \
  2>&1 | tee results/phase2_hotspot_benchmark/gelu.log
```

### Task 2.7: Roofline 模型分析

- [ ] **Step 2.7.1: 编写 Roofline 分析脚本**

创建 `scripts/analyze_roofline.py`：
```python
#!/usr/bin/env python3
"""Roofline 模型分析: 标注每个算子是 Compute-Bound 还是 Memory-Bound

基于:
  - 骁龙 865 理论峰值算力: A77 @ 2.84GHz, 4×FP32/cycle, 约 45.4 GFLOPS (4×A77)
  - 骁龙 865 理论内存带宽: LPDDR5 2750MHz × 64bit × 4ch = 约 44 GB/s
  (以上均为理论峰值，实测值需要微架构参数修正)
"""
import math

# 骁龙 865 SM8250 参数
PEAK_GFLOPS_FP32 = 45.4   # 4×A77 FP32 理论峰值
PEAK_GFLOPS_FP16 = 90.8   # FP16 理论翻倍（A77 支持 FP16 SIMD）
PEAK_BANDWIDTH_GBS = 34.0  # 实测可用带宽通常低于理论值
ROOFLINE_SLOPE = PEAK_GFLOPS_FP32 * 1e9 / (PEAK_BANDWIDTH_GBS * 1e9)  # FLOPS/Byte

def compute_arithmetic_intensity(op_type, params):
    """计算算子的算术强度 (FLOPS / Byte)"""
    if op_type == "conv1x1":
        M, C_in, C_out = params['M'], params['C_in'], params['C_out']
        flops = 2 * M * C_in * C_out
        bytes_read = 2 * (M * C_in + C_in * C_out)  # input + weight (FP16=2bytes)
        return flops / bytes_read, flops
    elif op_type == "matmul":
        M, N, K = params['M'], params['N'], params['K']
        flops = 2 * M * N * K
        bytes_read = 2 * (M * K + K * N)
        return flops / bytes_read, flops
    # ... 更多算子

def classify_bound(op_type, params, measured_time_ms):
    """判断算子是 Compute-Bound 还是 Memory-Bound"""
    ai, flops = compute_arithmetic_intensity(op_type, params)
    achieved_gflops = flops / (measured_time_ms * 1e6)  # GFLOPS
    if ai >= ROOFLINE_SLOPE:
        bound = "Compute-Bound"
        utilization = achieved_gflops / PEAK_GFLOPS_FP32
    else:
        bound = "Memory-Bound"
        utilization = achieved_gflops / (PEAK_BANDWIDTH_GBS * ai * 1e9 / 1e9)  # 简化
    return bound, utilization, ai, achieved_gflops
```

- [ ] **Step 2.7.2: 对所有热点算子生成 Roofline 散点图**

```bash
python3 scripts/analyze_roofline.py \
  --input results/phase2_hotspot_benchmark/ \
  --output results/phase2_hotspot_benchmark/roofline_chart.png \
  --platform sm8250
# 生成散点图: X轴=算术强度(FLOPS/Byte), Y轴=实际达到的GFLOPS
# 标注每个算子的 Bound 类型和利用率
```

### Task 2.8: Phase 2 数据整理与 Commit

- [ ] **Step 2.8.1: 汇总 Phase 2 所有数据**

创建 `results/phase2_hotspot_benchmark/summary.md`：
```markdown
# Phase 2: CPU 热点算子深度 Benchmark 汇总

## Conv1x1 (30测例)
- 网络提取测例: 15个
- 形状分桶: 10个
- 特化路径: 5个
- 关键发现: [待填入实测数据]

## DepthwiseConv (10测例)
- ...

## Conv3×3 (10测例)
- Winograd vs Im2Col 对比: [待填入]

## MatMul (15测例)
- 方阵 vs 长矩阵对比: [待填入]
- KV Cache 增长曲线: [待填入]

## LayerNorm/Softmax/GELU (8测例)
- ...

## Roofline 分析
- 见 roofline_chart.png
- Compute-Bound 算子列表: [Conv1x1(大M), MatMul(768×768)...]
- Memory-Bound 算子列表: [DWConv(全通道), LayerNorm(小seq_len)...]
```

- [ ] **Step 2.8.2: Commit**

```bash
git add results/phase2_hotspot_benchmark/ scripts/analyze_roofline.py src/single_op_benchmark.cpp
git commit -m "feat: Phase 2 CPU 热点算子深度 Benchmark — 80+测例 + Roofline 分析

- Conv1x1: 30测例 (网络提取15 + 形状分桶10 + 特化路径5)
- DepthwiseConv: 10测例 (通道极端值 + FP16对比)
- Conv3×3: 10测例 (Winograd vs Im2Col)
- MatMul: 15测例 (方阵/长矩阵/KV Cache)
- LayerNorm/Softmax/GELU: 8测例
- Roofline 分析脚本: 标注 Compute-Bound vs Memory-Bound
- 多线程加速比: 1/2/4/8 线程 × 所有Conv1x1测例"
```

---

## Phase 3: GPU 后端启用与对比（3-4 天）

### Task 3.1: 启用 MNN OpenCL 后端

- [ ] **Step 3.1.1: 检查 MNN 编译是否包含 OpenCL**

```bash
grep "MNN_OPENCL" third_party/MNN/CMakeLists.txt | head -10
# 检查是否有 MNN_OPENCL 选项
grep -r "MNN_OPENCL" CMakeLists.txt | head -5
```

- [ ] **Step 3.1.2: 修改 CMakeLists.txt 启用 MNN OpenCL 编译**

在 `CMakeLists.txt` 或 `third_party/CMakeLists.txt` 中添加/修改：
```cmake
# MNN OpenCL 支持
option(MNN_OPENCL "Enable OpenCL GPU backend for MNN" ON)
set(MNN_OPENCL ON CACHE BOOL "Enable OpenCL" FORCE)

# 在 MNN 的 add_subdirectory 之前设置
if(MNN_OPENCL)
    add_definitions(-DMNN_OPENCL=1)
endif()
```

- [ ] **Step 3.1.3: 确认设备 OpenCL 支持**

```bash
$ADB shell "ls /system/vendor/lib64/libOpenCL.so" && echo "OpenCL 库存在" || echo "需要安装 OpenCL 库"
$ADB shell "ls /vendor/lib64/egl/libGLESv2_adreno.so" && echo "Adreno GPU 驱动存在"
```

- [ ] **Step 3.1.4: 修改 mnn_backend.cpp 增加 GPU 推理路径**

在 `src/backends/mnn_backend.cpp` 中添加 GPU 后端初始化：
```cpp
class MNNBackend : public BenchmarkBackend {
private:
    BackendType backend_type_;  // MNN_CPU 或 MNN_GPU
    MNN::ScheduleConfig config_;
    
public:
    bool Init(const BenchmarkConfig& config) override {
        if (config.backend == BackendType::MNN_GPU) {
            // 设置 OpenCL 后端
            config_.type = MNN_FORWARD_OPENCL;
            // GPU 内存模式
            config_.backupType = MNN_FORWARD_CPU;  // 不支持 GPU 的算子回退 CPU
            backend_type_ = BackendType::MNN_GPU;
        } else {
            config_.type = MNN_FORWARD_CPU;
            config_.numThread = config.num_threads;
            backend_type_ = BackendType::MNN_CPU;
        }
        
        // 创建 Session
        auto net = LoadModel(config.model_path);
        session_ = net->createSession(config_);
        return session_ != nullptr;
    }
    
    std::string GetBackendName() const override {
        return (backend_type_ == BackendType::MNN_GPU) ? "MNN_OpenCL" : "MNN_CPU";
    }
};
```

- [ ] **Step 3.1.5: 修改 config.h 增加 MNN_GPU 枚举值**

在 `src/common/config.h` 中：
```cpp
enum class BackendType {
    MNN_CPU = 0,
    MNN_GPU = 1,     // 新增
    ORT_CPU = 2,
    NCNN = 3,
    // ...
};
```

- [ ] **Step 3.1.6: 编译并验证 GPU 后端可加载**

```bash
./scripts/build_android.sh 2>&1 | tee results/phase3_gpu_comparison/build_gpu.log

# 快速冒烟测试
$ADB push build_android/src/benchmark_inference /data/local/tmp/
$ADB shell "/data/local/tmp/benchmark_inference --model mobilenetv2 --backend mnn_gpu --iterations 10" \
  2>&1 | tee results/phase3_gpu_comparison/gpu_smoke_test.log
# 检查日志是否有 "MNN_OpenCL" 输出和正确的推理结果
```

- [ ] **Step 3.1.7: 精度验证 — GPU vs CPU 输出余弦相似度 ≥ 0.999**

```bash
$ADB shell "/data/local/tmp/benchmark_inference --model mobilenetv2 --backend mnn_gpu --verify mnn_cpu --iterations 100" \
  2>&1 | tee results/phase3_gpu_comparison/gpu_precision_validation.log
# 检查余弦相似度输出
```

### Task 3.2: CPU vs GPU 全模型端到端对比

- [ ] **Step 3.2.1: 5 模型 × (MNN_CPU + MNN_GPU) 端到端 Latency**

```bash
for model in mobilenetv2 resnet50 bert yolov8n mobilevit_s; do
  for backend in mnn_cpu mnn_gpu; do
    ./scripts/run_benchmark_android.sh --model $model --backend $backend \
      --iterations 100 --warmup 50 \
      2>&1 | tee results/phase3_gpu_comparison/${model}_${backend}_e2e.log
  done
done
```

- [ ] **Step 3.2.2: 记录 5 模型的初始化时间对比（GPU 额外开销）**

```bash
# GPU 首次推理需要 OpenCL 编译和初始化
for model in mobilenetv2 resnet50 bert yolov8n; do
  $ADB shell "/data/local/tmp/benchmark_inference --model $model --backend mnn_gpu --report-init --iterations 10" \
    2>&1 | tee results/phase3_gpu_comparison/${model}_gpu_init.log
  $ADB shell "/data/local/tmp/benchmark_inference --model $model --backend mnn_cpu --report-init --iterations 10" \
    2>&1 | tee results/phase3_gpu_comparison/${model}_cpu_init.log
done
```

- [ ] **Step 3.2.3: 计算 CPU vs GPU 加速比**

```bash
python3 scripts/generate_report.py \
  --input-dir results/phase3_gpu_comparison/ \
  --metric latency \
  --compare mnn_cpu mnn_gpu \
  --output results/phase3_gpu_comparison/cpu_vs_gpu_comparison.md
```

### Task 3.3: CPU vs GPU 逐算子对比（关键分析）

- [ ] **Step 3.3.1: 拆分每个模型到算子级，CPU vs GPU 逐算子对比**

```bash
# 需要修改 MNN backend 在 GPU 模式下也输出逐算子 Profiling
# 使用 MNN_PROFILING=1 + MNN_FORWARD_OPENCL

for model in mobilenetv2 resnet50 bert yolov8n; do
  MNN_PROFILING=1 $ADB shell "/data/local/tmp/benchmark_inference \
    --model $model --backend mnn_gpu --iterations 100" \
    2>&1 | tee results/phase3_gpu_comparison/${model}_gpu_operator.log
done
```

- [ ] **Step 3.3.2: 编写 CPU vs GPU 逐算子对比脚本**

```python
#!/usr/bin/env python3
# scripts/compare_cpu_gpu_operators.py
"""逐算子对比 CPU vs GPU 耗时，标注 GPU 加速比"""
# 输入: phase1 的 CPU Profiling (MNN CPU) + phase3 的 GPU Profiling
# 输出: 每个算子的 CPU耗时 / GPU耗时 / 加速比 / 推荐后端
```

- [ ] **Step 3.3.3: 输出异构调度判断规则**

```markdown
# 基于实测数据的算子放置规则:
- GEMM (Conv1x1, MatMul):     GPU 加速 2-5× → 放 GPU
- DWConv (访存密集):           GPU 几乎无加速 → 留 CPU
- LayerNorm (小张量):          GPU 启动开销 > 计算 → 留 CPU  
- Softmax (中等张量):          视具体情况
- 逐元素算子 (Add, Mul, ReLU):  GPU 启动开销 >> 计算 → 必须留 CPU
```

### Task 3.4: 跨后端数据传输开销分析

- [ ] **Step 3.4.1: 测量 CPU→GPU→CPU 数据拷贝时间**

```bash
# 对不同大小的张量测量传输开销
$ADB shell "/data/local/tmp/single_op_benchmark --benchmark data_transfer --sizes 1,4,16,64,256" \
  2>&1 | tee results/phase3_gpu_comparison/data_transfer_cost.log
```

- [ ] **Step 3.4.2: 计算"GPU 加速 - 传输开销"的净收益**

```markdown
# 核心公式:
# Net_Speedup = GPU_Compute_Speedup / (1 + T_transfer / T_compute_cpu)
#
# 小算子 (T_compute_cpu < T_transfer): Net_Speedup < 1 → 放 CPU
# 大算子 (T_compute_cpu >> T_transfer): Net_Speedup ≈ GPU_Speedup → 放 GPU
```

- [ ] **Step 3.4.3: Commit Phase 3**

```bash
git add results/phase3_gpu_comparison/ \
        scripts/compare_cpu_gpu_operators.py \
        src/backends/mnn_backend.cpp \
        src/common/config.h \
        CMakeLists.txt
git commit -m "feat: Phase 3 GPU OpenCL 后端启用与 CPU vs GPU 全矩阵对比

- MNN OpenCL 后端编译和加载
- 5模型 × 2后端 端到端 Latency 全量对比
- CPU vs GPU 逐算子拆解 + 加速比分析
- GPU 初始化 + 数据传输开销量化
- 异构调度规则: GEMM放GPU, DWConv/LayerNorm留CPU
- 精度验证: GPU vs CPU 余弦相似度≥0.999"
```

---

## Phase 4: 关键优化闭环 — BERT 反常现象（7-10 天）

> **这是面试核心故事，绝对不能跳过。**

### Task 4.1: BERT MNN vs ORT 逐算子 Gap 分析

- [ ] **Step 4.1.1: 确认 BERT MNN 比 ORT 慢 15% 的现象可复现**

```bash
# 独立跑 3 次确认结果一致（排除随机波动）
for run in 1 2 3; do
  echo "=== Run $run ==="
  ./scripts/run_benchmark_android.sh --model bert --backend mnn --iterations 200 --warmup 50 \
    2>&1 | grep "avg latency" | tee -a results/phase4_bert_optimization/bert_mnn_repro_${run}.log
  ./scripts/run_benchmark_android.sh --model bert --backend ort --iterations 200 --warmup 50 \
    2>&1 | grep "avg latency" | tee -a results/phase4_bert_optimization/bert_ort_repro_${run}.log
done
# 验证: MNN avg > ORT avg × 1.10 以上（15% 差距）
```

- [ ] **Step 4.1.2: BERT MNN Profiler — 拆出每个算子的 CPU 耗时**

```bash
MNN_PROFILING=1 ./scripts/run_benchmark_android.sh \
  --model bert --backend mnn --iterations 100 \
  2>&1 | tee results/phase4_bert_optimization/bert_mnn_op_breakdown.log

# 提取关键算子耗时
python3 scripts/extract_operator_profile.py \
  results/phase4_bert_optimization/bert_mnn_op_breakdown.log \
  > results/phase4_bert_optimization/bert_mnn_op_ranking.md
```

- [ ] **Step 4.1.3: BERT ORT Profiler — 拆出每个算子的 CPU 耗时**

```bash
# 对 ORT 也需要类似的算子级 Profiling
# 方法1: 开启 ORT 的 VERBOSE 日志
# 方法2: 用 simpleperf 火焰图反推
$ADB shell su -c "/data/local/tmp/simpleperf record \
  -e cpu-cycles -g --duration 15 --call-graph fp \
  -o /data/local/tmp/perf_bert_ort.data \
  /data/local/tmp/benchmark_inference --model bert --backend ort --iterations 50"

$ADB pull /data/local/tmp/perf_bert_ort.data results/phase4_bert_optimization/
simpleperf report -i results/phase4_bert_optimization/perf_bert_ort.data \
  --sort comm,dso,symbol > results/phase4_bert_optimization/bert_ort_symbols.txt
```

- [ ] **Step 4.1.4: 逐算子对表 — 找出 MNN 最慢的那个算子**

```markdown
| BERT 算子 | MNN CPU 耗时(ms) | ORT CPU 耗时(ms) | MNN/ORT 比值 |
|-----------|-----------------|-----------------|-------------|
| MatMul(768×768) | ? | ? | ? |
| MatMul(768×3072) | ? | ? | ? |
| LayerNorm(768) | ? | ? | ? |
| Softmax(768) | ? | ? | ? |
| GELU(768) | ? | ? | ? |
| Add(768) | ? | ? | ? |
| ... | ... | ... | ... |
```

### Task 4.2: 根因分析 — 为什么 MNN 的 GEMM 在 768 尺寸上不如 ORT

- [ ] **Step 4.2.1: GEMM 分块策略对比分析**

```bash
# 阅读 MNN 源码中的 GEMM 实现
find third_party/MNN/source/backend/cpu/ -name "*MatMul*" -o -name "*GEMM*" -o -name "*Matrix*" | head -10
# 找到 MNN GEMM 微内核文件，分析分块参数
grep -r "BLOCK_M\|BLOCK_N\|BLOCK_K\|KERNEL_SIZE\|TILE" \
  third_party/MNN/source/backend/cpu/arm/arm64/ --include="*.cpp" --include="*.h" | head -20
```

分析要点：
```markdown
MNN GEMM 关键参数:
- 寄存器分块大小 (mr × nr): 通常 8×8 或 12×8
- L1/L2 Cache 分块: 对齐到 32KB L1 / 256KB L2
- Pack 策略: NCHW4c 布局下的 Pack 开销

为什么 768 这个尺寸对 MNN 不利:
- 假设 mr=8, M=768 能被 8 整除 → 理论上对齐
- 但 K=768 对应的 Cache 分块如果失配，会导致 L2 cache thrashing
- NCHW4c 布局: 768/4=192 → 对齐 OK
```

- [ ] **Step 4.2.2: 用 simpleperf stat 验证 Cache Miss 假说**

```bash
# 单跑 BERT MatMul 算子，抓 PMU 计数器
$ADB shell su -c "/data/local/tmp/simpleperf stat \
  -e cache-misses,cache-references,L1-dcache-load-misses,ll_cache_miss_rd \
  --duration 10 \
  /data/local/tmp/single_op_benchmark --category matmul_768_768 --iterations 200" \
  2>&1 | tee results/phase4_bert_optimization/matmul_768_cache_stat.log
```

- [ ] **Step 4.2.3: 对比 ORT 的 GEMM 实现**

```bash
# ORT 使用 MLAS (Microsoft Linear Algebra Subprograms) 或 Eigen
# 找到 ORT 相关源码
find third_party/onnxruntime/ -name "*gemm*" -o -name "*matmul*" | head -10
# 对比分块参数
```

### Task 4.3: 针对性优化实现

基于 4.2 的分析结果，假设根因是 MNN 在 K=768/C=%4=0 场景下的分块参数失配：

- [ ] **Step 4.3.1: 定位需要修改的 MNN GEMM 微内核文件**

```bash
# 预计在以下位置之一:
# third_party/MNN/source/backend/cpu/arm/arm64/MNNMatMul_DGEMM_12x8.S
# third_party/MNN/source/backend/cpu/arm/arm64/MNNPackC4ForMatMul_A.S
# third_party/MNN/source/backend/cpu/ComputeMatMul.cpp
```

- [ ] **Step 4.3.2: 编写/修改 GEMM 微内核参数（假设调整 K 维分块）**

```cpp
// 示例: 在 ComputeMatMul.cpp 中调整 GEMM 分块参数
// 针对 K=768 场景，增加 K_BLOCK_SIZE 以更好利用 L2 Cache

// Before:
constexpr int K_BLOCK = 256;  // 原始分块

// After (针对 K=768 优化):
constexpr int K_BLOCK_768 = 384;  // 768/2 = 384, 一个 MatMul 只需2次分块迭代
// 或者更激进:
constexpr int K_BLOCK_768 = 768;  // 整个 K 维一次做完，利用 L2 Cache (256KB)
```

- [ ] **Step 4.3.3: 编译并跑 Before/After 对比**

```bash
# Before: 记录基线
./scripts/build_android.sh
./scripts/run_benchmark_android.sh --model bert --backend mnn --iterations 200 \
  2>&1 | tee results/phase4_bert_optimization/bert_before_fix.log

# 应用优化
# (编辑对应文件)

# After: 记录优化后
./scripts/build_android.sh
./scripts/run_benchmark_android.sh --model bert --backend mnn --iterations 200 \
  2>&1 | tee results/phase4_bert_optimization/bert_after_fix.log
```

- [ ] **Step 4.3.4: 验证其他模型不受影响（回归测试）**

```bash
for model in mobilenetv2 resnet50 yolov8n; do
  ./scripts/run_benchmark_android.sh --model $model --backend mnn --iterations 100 \
    2>&1 | tee results/phase4_bert_optimization/regression_${model}_after.log
done
# 对比 Phase 1 基线数据，确保 CV 模型性能未退化
```

- [ ] **Step 4.3.5: 如果 GEMM 优化无法追平 ORT — 探索算子融合方案**

```markdown
MNN 可能在某些算子融合 (Fusion) 上不如 ORT:
- ORT 的 Graph Optimizer 可能将 MatMul + Add + GELU 融合为单一算子
- 而 MNN 可能缺少这个 Fusion Pattern

修复方法:
1. 在 MNN 的 Post-Transform 优化中增加 Fusion Pattern
2. 或者手动实现 MatMul+Bias+GELU 融合算子

检查:
grep -r "FusedMatMul\|MatMulAddGelu\|GeluFusion" third_party/MNN/source/ --include="*.cpp" | head -10
```

### Task 4.4: Neon 指令级优化（如需要）

如果 GEMM 分块调整不够，进一步做指令级优化：

- [ ] **Step 4.4.1: 定位 MNN GEMM 微内核热点指令**

```bash
# 用 simpleperf annotate 查看 GEMM 内核的指令级热点
simpleperf annotate -i results/phase4_bert_optimization/perf_bert_mnn.data \
  --symbol "MNNMatMul" > results/phase4_bert_optimization/annotate_gemm.txt
```

- [ ] **Step 4.4.2: 用 Neon Intrinsics 重写热点循环体**

```cpp
// 示例: 利用 SDOT (Signed Dot Product) 指令加速
// ARMv8.2-A 支持 SDOT/UDOT，骁龙865 A77 完美支持
#include <arm_neon.h>

// 4×4 FP32 矩阵乘 (使用 vfmal 融合乘加)
inline void gemm_kernel_4x4(const float* A, const float* B, float* C,
                              int K, int lda, int ldb) {
    float32x4_t c0 = vld1q_f32(C + 0*lda);
    float32x4_t c1 = vld1q_f32(C + 1*lda);
    float32x4_t c2 = vld1q_f32(C + 2*lda);
    float32x4_t c3 = vld1q_f32(C + 3*lda);
    
    for (int k = 0; k < K; k++) {
        float32x4_t b = vld1q_f32(B + k*ldb);
        c0 = vfmaq_laneq_f32(c0, b, vld1q_f32(A + k*lda + 0), 0);
        // ... 展开循环体
    }
    
    vst1q_f32(C + 0*lda, c0);
    // ...
}
```

- [ ] **Step 4.4.3: 对比优化前后的性能**

```markdown
| 优化步骤 | BERT Latency (ms) | vs ORT | vs MNN 基线 |
|---------|------------------|--------|-----------|
| MNN 基线 | ? | 慢 15% | — |
| GEMM 分块调优 | ? | ? | ? |
| Neon 指令重写 | ? | ? | ? |
| 算子融合 | ? | ? | ? |
| **最终目标** | ≤598ms | 追平或反超 | — |
```

### Task 4.5: BERT 优化报告撰写

- [ ] **Step 4.5.1: 撰写 BERT 优化闭环报告**

创建 `results/phase4_bert_optimization/bert_optimization_report.md`：
```markdown
# BERT MNN 性能优化闭环报告

## 1. 问题发现
- BERT 在 MNN CPU 上推理耗时 689ms，比 ONNX Runtime (598ms) 慢 15%
- 而 CV 模型上 MNN 全面领先 1.3-2.4× → 这是一个异常信号

## 2. 现象定位
- MNN Profiler 逐算子拆解: [具体数据]
- ORT Profiler 逐算子拆解: [具体数据]
- 最大差距算子: MatMul(768×768): MNN XXms vs ORT YYms

## 3. 根因分析
- simpleperf PMU 计数器: cache-misses 对比
- Roofline 标注: MatMul(768) 在 MNN 上未达到峰值算力
- 根因: [分块策略/Cache利用/指令调度 — 具体原因]

## 4. 优化方案
- 方案 A: GEMM 分块参数调整 [具体参数]
- 方案 B: Neon SDOT 指令重写 [代码 diff]
- 方案 C: MNN Post-Transform Fusion 规则补充

## 5. 效果验证
| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| BERT 总耗时(ms) | 689 | [目标≤598] | [目标≥13%] |
| MatMul(768×768)(ms) | ? | ? | ? |
| 其他模型回归 | — | [无退化] | — |

## 6. 经验总结
- 为什么 MNN 在 768 这个尺寸上吃亏: [分析]
- 不同 GEMM 尺寸的最优分块策略规律: [总结]
- 对 MNN 社区的建议 (可提 PR)
```

- [ ] **Step 4.5.2: Commit Phase 4**

```bash
git add results/phase4_bert_optimization/ \
        third_party/MNN/source/backend/cpu/  # 如果有 MNN 源码修改
git commit -m "feat: Phase 4 BERT 性能优化闭环 — MatMul(768)根因分析与GEMM分块优化

问题: BERT MNN 比 ORT 慢 15% (689ms vs 598ms)
根因: [填具体根因, 如: MNN GEMM 在 K=768 时 Cache 分块失配]
优化: [填具体改动, 如: K_BLOCK 256→384, 配合 Neon SDOT 指令]
效果: BERT 推理 XXms → XXms (提升 XX%), 追平/反超 ORT
验证: CV 模型回归测试无退化"
```

---

## Phase 5: LLM 推理专项（3-5 天）

### Task 5.1: llama.cpp 启用 + Qwen2-0.5B 加载

- [ ] **Step 5.1.1: 恢复 llama.cpp 后端编译**

```bash
# 修改 CMakeLists.txt
# 将 BENCHMARK_LLAMACPP 从 OFF 改为 ON

# 检查 third_party/llama.cpp 子模块状态
git submodule status third_party/llama.cpp

# 编译
./scripts/build_android.sh 2>&1 | tee results/phase5_llm_profiling/build_llamacpp.log
```

- [ ] **Step 5.1.2: 验证 Qwen2-0.5B GGUF 模型可加载**

```bash
$ADB push models/nlp/qwen2_0.5b/*.gguf /data/local/tmp/models/
$ADB shell "/data/local/tmp/benchmark_inference --model qwen2_0.5b --backend llamacpp \
  --prompt '你好，世界' --max-tokens 10" \
  2>&1 | tee results/phase5_llm_profiling/qwen_load_test.log
# 验证能正常输出 token
```

- [ ] **Step 5.1.3: 修改 llamacpp_backend.cpp 增加 Profiling 钩子**

```cpp
class LlamaCppBackend : public BenchmarkBackend {
    // 增加 Prefill/Decode 阶段的分开计时
    struct LLMTiming {
        double prefill_ms;       // Prefill 阶段耗时
        double decode_per_token_ms;  // 每个 token 的 Decode 耗时
        double total_ms;         // 总耗时
        int prefill_tokens;      // Prompt token 数
        int generated_tokens;    // 生成的 token 数
    };
    
    LLMTiming ProfileGeneration(const std::string& prompt, int max_tokens) {
        // 分开计时 Prefill 和 Decode
    }
};
```

### Task 5.2: Qwen2-0.5B Prefill & Decode Profiling

- [ ] **Step 5.2.1: Prefill 阶段 Profiling**

```bash
# 用 simpleperf stat 抓 Prefill 阶段的 PMU 数据
# Prefill 是计算密集：一次性处理所有 prompt tokens
$ADB shell su -c "/data/local/tmp/simpleperf stat \
  -e cpu-cycles,instructions,cache-misses,branch-misses \
  --duration 15 \
  /data/local/tmp/benchmark_inference --model qwen2_0.5b --backend llamacpp \
  --prompt-file /data/local/tmp/prompt_512.txt --max-tokens 1" \
  2>&1 | tee results/phase5_llm_profiling/prefill_pmu.log
```

- [ ] **Step 5.2.2: Decode 阶段 Profiling**

```bash
# Decode 是访存密集: 每个 token 主要访问 KV Cache
# 抓 memory bandwidth 相关的 PMU 事件
$ADB shell su -c "/data/local/tmp/simpleperf stat \
  -e cpu-cycles,instructions,bus-access,bus-cycles,stalled-cycles-backend \
  --duration 20 \
  /data/local/tmp/benchmark_inference --model qwen2_0.5b --backend llamacpp \
  --prompt '你好' --max-tokens 50" \
  2>&1 | tee results/phase5_llm_profiling/decode_pmu.log
```

- [ ] **Step 5.2.3: 记录 Prefill vs Decode 的核心指标对比**

```markdown
| 指标 | Prefill (512 tokens) | Decode (per token) |
|------|--------------------|-------------------|
| 耗时 (ms) | ? | ? |
| 吞吐 (tokens/s) | ? | ? |
| IPC (instructions/cycle) | ? (高→计算密集) | ? (低→访存密集) |
| Cache Miss Rate | ? (低) | ? (高, KV Cache 访问) |
| Memory Bandwidth (GB/s) | ? (中等) | ? (接近峰值) |
| 算力利用率 | ? | ? (<5%, 几乎全部 stall) |
```

### Task 5.3: MNN 的 llm.cpp vs llama.cpp 对比

- [ ] **Step 5.3.1: 用 MNN 的 llm.cpp 加载 Qwen2-0.5B**

```bash
# MNN 的 LLM 推理组件在 tools/cpp/ 或 express/
# 查看是否支持 Qwen2 的模型格式
find third_party/MNN/ -name "*llm*" -o -name "*LLM*" | head -15
```

- [ ] **Step 5.3.2: MNN vs llama.cpp 的 Prefill 速度对比**

```bash
# MNN llm.cpp Prefill
$ADB shell "/data/local/tmp/mnn_llm_bench --model qwen2_0.5b.mnn \
  --prompt-file /data/local/tmp/prompt_512.txt --max-tokens 1" \
  2>&1 | tee results/phase5_llm_profiling/mnn_llm_prefill.log

# llama.cpp Prefill
$ADB shell "/data/local/tmp/benchmark_inference --model qwen2_0.5b --backend llamacpp \
  --prompt-file /data/local/tmp/prompt_512.txt --max-tokens 1" \
  2>&1 | tee results/phase5_llm_profiling/llamacpp_prefill.log
```

- [ ] **Step 5.3.3: MNN vs llama.cpp 的 Decode 速度对比**

```markdown
| 框架 | Prefill (tok/s) | Decode (tok/s) | 内存峰值(GB) | KV Cache 管理 |
|------|----------------|----------------|-------------|-------------|
| MNN llm.cpp | ? | ? | ? | MNN Tensor 管理 |
| llama.cpp | ? | ? | ? | llama.cpp 自管理 |
| ORT (GenAI) | ? | ? | ? | ORT IOBinding |
```

### Task 5.4: LLM 推理优化分析

- [ ] **Step 5.4.1: 分析 Decode 阶段的内存瓶颈**

```markdown
Qwen2-0.5B Decode 阶段每 token 的内存访问量:
- KV Cache 大小: 2 × layers × hidden_size × seq_len × sizeof(FP16)
  = 2 × 24 × 896 × seq_len × 2 bytes
  = ~84 KB × seq_len
- 当 seq_len=2048 时: 约 172 MB 的 KV Cache
- 每个 token 需要完整扫描 KV Cache → 这就是为什么 Decode 是访存密集

量化优化:
- KV Cache INT8 量化: 172MB → 86MB, 带宽压力减半
- KV Cache INT4 量化: 172MB → 43MB
```

- [ ] **Step 5.4.2: 输出 LLM 推理优化建议（不要求全部实现）**

```markdown
# LLM 推理优化清单 (面试时可逐条展开)

## 短期可做 (MNN 当前框架内):
1. KV Cache INT8 量化 → 预计 Decode 提速 30-50%
2. 算子融合: QKV Projection 合并为单次 MatMul → 减少 Kernel Launch
3. 使用 ARM SDOT 指令加速 Attention Score 计算

## 中期需要 (框架级改动):
4. FlashAttention: 减少 KV Cache 的 HBM 读取
5. Continuous Batching: 提高 GPU/CPU 利用率
6. Speculative Decoding: 小模型 draft + 大模型 verify

## 长期方向 (架构选择):
7. 权重 INT4 + KV Cache INT8 = 纯量化推理管线
8. CPU-GPU 异构: Prefill 放 GPU, Decode 放 CPU
```

### Task 5.5: Commit Phase 5

```bash
git add results/phase5_llm_profiling/ \
        src/backends/llamacpp_backend.cpp \
        CMakeLists.txt
git commit -m "feat: Phase 5 LLM 推理专项 — Qwen2-0.5B Prefill/Decode Profiling + MNN vs llama.cpp 对比

- llama.cpp 后端恢复编译 + Qwen2-0.5B GGUF 加载
- Prefill/Decode 分阶段 Profiling (PMU 计数器)
- Prefill=计算密集 vs Decode=访存密集 的量化验证
- KV Cache 内存访问模式分析 + 量化优化建议
- MNN llm.cpp vs llama.cpp 的吞吐/内存对比
- LLM 推理优化建议清单 (FlashAttention, KV Cache量化, 异构调度)"
```

---

## Phase 6: 收尾文档与面试准备（2-3 天）

### Task 6.1: 全量数据汇总为结构化报告

- [ ] **Step 6.1.1: 编写综合性能报告生成脚本**

创建 `scripts/generate_final_report.py`：
```python
#!/usr/bin/env python3
"""
综合性能报告生成器
输入: Phase 1-5 的所有数据
输出: 一份完整的 Markdown 报告 + HTML 可视化
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_phase_data():
    """加载 Phase 1-5 的所有 CSV/日志数据"""
    data = {}
    
    # Phase 1: CPU 全量 Profiling
    data['phase1'] = {
        'mobilenetv2_mnn': parse_log('results/phase1_cpu_profiling/mobilenetv2_mnn_cpu_profile.log'),
        'resnet50_mnn': parse_log('results/phase1_cpu_profiling/resnet50_mnn_cpu_profile.log'),
        # ...
    }
    # Phase 2: 热点算子
    # Phase 3: GPU 对比
    # Phase 4: BERT 优化
    # Phase 5: LLM Profiling
    
    return data

def generate_operator_comparison_chart(data, output_path):
    """生成算子性能对比柱状图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    # Conv1x1 对比
    # DWConv 对比
    # Conv3×3 对比
    # MatMul 对比
    # 多线程加速比
    # CPU vs GPU 加速比
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

def generate_roofline_scatter(data, output_path):
    """生成 Roofline 散点图"""
    # X轴: Arithmetic Intensity (FLOPS/Byte)
    # Y轴: Achieved GFLOPS
    # 标注每个算子的 Bound 类型

def generate_timeline_chart(data, output_path):
    """生成优化前后对比时间线图"""
    # BERT 优化各步骤的 Latency 变化

if __name__ == '__main__':
    data = load_all_phase_data()
    generate_operator_comparison_chart(data, 'results/final_operator_comparison.png')
    generate_roofline_scatter(data, 'results/roofline_scatter.png')
    generate_timeline_chart(data, 'results/bert_optimization_timeline.png')
```

- [ ] **Step 6.1.2: 运行报告生成**

```bash
python3 scripts/generate_final_report.py --all-phases \
  --output results/final_comprehensive_report.md
# 生成 3-5 张关键可视化图表
```

### Task 6.2: 产出每个模型的性能分析卡片

- [ ] **Step 6.2.1: 为每个模型产出一页分析卡片**

```markdown
# MobileNetV2 性能分析卡片

**计算特征:** Conv1x1 密集 + DWConv 访存密集
**热点算子 Top 3:**
1. Conv1x1 (55%) — Compute-Bound, GPU 加速 3.2×
2. DWConv (25%) — Memory-Bound, GPU 几乎无加速
3. Add (8%) — 逐元素, GPU 启动开销 > 计算

**MNN vs ORT:** MNN 领先 1.4× (整体), 主要是 Conv1x1 实现更优
**优化潜力:** DWConv 是大头但访存密集，优化空间有限
**异构建议:** Conv1x1 放 GPU, DWConv 和逐元素留 CPU
```

- [ ] **Step 6.2.2: 对 ResNet50, BERT, YOLOv8n, MobileViT-S 各产出一张卡片**

### Task 6.3: 面试准备材料

- [ ] **Step 6.3.1: 撰写项目讲述话术（1分钟 + 3分钟 + 追问应对）**

```markdown
# 1分钟版（总括）

"我基于骁龙 865 搭了一套完整的端侧推理性能分析体系。
选了 5 个模型覆盖 CV 到 LLM 的五种计算模式，
用 MNN Profiler + simpleperf 做全量 Profiling，
按二八原则对热点算子做三层深度的 Benchmark。
然后用这个体系发现了 MNN 上 BERT 的异常瓶颈，
定位到 MatMul(768) 的 GEMM 分块参数失配，
调整分块策略 + Neon SDOT 指令重写后
BERT 推理从 689ms 降到 [XX]ms，[追平/反超] ORT。
还做了 CPU vs GPU 的逐算子异构调度分析，
对 LLM 推理的 Prefill/Decode 计算特征做了量化对比。"

# 3分钟版（展开）

[1 min] 项目背景: 为什么做 / 平台选择 / 5 模型选择逻辑
  技术栈: MNN + ORT + llama.cpp, 骁龙865 ARMv8.2-A

[1 min] Benchmark 设计方法论
  - Profiling 驱动 + 二八原则 (累计≥80%的热点算子深度挖)
  - 三层覆盖: 经典网络提取 + 形状分桶 + 框架特化路径
  - Roofline 模型标注: Compute-Bound vs Memory-Bound
  - simpleperf PMU 计数器交叉验证

[1 min] 核心发现 + 优化闭环
  - BERT 反转现象: CV 模型 MNN 领先 1.3-2.4×, 但 BERT 反而慢 15%
  - 根因定位: GEMM 在 K=768 矩阵尺寸下的 Cache 分块失配
  - 优化方案 + 效果: [具体数字]
  - CPU vs GPU 异构调度核心发现

# 追问应对

Q: "只测了一台手机？"
A: "是的，但我在这台手机上开了 CPU 和 GPU 两个后端做深度对比。
    多设备的价值在于架构差异分析，而我当前的优先级是把单设备的
    性能特征挖透——这和写算子是一个思路，先做深再做宽。
    后续计划是骁龙 8 Gen 2 (独立 NPU) 做高低端架构对比。"

Q: "为什么不用 ncnn？"
A: "ncnn 和 MNN 的优化路线完全一致——端侧轻量级手写 Neon 优化。
    加了 ncnn 不会增加分析维度，而 MNN 的工具链 (Profiler/可视化) 
    更适合做深度分析。我选的 4 个框架各代表一种不同的优化哲学。"

Q: "Roofline 模型对你实际优化有什么帮助？"
A: "举个例子——DWConv 在 Roofline 上落在 Memory-Bound 区域，
    意味着优化方向应该是减少内存访问 (如 channel shuffle, 
    内存布局优化)，而不是花时间调计算指令。反过来 Conv1x1 在 
    Compute-Bound 区域，应该专注 SIMD 指令利用率和多线程并行。
    Roofline 帮我避免了方向性的错误。"
```

- [ ] **Step 6.3.2: 更新简历项目描述**

```markdown
★ ARM端多框架推理性能Benchmark与优化实践
项目使用技术: C++ / ARM Neon / MNN / ONNX Runtime / llama.cpp / simpleperf
硬件平台: 骁龙 865 (红米 K30S)  — Cortex-A77 + A55 + Adreno 650 GPU
项目描述: 建立端侧推理框架标准化性能Benchmark体系，以MNN为核心分析对象，
深度对比ORT/llama.cpp三种优化路线，输出系统级性能分析方法论

核心工作:
1. 搭建 Profiling 驱动的 Benchmark 体系，覆盖 5 模型 × 4 框架 × 2 后端
2. 按二八原则对热点算子做三层深度 Benchmark (80+测例)，结合 Roofline 模型
   标注算子的 Compute-Bound vs Memory-Bound 属性
3. 基于 ARM PMU 性能计数器 (simpleperf) 做性能根因分析——不仅仅是"慢"
4. 对比分析 MNN 与 ORT 的性能 Gap，发现并修复 BERT 在 MNN 上的 GEMM 瓶颈
5. 启用 MNN OpenCL GPU 后端，完成 CPU vs GPU 逐算子对比和异构调度分析
6. Qwen2-0.5B LLM 推理 Profiling，量化 Prefill=计算密集 vs Decode=访存密集

项目成果:
- MNN BERT 推理性能提升 [XX%]，[追平/反超] ONNX Runtime
- 建立端侧推理性能数据库: 5模型 × 4框架 × 80+算子测例 × 3精度模式
- 输出 CPU vs GPU 异构调度判断规则 (哪些算子适合GPU、哪些必须CPU)
- 自动化脚本一键生成 5 张可视化图表 + 结构化性能报告
```

### Task 6.4: 最终 Commit + 项目收尾

- [ ] **Step 6.4.1: 生成最终的综合报告**

```bash
python3 scripts/generate_final_report.py --all-phases --format html \
  --output results/final_comprehensive_report.html
python3 scripts/generate_final_report.py --all-phases --format md \
  --output results/final_comprehensive_report.md
```

- [ ] **Step 6.4.2: 最终 Commit**

```bash
git add results/ scripts/ docs/ src/models/model_info.h
git add results/final_comprehensive_report.md results/final_comprehensive_report.html
git commit -m "feat: Phase 6 收尾 — 全量数据综合报告 + 面试准备材料 + 可视化图表

项目产出:
- 综合性能报告 (final_comprehensive_report.md/html)
- 5张可视化图表: 算子对比/Roofline散点/优化时间线/GPU加速比/多线程
- 每个模型的性能分析卡片 (Phase 1-5 精华版)
- 面试话术准备 (1min/3min/追问应对)
- 简历项目描述更新
- 所有 Phase 数据归档在 results/ 下

Benchmark 体系核心发现:
- CV 模型: MNN 领先 ORT 1.3-2.4× (Conv1x1优化 + NCHW4c布局)
- BERT: MNN 初始慢15% → 定位到MatMul(768)GEMM分块 → 优化后追平/反超
- GPU 异构: GEMM放GPU加速2-5×; DWConv/LayerNorm必须留CPU
- LLM: Decode阶段访存密集, IPC接近0%, KV Cache量化是最有效的优化"
```

---

---

## 附录 D: TVM 编译器基线（可选扩展）

> ⚠️ **已从主线移除。** 等到 Phase 0-6 全部完成后，如有余力再考虑。

### 为什么不放在主线

```
TVM 工作流: AutoTVM 调优(单模型数小时~数天) → 生成调优日志 → 编译模型 → 推理
对比:
  MNN/ORT/llama.cpp: 下载模型 → 直接推理 (分钟级)

TVM 一条模型的时间 ≈ MNN+ORT+llama.cpp 三个框架全部跑完的时间
且 TVM 调优需要编写调优脚本、调试编译错误、处理算子不支持等问题
风险高、ROI 低，不适合放在主线，但作为扩展项面试加分很大
```

### 如果需要加入（可选路线）

```
Phase D.1: TVM 编译 MNN 的瓶颈算子 (1-2天)
  □ 从 Phase 4 已定位的 BERT MatMul(768×768) 开始
  □ 编写 TVM 调优脚本: tvm.autotvm.tuner 对 MatMul 调优
  □ 导出调优后的 .so 动态库
  □ 集成到 ORT 作为 Custom OP (体现 "TVM+ORT 联合优化")

Phase D.2: TVM 编译整模型 MobilenetV2 (1-2天)
  □ TVM 编译 MobilenetV2 (简化: 直接用已调优的 TOPI 模板)
  □ 对比 TVM vs MNN vs ORT on MobileNetV2
  □ 分析 TVM 优缺点和适用场景

Phase D.3: 产出 TVM 编译器路线分析 (0.5天)
  □ "手写优化 vs 自动代码生成的边界在哪里？"
  □ 哪些算子 TVM 自动生成已经接近甚至超过手写？
  □ 哪些算子 TVM 调优后仍有明显差距？为什么？

面试加分:
  "我不仅做了手写优化的 MNN 路线，还探索了 TVM 自动代码生成路线。
  结论是: 对于 Conv1x1/MatMul 这类规则计算，TVM 自动调优可以达到手写的 85-90%；
  但对于 DWConv 这种访存密集 + 不规则访问的算子，手写优化仍然有明显优势。
  这让我理解了编译器自动生成的边界在哪里。"
```

## 附录 A: 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| MNN OpenCL 在骁龙865上无法加载 | 中 | Phase 3 全阻塞 | 提前验证 OpenCL .so 存在; 备选: 用 OpenGL ES 3.1 compute shader |
| Qwen2-0.5B GGUF 转换失败 | 低 | Phase 5 部分阻塞 | 用 llama.cpp 官方 convert.py; 备选: 直接用 ONNX 格式 + ORT GenAI |
| BERT GEMM 优化无法追平 ORT | 中 | 核心故事受损 | 不能追平也能讲清楚"为什么追不平", 同样是好的面试素材 |
| MobileViT-S ONNX 模型无现成预训练 | 中 | Phase 1 缺一个模型 | 用 timm/torchvision 导出; 备选: 换成 EfficientNet-Lite |
| 业余投入时间不足 | 高 | 全局延期 | Phase 4 (BERT优化) 为最高优先级; Phase 5 (LLM) 可压缩; Phase 3 (GPU) 不能跳过 |

## 附录 B: 每日工作检查清单

每天开始前的检查:
- [ ] `adb devices` — 设备在线
- [ ] `adb shell getprop ro.product.model` — 型号正确
- [ ] `adb shell cat /sys/class/thermal/thermal_zone0/temp` — 温度 < 45000 (45°C)
- [ ] `git status` — 确认在正确分支
- [ ] `ls build_android/src/benchmark_inference` — 二进制存在

每天结束后的归档:
- [ ] 原始日志已保存到 `results/phaseN_*/` 下
- [ ] 关键数据已提取到当天的 summary.md
- [ ] git commit (即使只是 WIP)
- [ ] 设备温度已恢复正常
- [ ] CPU 调度器已恢复 (如果锁了频)

## 附录 C: 关键命令速查

```bash
# === 快速验证 ===
export ADB=/mnt/e/andorid/adb/adb.exe
export ANDROID_NDK=/home/liu/android-ndk
$ADB devices | grep "device$"

# === 编译 ===
./scripts/build_android.sh 2>&1 | tail -20

# === 单模型快速测试 ===
./scripts/run_benchmark_android.sh --model mobilenetv2 --backend mnn --iterations 10

# === MNN Profiling ===
MNN_PROFILING=1 ./scripts/run_benchmark_android.sh --model bert --backend mnn --iterations 100

# === 设备环境 ===
./scripts/setup_test_environment.sh    # 锁频 + 清缓存
./scripts/restore_test_environment.sh  # 恢复

# === simpleperf ===
$ADB shell su -c "/data/local/tmp/simpleperf stat -e cpu-cycles,cache-misses --duration 10 \
  /data/local/tmp/benchmark_inference --model mobilenetv2 --backend mnn --iterations 200"

# === 报告生成 ===
python3 scripts/generate_report.py --input results/phase1_cpu_profiling/ --output report.md
python3 scripts/extract_operator_profile.py results/*.log > hotpot_ranking.md
python3 scripts/analyze_roofline.py --input results/phase2_hotspot_benchmark/ --plot roofline.png
```

---

> **计划版本:** v1.1  
> **创建日期:** 2026-06-06  
> **最后更新:** 2026-06-06 (v1.1: TVM 移至附录 D，主线聚焦 MNN/ORT/llama.cpp 三框架)  
> **基于文档:** `docs/01-guides/android-benchmark-guide.md` + `docs/03-plans/full-plan.md`  
> **目标硬件:** 骁龙 865 (SM8250) / 红米 K30S  
> **预估总工期:** 28 天 (业余投入，移除 TVM 后主线压缩 2 天)
