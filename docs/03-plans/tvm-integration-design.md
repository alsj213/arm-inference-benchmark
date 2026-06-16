# TVM 集成方案 — 自动调优 + 推理 Benchmark

## 一、TVM 的工作流为什么特殊

MNN/ORT/llama.cpp 都是 **「加载模型 → 推理」** 两步走，TVM 多了一个调优阶段：

```
┌──────────────────────────────────────────────────────────────┐
│  MNN / ORT:  加载模型 ──────────────────────▶ 推理 (ms)        │
│                                                              │
│  TVM:        模型 ──▶ 自动调优 ──▶ 编译 ──▶ 推理 (ms)         │
│                      (小时/天)     (分钟)                      │
└──────────────────────────────────────────────────────────────┘
```

**核心设计原则**：调优是一次性投入（缓存调优记录），推理才是 benchmark 对比的指标。

---

## 二、三阶段工作流

### 阶段 A：自动调优（一次，几小时~天，结果可缓存）

```
  ONNX Model ──▶ AutoTVM Tuner ──▶ tuning_records.json
                    │                    │
                    │  每个算子尝试         │  最优 schedule 参数
                    │  数百种 schedule      │  缓存下来永久复用
                    │                    │
                    在设备上跑（最准确）    或：在主机上交叉调优
```

**调优策略**（二选一）：

| 策略 | 方式 | 准确度 | 耗时 | 适用场景 |
|------|------|--------|------|---------|
| **设备端调优** | RPC Tracker 连手机，在骁龙865上实测每个 schedule | ⭐⭐⭐ 最准 | 慢（几小时~天） | 最终性能数据 |
| **主机端调优** | x86 上用模拟参数调优，交叉编译到 ARM | ⭐⭐ 近似 | 快（分钟~小时） | 快速验证、开发迭代 |

建议：**先用主机端调优跑通流程，最后用设备端调优拿最终数据**。

### 阶段 B：编译（每次几分钟，依赖调优记录）

```
  ONNX Model ──┐
               ├──▶ TVM Relay Build ──▶ model.so / model.tar
  tuning.json ─┘       (应用调优记录)
```

### 阶段 C：推理（毫秒级，benchmark 对比的主体）

```
  model.so ──▶ TVM GraphRuntime ──▶ Latency (ms)
```

---

## 三、目录结构设计

```
benchmark/
├── scripts/
│   ├── tvm_tune.sh                    # [新增] 一键调优脚本（封装 Python）
│   ├── tvm_compile.sh                 # [新增] 编译脚本（ONNX → .so）
│   ├── tvm_benchmark.sh               # [新增] 推理 benchmark（推送 .so 到手机 + 运行）
│   └── tvm_full_pipeline.sh           # [新增] 三阶段一键执行
│
├── tools/
│   └── tvm/
│       ├── tune_model.py              # [新增] AutoTVM 调优脚本
│       ├── compile_model.py           # [新增] Relay 编译（应用调优记录）
│       ├── tuning_records/            # [新增] 调优记录缓存目录
│       │   ├── mobilenetv2_tuning.json
│       │   ├── resnet50_tuning.json
│       │   └── bert_tuning.json
│       └── compiled_models/           # [新增] 编译产出
│           ├── mobilenetv2_tvm.so
│           ├── resnet50_tvm.so
│           └── bert_tvm.so
│
├── src/backends/
│   └── tvm_backend.cpp/h              # [重写] 替换现有手写模拟为真正的 TVM Runtime
│
├── models/
│   └── (ONNX 模型，作为 TVM 输入，复用现有)
│
└── results/
    └── phase_tvm/
        ├── tuning_log.txt             # 调优过程日志
        ├── tvm_vs_mnn_vs_ort.md       # 三方对比报告
        └── *.log                      # Benchmark 原始日志
```

---

## 四、核心文件设计

### 4.1 `tools/tvm/tune_model.py` — AutoTVM 调优脚本

```python
#!/usr/bin/env python3
"""
TVM AutoTVM 调优脚本
用法: python3 tune_model.py --model mobilenetv2 --target "llvm -device=arm_cpu -mtriple=aarch64-linux-android"
"""

import os, sys, argparse, json, logging
import numpy as np
import tvm
from tvm import relay, autotvm

# ===== 调优配置 =====
TUNING_OPTIONS = {
    "mobilenetv2": {
        "onnx_path": "models/classification/mobilenetv2/mobilenetv2.onnx",
        "input_shape": [1, 3, 224, 224],
        "n_trial": 1000,            # 每个算子尝试 1000 个 schedule
        "early_stopping": 100,       # 100 个 trial 无提升就停
        "timeout": 10,               # 每个 trial 最多 10 秒
    },
    "resnet50": {
        "onnx_path": "models/classification/resnet50/resnet50.onnx",
        "input_shape": [1, 3, 224, 224],
        "n_trial": 1000,
        "early_stopping": 100,
        "timeout": 10,
    },
    "bert": {
        "onnx_path": "models/nlp/bert/bert.onnx",
        "input_shape": [1, 128],      # seq_len=128
        "n_trial": 1500,              # Transformer 算子更多
        "early_stopping": 150,
        "timeout": 10,
    },
}

# ===== 需要调优的 TOP 算子 =====
# 不是所有算子都调优，只调优计算密集的 kernel
TUNING_TASKS = [
    # Conv2d (最核心)
    "conv2d_nchw_1x1",   # MobileNetV2 核心
    "conv2d_nchw_3x3",   # ResNet50 核心
    "conv2d_nchw_depthwise",
    # MatMul (BERT 核心)
    "dense_nopack",
    "dense_pack",
    # 其他（可选）
    "conv2d_nchw_winograd",
]


def get_tuning_tasks(mod, params, target):
    """提取需要调优的 task"""
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params
    )
    # 只保留我们关心的算子类型
    filtered = []
    for t in tasks:
        for pattern in TUNING_TASKS:
            if pattern in str(t):
                filtered.append(t)
                break
    return filtered


def tune_model(model_name, target_str, tuning_records_dir, use_rpc=False, rpc_host=None):
    """执行 AutoTVM 调优"""
    config = TUNING_OPTIONS[model_name]

    # 1. 加载 ONNX 模型
    print(f"[1/4] Loading ONNX model: {config['onnx_path']}")
    import onnx
    onnx_model = onnx.load(config["onnx_path"])
    shape_dict = {"input": config["input_shape"]}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # 2. 提取调优任务
    target = tvm.target.Target(target_str)
    tasks = get_tuning_tasks(mod, params, target)
    print(f"[2/4] {len(tasks)} tuning tasks extracted: "
          f"{[str(t) for t in tasks]}")

    # 3. 执行调优
    records_path = os.path.join(tuning_records_dir, f"{model_name}_tuning.json")
    print(f"[3/4] Tuning {len(tasks)} tasks...")

    for i, task in enumerate(tasks):
        print(f"  Task {i+1}/{len(tasks)}: {task}")

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"),
            runner=autotvm.LocalRunner(
                number=10,      # 每次测试跑 10 次取平均
                repeat=3,       # 重复 3 次
                timeout=config["timeout"],
                min_repeat_ms=100,
            )
        )

        tuner = autotvm.tuner.XGBTuner(task, loss_type="reg")
        tuner.tune(
            n_trial=min(config["n_trial"], len(task.config_space)),
            early_stopping=config["early_stopping"],
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(config["n_trial"]),
                autotvm.callback.log_to_file(records_path),
            ],
        )

    # 4. 保存调优记录
    print(f"[4/4] Tuning records saved to: {records_path}")
    return records_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", default="llvm -device=arm_cpu -mtriple=aarch64-linux-android")
    parser.add_argument("--records-dir", default="tools/tvm/tuning_records")
    parser.add_argument("--rpc", action="store_true",
                        help="Use RPC tracker for device-side tuning")
    parser.add_argument("--rpc-host", default="localhost")
    args = parser.parse_args()

    tune_model(args.model, args.target, args.records_dir,
               use_rpc=args.rpc, rpc_host=args.rpc_host)
```

### 4.2 `tools/tvm/compile_model.py` — 编译脚本

```python
#!/usr/bin/env python3
"""
TVM 编译脚本（应用调优记录）
用法: python3 compile_model.py --model mobilenetv2 --records tools/tvm/tuning_records/mobilenetv2_tuning.json
"""

import os, sys, argparse, json
import numpy as np
import tvm
from tvm import relay, autotvm


def compile_model(model_name, onnx_path, input_shape, tuning_records_path, output_dir):
    """编译 ONNX 模型，应用调优记录"""

    target = "llvm -device=arm_cpu -mtriple=aarch64-linux-android"
    target_host = "llvm -mtriple=aarch64-linux-android"

    # 1. 加载 ONNX
    print(f"[1/3] Loading ONNX: {onnx_path}")
    import onnx
    onnx_model = onnx.load(onnx_path)
    shape_dict = {"input": input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # 2. 应用调优记录
    if tuning_records_path and os.path.exists(tuning_records_path):
        print(f"[2/3] Applying tuning records: {tuning_records_path}")
        with autotvm.apply_history_best(tuning_records_path):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, target_host=target_host, params=params)
    else:
        print(f"[2/3] No tuning records found, using default schedules")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)

    # 3. 导出
    os.makedirs(output_dir, exist_ok=True)
    so_path = os.path.join(output_dir, f"{model_name}_tvm.so")
    lib.export_library(so_path)

    # 同时导出 graph + params（备选加载方式）
    graph_path = os.path.join(output_dir, f"{model_name}_tvm.json")
    params_path = os.path.join(output_dir, f"{model_name}_tvm.params")
    with open(graph_path, "w") as f:
        f.write(lib.get_graph_json())
    with open(params_path, "wb") as f:
        f.write(tvm.runtime.save_param_dict(lib.get_params()))

    # 元数据
    meta = {
        "model": model_name,
        "input_name": "input",
        "input_shape": list(input_shape),
        "tuning_records": tuning_records_path,
        "compiled_so": so_path,
    }
    meta_path = os.path.join(output_dir, f"{model_name}_tvm_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[3/3] Compiled artifacts:")
    print(f"  {so_path}")
    print(f"  {graph_path}")
    print(f"  {params_path}")
    print(f"  {meta_path}")
    return so_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--input-shape", type=int, nargs="+", default=[1,3,224,224])
    parser.add_argument("--records", default=None, help="Path to tuning_records.json")
    parser.add_argument("--output-dir", default="tools/tvm/compiled_models")
    args = parser.parse_args()

    compile_model(args.model, args.onnx, args.input_shape, args.records, args.output_dir)
```

### 4.3 `src/backends/tvm_backend.cpp` — 重写为真正的 TVM Runtime

```cpp
// 当前: 手写 C++ 模拟算子（208行假的）
// 重写: 加载 TVM 编译的 .so，用 TVM GraphRuntime 做真实推理

#include "tvm_backend.h"
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dlfcn.h>

class TVMBackend : public BenchmarkBackend {
private:
    tvm::runtime::Module mod_;
    tvm::runtime::PackedFunc set_input_;
    tvm::runtime::PackedFunc run_;
    tvm::runtime::PackedFunc get_output_;
    tvm::runtime::NDArray input_tensor_;

public:
    bool Init(const BenchmarkConfig& config) override {
        // 1. 加载编译好的 .so
        std::string so_path = config.model_path;  // eg: tools/tvm/compiled_models/mobilenetv2_tvm.so
        mod_ = tvm::runtime::Module::LoadFromFile(so_path);

        // 2. 获取 GraphRuntime 函数
        set_input_ = mod_.GetFunction("set_input");
        run_       = mod_.GetFunction("run");
        get_output_ = mod_.GetFunction("get_output");

        // 3. 分配输入 Tensor
        // input_shape 从 config 或 meta.json 读取
        DLDevice ctx = {kDLCPU, 0};
        input_tensor_ = tvm::runtime::NDArray::Empty(
            config.input_shape, {kDLFloat, 32, 1}, ctx);

        return set_input_ != nullptr && run_ != nullptr;
    }

    bool Run(std::vector<float>& output) override {
        // 1. 拷贝输入数据
        input_tensor_.CopyFromBytes(input_data_.data(),
                                     input_data_.size() * sizeof(float));
        set_input_("input", input_tensor_);

        // 2. 执行推理
        run_();

        // 3. 获取输出
        tvm::runtime::NDArray out = get_output_(0);
        output.resize(out->shape[0] * out->shape[1] * out->shape[2] * out->shape[3]);
        out.CopyToBytes(output.data(), output.size() * sizeof(float));

        return true;
    }

    // ... NDArray 引用计数管理
};
```

### 4.4 `scripts/tvm_full_pipeline.sh` — 一键三阶段

```bash
#!/bin/bash
# TVM 完整 Pipeline: 调优 → 编译 → Benchmark
# 用法: ./scripts/tvm_full_pipeline.sh --model mobilenetv2 [--tune] [--skip-tune]

set -euo pipefail
MODEL="${1:-mobilenetv2}"
DO_TUNE="${2:-true}"  # 默认调优；传 --skip-tune 跳过

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RECORDS_DIR="$PROJECT_DIR/tools/tvm/tuning_records"
OUTPUT_DIR="$PROJECT_DIR/tools/tvm/compiled_models"
RESULTS_DIR="$PROJECT_DIR/results/phase_tvm"
mkdir -p "$RECORDS_DIR" "$OUTPUT_DIR" "$RESULTS_DIR"

# ===== 阶段 A: 自动调优 =====
if [ "$DO_TUNE" != "--skip-tune" ]; then
  echo "=== Phase A: AutoTVM Tuning ==="
  RECORDS_FILE="$RECORDS_DIR/${MODEL}_tuning.json"
  
  if [ -f "$RECORDS_FILE" ]; then
    echo "Tuning records exist: $RECORDS_FILE"
    echo "Use --force-retune to re-tune"
  else
    python3 "$PROJECT_DIR/tools/tvm/tune_model.py" \
      --model "$MODEL" \
      --records-dir "$RECORDS_DIR" \
      2>&1 | tee "$RESULTS_DIR/${MODEL}_tune.log"
  fi
else
  echo "=== Phase A: Skipped (--skip-tune) ==="
fi

# ===== 阶段 B: 编译 =====
echo "=== Phase B: Compilation ==="
RECORDS_FILE="$RECORDS_DIR/${MODEL}_tuning.json"

# 确定 ONNX 路径
case "$MODEL" in
  mobilenetv2) ONNX="$PROJECT_DIR/models/classification/mobilenetv2/mobilenetv2.onnx"; SHAPE="1 3 224 224" ;;
  resnet50)    ONNX="$PROJECT_DIR/models/classification/resnet50/resnet50.onnx";         SHAPE="1 3 224 224" ;;
  bert)        ONNX="$PROJECT_DIR/models/nlp/bert/bert.onnx";                            SHAPE="1 128" ;;
  *) echo "Unknown model: $MODEL"; exit 1 ;;
esac

RECORDS_ARG=""
[ -f "$RECORDS_FILE" ] && RECORDS_ARG="--records $RECORDS_FILE"

python3 "$PROJECT_DIR/tools/tvm/compile_model.py" \
  --model "$MODEL" \
  --onnx "$ONNX" \
  --input-shape $SHAPE \
  $RECORDS_ARG \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$RESULTS_DIR/${MODEL}_compile.log"

# ===== 阶段 C: Benchmark =====
echo "=== Phase C: Inference Benchmark ==="
COMPILED_SO="$OUTPUT_DIR/${MODEL}_tvm.so"

if [ ! -f "$COMPILED_SO" ]; then
  echo "ERROR: Compiled .so not found: $COMPILED_SO"
  exit 1
fi

# 编译 benchmark 二进制（确保 TVM 后端已启用）
cd "$PROJECT_DIR"
# CMAKE 参数: -DBENCHMARK_TVM=ON
./scripts/build_android.sh 2>&1 | tail -20

# 推送到设备并运行
ADB="${ADB:-adb}"
$ADB push "$COMPILED_SO" /data/local/tmp/tvm_models/
$ADB push build_android/src/benchmark_inference /data/local/tmp/

$ADB shell "/data/local/tmp/benchmark_inference \
  --model $MODEL --backend tvm \
  --model-path /data/local/tmp/tvm_models/${MODEL}_tvm.so \
  --iterations 100 --warmup 50" \
  2>&1 | tee "$RESULTS_DIR/${MODEL}_tvm_benchmark.log"

echo "=== Pipeline Complete ==="
echo "Results: $RESULTS_DIR/"
```

---

## 五、与现有 Benchmark 框架的集成点

### 5.1 配置

在 `.benchmarkrc.yml` 中增加 TVM：

```yaml
backends:
  - mnn
  - onnxruntime
  - tvm                    # 新增

tvm:
  tuning_records_dir: tools/tvm/tuning_records
  compiled_models_dir: tools/tvm/compiled_models
  auto_tune: true          # 缺少调优记录时自动触发调优
  tune_timeout_hours: 4    # 调优超时（单个模型）
```

### 5.2 模型路径映射

在 `src/models/model_info.h` 中增加 TVM 路径：

```cpp
const ModelInfo MOBILENETV2 = {
    .name = "mobilenetv2",
    .type = ModelType::CLASSIFICATION,
    .mnn_path  = "models/classification/mobilenetv2/mobilenetv2.mnn",
    .onnx_path = "models/classification/mobilenetv2/mobilenetv2.onnx",
    .tvm_path  = "tools/tvm/compiled_models/mobilenetv2_tvm.so",   // 新增
    .tvm_tuning_path = "tools/tvm/tuning_records/mobilenetv2_tuning.json",  // 新增
};
```

### 5.3 BackendType 枚举

```cpp
enum class BackendType {
    MNN_CPU  = 0,
    MNN_GPU  = 1,
    ORT_CPU  = 2,
    TVM      = 3,   // 已存在但需确认
    NCNN     = 4,
    LLAMACPP = 5,
};
```

### 5.4 推理流程对比

```
现有流程:
  benchmark_inference --model mobilenetv2 --backend mnn
  └─▶ MNNBackend::Init() 加载 .mnn → 推理

TVM 流程:
  benchmark_inference --model mobilenetv2 --backend tvm
  └─▶ TVMBackend::Init()
        ├─ 检查 .so 是否存在
        ├─ 不存在 → 检查 tuning_records 是否存在
        │   ├─ 不存在 → 自动触发调优 (可选, 耗时)
        │   └─ 存在 → 自动触发编译 (几分钟)
        └─ 加载 .so → 推理
```

---

## 六、调优策略对比

### 策略 A: 主机端调优 + 交叉编译（推荐先跑通）

```
优点:
  - 不需要 RPC 连接手机
  - 调优在 x86 上跑，速度快（分钟~小时）
  - 交叉编译到 aarch64，.so 可直接在手机上运行

缺点:
  - x86 调优的 schedule 在 ARM 上不一定最优
  - 需要 LLVM 交叉编译工具链

工作流:
  python3 tools/tvm/tune_model.py --model mobilenetv2 --target "llvm"
  python3 tools/tvm/compile_model.py --model mobilenetv2 --target aarch64-linux-android
```

### 策略 B: 设备端 RPC 调优（最终性能数据）

```
优点:
  - 每个 schedule 在骁龙865上实测，结果最准确
  - 面试含金量更高（"我在手机上做的 AutoTVM 调优"）

缺点:
  - 需要手机一直连着、不能息屏、充电
  - 调优时间可能是主机端的 5-10 倍

工作流:
  # 手机上启动 RPC Tracker
  adb shell "/data/local/tmp/tvm_rpc server --host=0.0.0.0 --port=9090"
  adb forward tcp:9090 tcp:9090

  # 主机上连接 RPC 调优
  python3 tools/tvm/tune_model.py --model mobilenetv2 --rpc --rpc-host localhost --rpc-port 9090
```

---

## 七、面试叙事线

```
"我在 MNN 和 ORT 对比的基础上，加了 TVM 作为编译器基线。

TVM 和 MNN 的优化哲学完全不同：
  - MNN 是工程师手写 Neon 汇编，每个 kernel 精心调优
  - TVM 是 AutoTVM 自动搜索 schedule 空间，机器学习驱动的调优

我做了两件事：
  1. 用 AutoTVM 对 Conv1x1/Conv3×3/MatMul 做设备端调优，
     对比自动生成的代码 vs MNN 手写代码的性能差距
  2. 把 TVM 调优后的 GPU kernel 集成到 MNN 作为 Custom Op，
     验证"编译器自动生成 + 手写框架"的混合路线

核心发现：
  - Conv1x1: TVM 自动生成能达到 MNN 手写的 85-90%（说明规则计算自动化可行）
  - DWConv: TVM 只有 MNN 的 60%（说明不规则访存仍需手写优化）
  - MatMul: TVM AutoTVM 生成的 schedule 和 MNN 手写的不谋而合

这让我理解了编译器自动生成的边界在哪里——
不是所有算子都能靠自动调优搞定，人类的经验在某些场景下仍然不可替代。"
```

---

## 八、实施计划

| 阶段 | 内容 | 预估时间 |
|------|------|---------|
| D.1 | 环境搭建：编译 TVM Python 包 + LLVM 交叉编译链 | 0.5 天 |
| D.2 | 编写 `tune_model.py` + `compile_model.py` | 1 天 |
| D.3 | 重写 `tvm_backend.cpp`（加载 .so + TVM Runtime） | 1 天 |
| D.4 | MobileNetV2 主机端调优 + 编译 + 推理对比 | 1 天 |
| D.5 | BERT MatMul 设备端 RPC 调优 + 对比 MNN | 1-2 天 |
| D.6 | 产出 TVM vs MNN vs ORT 三框架对比报告 | 0.5 天 |

**总计**: 5-6 天（前提：TVM 环境编译顺利）

---

## 九、关键风险

| 风险 | 缓解 |
|------|------|
| TVM Python 包编译失败 | 使用 `pip install apache-tvm` 预编译包；或 Docker 环境 |
| LLVM 交叉编译链缺失 | NDK 自带 clang 可作为 TVM 后端 |
| AutoTVM 在 ARM 上超时 | 减少 `n_trial`，先跑通流程再追求极致 |
| TVM Runtime .so 太大（>100MB） | 用 `export_library` 的 `cc` 选项减少体积 |
| 部分算子 TVM 不支持 | 标记，跳过该算子，用默认 schedule |

---

> **版本**: v1.0 · 2026-06-06 · 可作为 Phase 7 或独立扩展项执行
