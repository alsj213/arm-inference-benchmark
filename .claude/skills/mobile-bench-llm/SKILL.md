---
name: mobile-bench-llm
description: Use when benchmarking LLM inference on mobile devices — decode tokens/s, prefill latency, TTFT, KV cache memory, long-context trends, and parameter matrix sweeps, using the standard JSON_RESULT log contract
---

# Mobile Bench LLM

## Overview

为 LLM 推理建立专项指标体系,区分 prefill 与 decode 阶段、量化生成速度与首 token 延迟,并报告 KV cache 内存占用。适用于 llama.cpp / MNN LLM 等 LLM 后端。

配置通过项目根目录的 `.benchmarkrc.yml` 读取。指标沿用 benchmark 二进制的 `JSON_RESULT:` 行契约输出,插件侧解析保持一致(design D5)。

## LLM 特有指标

| 指标 | 含义 | 与 CV/NLP 指标的区别 |
|------|------|---------------------|
| Decode tokens/s | 生成阶段速度(tokens/s) | 替代 FPS |
| Prefill latency | 提示词处理阶段耗时(ms) | 无对应 |
| TTFT | 首 token 延迟(Time To First Token, ms) | 无对应 |
| KV cache | 推理时的 KV cache 内存占用 | 无对应 |
| 峰值内存 | 推理过程峰值内存 | 保留 |

## 完整流程

```text
准备(模型转换GGUF/量化) → 参数配置(max-tokens/prompt) → 执行(tee + JSON_RESULT) → 解析 → 报告
```

模型转换(GGUF、量化)见 mobile-bench-model-prep skill。

## 执行测试

### 基础执行

项目用 `llm_benchmark` 二进制直接测（与 `scripts/benchmark/bench-qwen3-06b-3way.sh` 同一模式）。先推二进制与模型到设备，再执行：

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")
DEV=/data/local/tmp/benchmark
# 推送（首次或二进制更新后）
$ADB push build_android/src/llm/llm_benchmark $DEV/
$ADB push <模型目录> $DEV/qwen3_models/
# 执行（必须 tee 保留日志）
$ADB shell "cd $DEV && LD_LIBRARY_PATH=. ./llm_benchmark --backend llamacpp \
  --model qwen3_models/Qwen3-0.6B-Q4_K_M.gguf --benchmark \
  --n-prompt 128 --max-tokens 128 --n-repeat 5 --json" 2>&1 | tee results/benchmark_$(date +%Y%m%d_%H%M%S).log
```

### 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--backend` | LLM 后端 | llamacpp / mnn_llm / mobilellm |
| `--model` | 模型路径 | GGUF 路径 或 MNN config.json |
| `--benchmark` | benchmark 模式（随机 token，仅性能） | - |
| `--n-prompt` | prompt token 数 | 128 |
| `--max-tokens` | 生成 token 数 | 32, 128, 512 |
| `--n-repeat` | 重复次数 | 5 |
| `--require-precision` | 精度对齐：强制量化级别（f32/f16/q8/q4/q3/q2），不匹配则失败 | q4 |
| `--json` | JSON_RESULT 输出 | - |

### 长上下文趋势

对同一模型以递增上下文长度多次测量,观察内存与速度随上下文的变化:

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")
for ctx in 512 1024 2048 4096; do
  $ADB shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=. ./llm_benchmark \
    --backend llamacpp --model qwen3_models/Qwen3-0.6B-Q4_K_M.gguf --benchmark \
    --n-prompt $ctx --max-tokens $ctx --n-repeat 5 --json" 2>&1 | tee results/ctx_$ctx.log
done
```

### 参数矩阵

在不同 max-tokens / 量化组合下测量,汇总为参数-性能矩阵（跨框架对比必须加 `--require-precision` 对齐量化级别）:

```bash
ADB=$(python3 -c "import yaml; print(yaml.safe_load(open('.benchmarkrc.yml'))['device']['adb'])")
for quant in q8_0 q4_k_m; do
  for mt in 32 128; do
    $ADB shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=. ./llm_benchmark \
      --backend llamacpp --model qwen3_models/...-$quant.gguf --benchmark \
      --n-prompt 128 --max-tokens $mt --n-repeat 5 --require-precision q4 --json" \
      2>&1 | tee results/quant_${quant}_mt${mt}.log
  done
done
```

## 结果解析

LLM 指标通过 `JSON_RESULT:` 行输出（实际字段：`prefill_tok_per_s`、`decode_tok_per_s`、`ttft_ms`、`tpot_ms`、`peak_memory_mib`、`precision`、`quant_label`），`parse_log.py` 透传:

```bash
python3 scripts/analyze/mobilebench/parse_log.py results/benchmark_*.log
```

> **契约约定**:LLM 指标字段的产出在 benchmark 二进制侧;插件侧沿用 `parse_log.py` 的统一解析,不新增插件专属解析格式。

## 精度对齐（跨框架对比 MUST 遵守）

**规则**：同一次跨框架对比（MNN vs llama.cpp vs MobileLLM），所有框架必须使用**同一量化级别**。
禁止出现「一个 int8 一个 Q4_K_M」这类不公平对比。llm_benchmark 已内置检测，用
`--require-precision <level>` 强制，不匹配直接失败。

### 量化级别映射表

| 级别 | llama.cpp (GGUF ftype) | MNN (quant_bit) | MobileLLM (GGUF) |
|------|------------------------|------------------|------------------|
| f32 | F32 | (未量化) | f32 |
| f16 | F16 / BF16 | 0（导出默认 fp16） | f16 |
| **q8** | Q8_0 | **8** | q8_0 |
| **q4** | Q4_0 / Q4_1 / Q4_K_S / Q4_K_M | **4** | q4_* |
| q3 | Q3_* | 3 | q3_* |
| q2 | Q2_K | 2 | q2_* |
| q5/q6/iq | 对应 ftype | - | 对应 |

> MNN 检测顺序：flatbuffer 解析 `llm.mnn` 的 `Convolution.quanParameter.aMaxOrBits` 众数
> （线性层/权重主体量化）→ `export_args.json` 的 `quant_bit` → config.json 的 `precision` 字段。
> ⚠️ 勿用 `tie_embeddings[3]` 作整体量化依据——它只反映嵌入层（如官方 0.6B 嵌入层 8bit、线性层 4bit）。
> llama.cpp/MobileLLM 读 GGUF `general.file_type`。

### 使用方法

```bash
# 对比前，所有后端加 --require-precision 同一级别（如三方都要求 q4）
$ADB shell "cd /data/local/tmp/benchmark && LD_LIBRARY_PATH=. ./llm_benchmark \
  --backend mnn_llm --model qwen3_models/qwen3-0.6b-mnn/config.json \
  --benchmark --n-prompt 128 --max-tokens 128 --n-repeat 5 --require-precision q4 --json"
# 若 MNN 模型实际是 int8（quant_bit=8），将打印 ERROR 并拒绝运行 —— 这就是对齐校验
```

JSON_RESULT 自带 `precision`（规范级别）与 `quant_label`（如 `Q4_K_M` / `int8`），报告 MUST 标注。

### 已知陷阱

- MNN 官方 release 模型（如 Qwen3-0.6B-MNN-official）线性层主体为 **Q4（int4）**，但嵌入层
  可能单独 8bit（tie_embeddings）——检测以线性层 aMaxOrBits 为准，勿被嵌入层误导
- 若某 MNN 模型经 `--require-precision` 报 `quantized`（无法确定 bit），多半是缺
  `llm.mnn.json` 且 flatbuffer 解析失败，MUST 手动确认（MNNConvert 转 json 查 aMaxOrBits）
- GGUF 里 Q4_K_M 与 Q4_0 都是 q4 级别，可与 MNN int4 对齐；但 Q5/Q6 与 q4 不等价

## 报告模板

```markdown
## LLM Benchmark 结果

| 模型 | 精度 | max-tokens | Decode(tokens/s) | Prefill(ms) | TTFT(ms) | KV cache(MB) | 峰值内存(MB) | 来源 |
|------|------|-----------|------------------|-------------|----------|--------------|--------------|------|
| qwen2.5_1.5b | q8_0 | 128 | 12.5 | 320 | 350 | 512 | 1800 | grep "JSON_RESULT" LOG |

### 长上下文趋势

| 上下文长度 | tokens/s | 峰值内存(MB) |
|-----------|----------|--------------|
| 512 | 13.0 | 1500 |
| 2048 | 11.5 | 1700 |
| 4096 | 10.2 | 2100 |

> 若内存随上下文非线性增长,标注内存增长风险。
```

## 数据真实性规则

- Decode/tokens/s、TTFT、KV cache MUST 来自 tee 日志或 `JSON_RESULT:` 行,标注来源
- 报告 MUST 标注 max-tokens、batch、prompt 长度等参数,便于复现
- 长上下文趋势 MUST 基于真实多组测量,不得外推
- LLM 推理总耗时长(如 VL ~17s),确保 `--duration`/超时设置足够,见 mobile-bench-profiling 的 LLM 火焰图说明

## 常见问题

- **JSON_RESULT 里没有 LLM 字段**: 确认二进制支持 LLM 指标输出;若为旧版本,可能只有文本格式
- **TTFT 为 0 或缺失**: 后端未实现 TTFT 测量,标注 N/A
- **KV cache 无法读取**: 从峰值内存差值估算并标注为估算
- **生成为空/超时**: 检查 max-tokens、prompt 长度与预热是否充足

## 关联 Skill

- [mobile-bench-run](../mobile-bench-run/SKILL.md) — 完整基准测试流程(tee、环境控制、报告)
- [mobile-bench-profiling](../mobile-bench-profiling/SKILL.md) — LLM/VL 火焰图(进程退出型模型需命令行模式采样)
- [mobile-bench-model-prep](../mobile-bench-model-prep/SKILL.md) — GGUF 转换与量化
- [mobile-bench-memory](../mobile-bench-memory/SKILL.md) — KV cache 与内存深入分析
