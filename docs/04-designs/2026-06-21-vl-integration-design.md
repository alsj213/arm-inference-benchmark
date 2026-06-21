# VL 多模态推理集成设计方案

## 1. 概述

将 Qwen3-VL-4B / Qwen2-VL-2B 多模态视觉语言模型推理集成到统一的 `llm_benchmark` 工具中，支持 MNN 和 llama.cpp 双后端，覆盖 text-only 和 vision-language 两种推理模式，并建立精度验证流程。

## 2. 架构

```
llm_benchmark (统一入口)
    ├── --backend llamacpp
    │       ├── text-only: LlamaCppBackend::generate() / benchmark_decode()
    │       └── VL:        LlamaCppBackend::generate_vl()  (子进程调用 llama-mtmd-cli)
    │
    ├── --backend mnn_llm
    │       ├── text-only: MnnLlmBackend::generate() / benchmark()
    │       └── VL:        MnnLlmBackend::generate_vl()  (直接调用 MNN MultimodalPrompt)
    │
    └── --accuracy        # 精度对比模式
            ├── 贪婪解码一致性 (temperature=0, seed=42)
            └── Logits 余弦相似度 (>0.95 通过)
```

## 3. 新参数

```
--image <path>            图片路径（启用 VL 模式）
--image-size <w> <h>      RAW 图片宽高（仅 MNN RAW 模式需要）
--accuracy <mnn|llamacpp> 精度对比模式：以指定后端为基准
--seed <n>               随机种子（默认 42）
```

## 4. VL 推理路径

### 4.1 MNN VL 路径

```
--image test.png → decode to RGB → MultimodalPrompt
    → llm_->response(mm_prompt, &oss, nullptr, max_tokens)
    → 输出 timing (ctx->vision_us / prefill_us / decode_us)
```

MNN 的 `Llm::response(MultimodalPrompt)` API 已由 `mnn_vl_test.cpp` 验证可行：
- 支持 RAW 和 PNG 输入
- 自动做 resize/normalize
- 输出 vision/prefill/decode 三段式 timing

### 4.2 llama.cpp VL 路径

```
--image test.png → llama-mtmd-cli 子进程
    → 解析 stdout 获取输出文本
    → 解析 perf 输出获取 timing
```

两种子方案对比：
| 方案 | 优点 | 缺点 |
|------|------|------|
| **A) 子进程调用** | 无需改 llama.cpp 代码，复用已有二进制 | 进程间通信开销，timing 解析脆 |
| **B) 直接链接** | 无额外开销，timing 精确 | 需要链接 libllama + CLIP，编译复杂 |

**推荐方案 A**（子进程调用），因为：
- 设备上已有 `llama-mtmd-cli` 二进制
- 子进程模式足够 benchmark 粒度（整体 timing）
- 避免复杂的编译依赖

### 4.3 输出指标

| 指标 | 说明 |
|------|------|
| vision_time | 视觉编码耗时（图片→visual tokens） |
| prefill_time | Prefill 耗时（prompt + visual tokens） |
| prefill_speed | Prefill 速度 (tok/s) |
| decode_time | 自回归解码耗时 |
| decode_speed | 解码速度 (tok/s) |
| total_time | 总耗时 |
| output_text | 生成文本 |

## 5. 精度验证

### 5.1 贪婪解码一致性

```python
# 两端固定相同 seed + temperature=0
backend_a = generate_vl(image, prompt, temperature=0.0, seed=42)
backend_b = generate_vl(image, prompt, temperature=0.0, seed=42)

# 比较 token ID 序列
if token_ids_a == token_ids_b:
    pass  # ✅ 一致
else:
    # 计算 ROUGE-L 或人工审查
```

### 5.2 Logits 余弦相似度

从后端提取每一步的 logits 向量（vocab 维），在关键 token 位置计算余弦相似度：

```python
# 对每个解码步
for step in range(n_tokens):
    logits_a = get_logits(backend_a, step)  # [vocab_size]
    logits_b = get_logits(backend_b, step)
    cos_sim = cosine_similarity(logits_a, logits_b)
    if cos_sim < threshold:
        warn(f"Step {step}: cos_sim={cos_sim}")
```

**注意**：MNN Q4 与 llama.cpp Q4_K_M 的量化方式不同，即使同源模型也会有微小差异。阈值设为 **0.95**。

## 6. 文件变更

| 文件 | 变更 |
|------|------|
| `src/llm_benchmark.cpp` | 新增 VL 参数解析 + 分支逻辑 |
| `src/backends/mnn_llm_backend.h` | 新增 `generate_vl()` / `benchmark_vl()` |
| `src/backends/mnn_llm_backend.cpp` | 实现 MultimodalPrompt 路径 |
| `src/backends/llamacpp_backend.h` | 新增 `generate_vl()` / `benchmark_vl()` |
| `src/backends/llamacpp_backend.cpp` | 实现子进程调用 + 精度对比辅助 |
| `src/CMakeLists.txt` | 新增 `llm/` 子目录 |
| `tools/mnn_vl_test.cpp` | 保留为独立测试工具 |

## 7. 测试计划

1. **text-only 回归**：`llm_benchmark --benchmark` 输出与之前一致
2. **MNN VL**：同图跑 3 次，timing 波动 < 10%
3. **llama.cpp VL**：同图跑 3 次，timing 波动 < 10%
4. **精度验证**：`--accuracy mnn` 温度 0 下与 llama.cpp 对比
5. **长期**：将 VL benchmark 纳入 `run_benchmark_android.sh`
