# Benchmark Methodology

> ARM Inference Benchmark follows the measurement principles of [MLPerf Inference: Mobile](https://arxiv.org/abs/2012.02328) while adapting to the practical constraints of cross-framework mobile benchmarking.

## 1. Measurement Protocol

### 1.1 Single-Stream Scenario

All benchmarks use the **single-stream** scenario: the next inference query is issued only after the previous one completes. This matches real-world mobile app behavior where frames are processed one at a time.

```
for i in 1..N:
    t0 = now()
    backend.infer(input)
    t1 = now()
    record(t1 - t0)
```

### 1.2 Fixed Random Seed

Input tensors are filled with `fill_random_float(seed=42)` to ensure **identical input across all frameworks**. This eliminates input-dependent variance as a confounding factor.

### 1.3 Warmup

**10 warmup iterations** are run before measurement. This excludes:
- Cold-start overhead (JIT compilation, memory allocation)
- First-run cache misses

### 1.4 Measurement Window

**50 measurement iterations** (configurable). The harness records wall-clock time for each inference call using `std::chrono::high_resolution_clock`.

### 1.5 Metrics

| Metric | Definition | Report |
|--------|-----------|--------|
| **P50** | 50th percentile latency | Primary comparison metric |
| **P90** | 90th percentile latency | Tail latency |
| **P99** | 99th percentile latency | Worst-case latency |
| **FPS** | 1000 / mean_latency | Throughput |
| **Peak Memory** | RSS delta before/after benchmark | Memory efficiency |

---

## 2. Accuracy Validation

### 2.1 Reference Baseline

**ONNX Runtime FP32** serves as the accuracy baseline. All other frameworks' outputs are compared against ORT's output using the same input.

### 2.2 Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Cosine Similarity** | > 0.99 | Directional similarity of output vectors |
| **Mean Absolute Error** | — | Average per-element difference |
| **Max Absolute Error** | — | Maximum per-element difference |

### 2.3 Pass Criteria

- FP32: cosine similarity > 0.99
- FP16: cosine similarity > 0.98
- INT8: cosine similarity > 0.95

---

## 3. Verification Protocol

### 3.1 Native Tool Cross-Check

Every framework's harness result is verified against the framework's own benchmark tool:

| Framework | Native Tool | Command |
|-----------|------------|---------|
| MNN | `benchmark.out` | `./benchmark.out models_folder 10 0 0 threads` |
| ORT | `onnxruntime_perf_test` | `./onnxruntime_perf_test model.onnx 10` |

### 3.2 Deviation Thresholds

| Deviation | Verdict | Action |
|-----------|---------|--------|
| < 5% | ✅ Trusted | No action |
| 5–10% | ⚠️ Check | Review wrapper implementation |
| > 10% | ❌ Unreliable | Fix harness or document methodology difference |

### 3.3 Known Deviations

**MNN benchmark.out vs Harness (+59%):** The native `benchmark.out` measures raw forward time (`avg` in output). The harness measures full inference cycle including `memcpy → runSession → memcpy` for input/output tensors. This difference is documented and expected.

---

## 4. Environment Control

### 4.1 CPU Frequency Lock

Before each benchmark session, all CPU cores are locked to their maximum frequency:

```bash
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 4.2 Cache Clearing

```bash
echo 3 > /proc/sys/vm/drop_caches
```

### 4.3 Temperature Monitoring

Device temperature is recorded before each run. Results above 45°C are flagged with a throttling warning.

### 4.4 Thermal Zone

```
cat /sys/class/thermal/thermal_zone0/temp  # in millidegrees Celsius
```

---

## 5. Track Definitions

### 5.1 CNN Track

**Models:** MobileNetV2, ResNet50, YOLOv8n, BERT, ShuffleNetV2, MobileViT-S

**Metrics:** P50, P90, P99 latency · FPS · Peak Memory · Init Time · Accuracy

**Input:** Random float32 tensor matching model input shape

### 5.2 LLM Track

**Models:** Qwen2-0.5B, Qwen3-4B

**Metrics:** TTFT (Time To First Token) · TPS (Tokens Per Second) · Peak Memory

**Input:** Random token sequence of `n_prompt` length

### 5.3 SingleOp Track

**Operators:** Conv1x1, DWConv, MatMul, LayerNorm, Softmax, GELU, Concat (80+ test cases)

**Metrics:** Mean latency per operator

**Input:** ONNX models generated from PyTorch/ONNX API

---

## 6. Reproducibility

### 6.1 Audit Trail

Every benchmark run is stored in SQLite with:

```sql
run_id | timestamp | git_commit | framework | model | precision | threads |
warmup | test_runs | device_model | device_temp | metrics_json
```

### 6.2 Version Pinning

Submodules are pinned to official release tags:

| Submodule | Tag |
|-----------|-----|
| MNN | `3.6.1` |
| ONNX Runtime | `v1.28.0` |
| TVM | custom `v0.15.dev0` |
| llama.cpp | `b10121` |

### 6.3 Replication

To replicate results:

1. Checkout the commit recorded in the benchmark run
2. Build with the same CMake options
3. Run `benchctl run <track> <model> -f <frameworks> -t <threads> -r <runs>`
4. Compare `benchctl history <framework> <model>`

---

## 7. Limitations

1. **Single device:** Results are specific to Snapdragon 865 (SM8250). Performance rankings may differ on other SoCs.
2. **Fixed input:** Random input may not represent real-world data distribution.
3. **No power measurement:** MLPerf Mobile does not require power; this project follows the same convention.
4. **Framework optimizer differences:** Graph-level optimizations (operator fusion, layout transformation) happen at model conversion time and are not separately measured.
5. **Thread scheduling:** Thread affinity and scheduling policy are left to the framework's default, which may differ between frameworks.

---

## 8. References

- [MLPerf Inference: Mobile Benchmark (arXiv:2012.02328)](https://arxiv.org/abs/2012.02328)
- [MLPerf Mobile Inference Rules](https://github.com/mlcommons/mobile_open)
- [DLI: Deep Learning Inference Benchmark](https://github.com/itlab-vision/dl-benchmark)
