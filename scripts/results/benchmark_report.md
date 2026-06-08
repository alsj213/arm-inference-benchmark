# Benchmark Report

Generated: 2026-05-27 00:31:25

Directory: results/

Total tests: 10


## MNN

| Model | Precision | Threads | P50(ms) | P99(ms) | Throughput(FPS) | Accuracy | Cosine Similarity |
|------|-----------|---------|---------|---------|-----------------|----------|-------------------|
| mobilenetv2 | fp32 | 1 | 18.60 | 18.87 | 53.76 | FAIL | 0.000000 |
| mobilenetv2 | fp32 | 1 | 18.67 | 19.20 | 53.53 | FAIL | 0.000000 |

## ALL

| Model | Precision | Threads | P50(ms) | P99(ms) | Throughput(FPS) | Accuracy | Cosine Similarity |
|------|-----------|---------|---------|---------|-----------------|----------|-------------------|
| all | fp32 | 4 | 8.70 | 9.38 | 114.33 | PASS | 1.000000 |
| yolov8n | fp32 | 4 | 329.49 | 595.46 | 3.04 | PASS | 1.000000 |

## MINDSPORE_LITE

| Model | Precision | Threads | P50(ms) | P99(ms) | Throughput(FPS) | Accuracy | Cosine Similarity |
|------|-----------|---------|---------|---------|-----------------|----------|-------------------|
| mobilenetv2 | fp32 | 1 | 0.24 | 0.29 | 4088.59 | FAIL | 0.000000 |

## MNN

| Model | Precision | Threads | P50(ms) | P99(ms) | Throughput(FPS) | Accuracy | Cosine Similarity |
|------|-----------|---------|---------|---------|-----------------|----------|-------------------|
| all | fp32 | 4 | 8.55 | 8.86 | 116.47 | FAIL | 0.000000 |
| mobilenetv2 | fp32 | 4 | 8.56 | 9.79 | 115.81 | PASS | 1.000000 |
| mobilenetv2 | fp32 | 4 | 8.54 | 8.70 | 116.98 | FAIL | 0.000000 |
| yolov8n | fp32 | 4 | 74.27 | 77.64 | 13.42 | FAIL | 0.000000 |

## ORT

| Model | Precision | Threads | P50(ms) | P99(ms) | Throughput(FPS) | Accuracy | Cosine Similarity |
|------|-----------|---------|---------|---------|-----------------|----------|-------------------|
| all | fp32 | 4 | 19.74 | 20.24 | 50.58 | FAIL | 0.000000 |

## Performance Summary

| Backend | Avg P50(ms) | Avg P99(ms) | Avg Throughput(FPS) |
|---------|-------------|-------------|---------------------|
| MNN | 18.63 | 19.04 | 53.64 |
| all | 169.09 | 302.42 | 58.69 |
| mindspore_lite | 0.24 | 0.29 | 4088.59 |
| mnn | 24.98 | 26.25 | 90.67 |
| ort | 19.74 | 20.24 | 50.58 |

## Accuracy Verification

Pass rate: 3/10 (30.0%)

### Failed Tests

- MNN mobilenetv2 fp32 1 threads (cosine similarity: 0.000000)
- ort all fp32 4 threads (cosine similarity: 0.000000)
- mnn all fp32 4 threads (cosine similarity: 0.000000)
- mnn yolov8n fp32 4 threads (cosine similarity: 0.000000)
- MNN mobilenetv2 fp32 1 threads (cosine similarity: 0.000000)
- mindspore_lite mobilenetv2 fp32 1 threads (cosine similarity: 0.000000)
- mnn mobilenetv2 fp32 4 threads (cosine similarity: 0.000000)
