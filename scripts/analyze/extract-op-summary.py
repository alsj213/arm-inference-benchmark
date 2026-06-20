#!/usr/bin/env python3
"""从 benchmark 日志中提取模型 Latency 摘要（MNN Profiler 逐算子数据由 MNN 内部输出）"""
import sys, re, os

def extract_summary(log_path):
    """从 benchmark 输出提取 Latency 关键指标"""
    with open(log_path) as f:
        text = f.read()

    mean_match = re.search(r'Mean:\s+([\d.]+)\s*ms', text)
    p50_match = re.search(r'P50:\s+([\d.]+)\s*ms', text)
    p99_match = re.search(r'P99:\s+([\d.]+)\s*ms', text)
    p90_match = re.search(r'P90:\s+([\d.]+)\s*ms', text)
    min_match = re.search(r'Min:\s+([\d.]+)\s*ms', text)
    max_match = re.search(r'Max:\s+([\d.]+)\s*ms', text)
    std_match = re.search(r'Std:\s+([\d.]+)\s*ms', text)
    fps_match = re.search(r'Throughput:\s+([\d.]+)\s*FPS', text)
    init_match = re.search(r'Init time:\s+([\d.]+)\s*ms', text)
    acc_match = re.search(r'\[Accuracy\]\s+(\w+):\s+(\w+)', text)

    # Determine backend and model from filename
    fname = os.path.basename(log_path)
    model = fname.split('_')[0]
    if '_mnn_' in fname:
        backend = 'MNN'
    elif '_ort_' in fname:
        backend = 'ORT'
    else:
        backend = '?'

    # Special: if filename has "mnn_cpu_profile" pattern
    if 'mnn_cpu_profile' in fname:
        backend = 'MNN'
    elif 'ort_cpu_profile' in fname:
        backend = 'ORT'

    return {
        'model': model,
        'backend': backend,
        'mean_ms': float(mean_match.group(1)) if mean_match else None,
        'p50_ms': float(p50_match.group(1)) if p50_match else None,
        'p90_ms': float(p90_match.group(1)) if p90_match else None,
        'p99_ms': float(p99_match.group(1)) if p99_match else None,
        'min_ms': float(min_match.group(1)) if min_match else None,
        'max_ms': float(max_match.group(1)) if max_match else None,
        'std_ms': float(std_match.group(1)) if std_match else None,
        'fps': float(fps_match.group(1)) if fps_match else None,
        'init_ms': float(init_match.group(1)) if init_match else None,
        'accuracy': acc_match.group(2) if acc_match else 'N/A',
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_op_summary.py <log_file> [log_file...]")
        sys.exit(1)

    results = []
    for log_path in sys.argv[1:]:
        if os.path.exists(log_path):
            r = extract_summary(log_path)
            if r['mean_ms']:
                results.append(r)

    # Group by model for MNN vs ORT comparison
    models = {}
    for r in results:
        if r['model'] not in models:
            models[r['model']] = {}
        models[r['model']][r['backend']] = r

    print("\n## MNN vs ORT 全模型 Latency 对比")
    print()
    print("| 模型 | 后端 | Mean(ms) | P50(ms) | P90(ms) | P99(ms) | Std(ms) | FPS | Init(ms) | 精度 |")
    print("|------|------|----------|---------|---------|---------|---------|-----|----------|------|")
    for model_name in sorted(models.keys()):
        for backend in ['MNN', 'ORT']:
            if backend in models[model_name]:
                r = models[model_name][backend]
                print(f"| {r['model']} | {r['backend']} | {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | "
                      f"{r['p90_ms']:.2f} | {r['p99_ms']:.2f} | {r['std_ms']:.2f} | "
                      f"{r['fps']:.1f} | {r['init_ms']:.2f} | {r['accuracy']} |")

    print()
    print("## MNN vs ORT 加速比")
    print()
    print("| 模型 | MNN Mean(ms) | ORT Mean(ms) | MNN/ORT 加速比 | 胜出 |")
    print("|------|-------------|-------------|---------------|------|")
    for model_name in sorted(models.keys()):
        if 'MNN' in models[model_name] and 'ORT' in models[model_name]:
            mnn = models[model_name]['MNN']
            ort = models[model_name]['ORT']
            ratio = ort['mean_ms'] / mnn['mean_ms']
            winner = 'MNN' if ratio > 1 else 'ORT'
            print(f"| {model_name} | {mnn['mean_ms']:.2f} | {ort['mean_ms']:.2f} | {ratio:.2f}x | **{winner}** |")

    print()
    print("## 测试配置")
    print("- Warmup: 50 iterations, Test: 100 iterations")
    print("- CPU: 单线程 (Threads=1)")
    print("- Precision: FP32")
    print("- 频率策略: schedutil (未锁频)")
    print("- Accuracy: vs ONNX Runtime 参考输出")

if __name__ == '__main__':
    main()
