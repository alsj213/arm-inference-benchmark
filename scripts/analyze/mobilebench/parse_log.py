#!/usr/bin/env python3
"""Parse benchmark log files into structured data."""
import sys
import json
import re


def normalize_real_json(obj):
    """Normalize a real benchmark JSON object into the downstream result shape.

    Real device output (benchmark_inference --json) looks like:
      {"framework": "MNN", "model": "mobilenetv2", "precision": "fp32",
       "threads": 1, "init_time_ms": ..., "peak_memory_kb": ...,
       "metrics": {"p50_ms": ..., "p90_ms": ..., "p99_ms": ...,
                   "mean_ms": ..., "throughput_fps": ...},
       "accuracy": {"cosine_similarity": ..., "passed": ...}}
    which is mapped to the same {backend, model, latency:{p50,...}, ...} shape
    that parse_text produces, so downstream consumers stay uniform.
    """
    if not isinstance(obj, dict) or "metrics" not in obj:
        return obj
    metrics = obj.get("metrics") or {}
    accuracy = obj.get("accuracy") or {}
    return {
        "backend": obj.get("framework") or obj.get("backend"),
        "model": obj.get("model"),
        "precision": obj.get("precision"),
        "threads": obj.get("threads"),
        "latency": {
            "p50": metrics.get("p50_ms"),
            "p90": metrics.get("p90_ms"),
            "p99": metrics.get("p99_ms"),
            "mean": metrics.get("mean_ms"),
        },
        "throughput_fps": metrics.get("throughput_fps"),
        "init_time_ms": obj.get("init_time_ms"),
        "memory_kb": obj.get("peak_memory_kb"),
        "accuracy_passed": bool(accuracy.get("passed")),
        "cosine_similarity": accuracy.get("cosine_similarity"),
        "raw": obj,
    }


def parse_json_lines(content):
    """Parse benchmark JSON lines.

    Accepts both the documented "JSON_RESULT: {...}" prefix and the real
    device output of a bare JSON object on its own line ({"framework": ...}).
    """
    results = []
    for line in content.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('JSON_RESULT: '):
            try:
                results.append(normalize_real_json(json.loads(stripped[13:])))
            except json.JSONDecodeError:
                pass
        elif stripped.startswith('{'):
            try:
                results.append(normalize_real_json(json.loads(stripped)))
            except json.JSONDecodeError:
                pass
    return results


def parse_text(content):
    """Fallback: parse human-readable benchmark output."""
    results = []
    # Extract backend
    backend_match = re.search(r'Backend:\s*(\w+)', content)
    model_match = re.search(r'Model:\s*(\w+)', content)
    precision_match = re.search(r'Precision:\s*(\w+)', content)
    threads_match = re.search(r'Threads:\s*(\d+)', content)
    p50_match = re.search(r'P50:\s*([\d.]+)\s*ms', content)
    p90_match = re.search(r'P90:\s*([\d.]+)\s*ms', content)
    p99_match = re.search(r'P99:\s*([\d.]+)\s*ms', content)
    mean_match = re.search(r'Mean:\s*([\d.]+)\s*ms', content)
    fps_match = re.search(r'Throughput:\s*([\d.]+)\s*FPS', content)
    init_match = re.search(r'Init time:\s*([\d.]+)\s*ms', content)
    memory_match = re.search(r'Memory:\s*(\d+)\s*KB', content)
    cosine_match = re.search(r'Cosine Similarity:\s*([\d.]+)', content)
    accuracy_passed = '✅ [Accuracy] PASSED' in content

    if p50_match:
        results.append({
            'backend': backend_match.group(1) if backend_match else 'unknown',
            'model': model_match.group(1) if model_match else 'unknown',
            'precision': precision_match.group(1) if precision_match else None,
            'threads': int(threads_match.group(1)) if threads_match else None,
            'latency': {
                'p50': float(p50_match.group(1)),
                # M2 修复:缺省 None(未测量),不再填 0.0 伪装成真实样本
                'p90': float(p90_match.group(1)) if p90_match else None,
                'p99': float(p99_match.group(1)) if p99_match else None,
                'mean': float(mean_match.group(1)) if mean_match else None,
            },
            'throughput_fps': float(fps_match.group(1)) if fps_match else None,
            'init_time_ms': float(init_match.group(1)) if init_match else None,
            'memory_kb': int(memory_match.group(1)) if memory_match else None,
            'accuracy_passed': accuracy_passed,
            'cosine_similarity': float(cosine_match.group(1)) if cosine_match else None,
        })
    return results


def parse_log(filepath):
    """Parse a benchmark log file, returning structured results.

    Tries JSON lines mode first; falls back to text regex parsing.
    """
    with open(filepath) as f:
        content = f.read()
    results = parse_json_lines(content)
    if not results:
        results = parse_text(content)
    return results


def main():
    if len(sys.argv) < 2:
        print('Usage: parse_log.py <logfile> [logfile...]')
        sys.exit(1)

    all_results = {}
    for path in sys.argv[1:]:
        results = parse_log(path)
        all_results[path] = results
        print(f'{path}: {len(results)} result(s)')

    if len(sys.argv) > 2:
        print(json.dumps(all_results, indent=2))
    elif all_results:
        for path, results in all_results.items():
            if results:
                print(json.dumps(results, indent=2))
                break


if __name__ == '__main__':
    main()
