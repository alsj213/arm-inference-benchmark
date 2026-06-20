#!/usr/bin/env python3
"""
生成基准测试报告脚本
解析结果并生成 Markdown 格式的报告
"""

import sys
import os
import re
import glob
from datetime import datetime

def parse_log_file(log_path):
    """解析基准测试日志文件"""
    results = {
        'backend': None,
        'model': None,
        'precision': None,
        'threads': None,
        'accuracy_passed': False,
        'cosine_similarity': 0.0,
        'latency_p50': 0.0,
        'latency_p90': 0.0,
        'latency_p99': 0.0,
        'latency_mean': 0.0,
        'throughput_fps': 0.0,
        'memory_kb': 0,
        'init_time_ms': 0.0
    }

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取后端名称
        backend_match = re.search(r'Backend:\s*(\w+)', content)
        if backend_match:
            results['backend'] = backend_match.group(1)

        # 提取模型名称
        model_match = re.search(r'Model:\s*(\w+)', content)
        if model_match:
            results['model'] = model_match.group(1)

        # 提取精度
        precision_match = re.search(r'Precision:\s*(\w+)', content)
        if precision_match:
            results['precision'] = precision_match.group(1)

        # 提取线程数
        threads_match = re.search(r'Threads:\s*(\d+)', content)
        if threads_match:
            results['threads'] = int(threads_match.group(1))

        # 检查准确性
        if '✅ [Accuracy]' in content and 'PASSED' in content:
            results['accuracy_passed'] = True

        # 提取余弦相似度
        cosine_match = re.search(r'Cosine Similarity:\s*([\d.]+)', content)
        if cosine_match:
            results['cosine_similarity'] = float(cosine_match.group(1))

        # 提取延迟统计
        p50_match = re.search(r'P50:\s*([\d.]+)\s*ms', content)
        if p50_match:
            results['latency_p50'] = float(p50_match.group(1))

        p90_match = re.search(r'P90:\s*([\d.]+)\s*ms', content)
        if p90_match:
            results['latency_p90'] = float(p90_match.group(1))

        p99_match = re.search(r'P99:\s*([\d.]+)\s*ms', content)
        if p99_match:
            results['latency_p99'] = float(p99_match.group(1))

        mean_match = re.search(r'Mean:\s*([\d.]+)\s*ms', content)
        if mean_match:
            results['latency_mean'] = float(mean_match.group(1))

        # 提取吞吐量
        throughput_match = re.search(r'Throughput:\s*([\d.]+)\s*FPS', content)
        if throughput_match:
            results['throughput_fps'] = float(throughput_match.group(1))

        # 提取内存使用
        memory_match = re.search(r'Memory:\s*(\d+)\s*KB', content)
        if memory_match:
            results['memory_kb'] = int(memory_match.group(1))

        # 提取初始化时间
        init_match = re.search(r'Init time:\s*([\d.]+)\s*ms', content)
        if init_match:
            results['init_time_ms'] = float(init_match.group(1))

    except Exception as e:
        print(f"警告: 无法解析文件 {log_path}: {e}")

    return results


def generate_report(results_dir):
    """生成 Markdown 报告"""
    # 查找所有日志文件
    log_files = glob.glob(os.path.join(results_dir, "*.log"))

    if not log_files:
        print("错误: 未找到日志文件")
        return

    # 解析所有结果
    all_results = [parse_log_file(f) for f in log_files]

    # 按后端、模型、精度、线程分组
    grouped = {}
    for r in all_results:
        key = (r['backend'], r['model'], r['precision'], r['threads'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    # 生成报告
    report_path = os.path.join(results_dir, "benchmark_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 基准测试报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"测试目录: {results_dir}\n\n")
        f.write(f"总测试数: {len(all_results)}\n\n")

        # 按后端分组输出
        backends = sorted(set(r['backend'] for r in all_results if r['backend']))
        for backend in backends:
            f.write(f"\n## {backend.upper()}\n\n")
            f.write("| 模型 | 精度 | 线程 | P50(ms) | P99(ms) | Throughput(FPS) | Accuracy | Cosine Similarity |\n")
            f.write("|------|------|------|---------|---------|-----------------|----------|-------------------|\n")

            backend_results = [r for r in all_results if r['backend'] == backend]
            for r in sorted(backend_results, key=lambda x: (x['model'], x['precision'], x['threads'])):
                accuracy_str = "✅" if r['accuracy_passed'] else "❌"
                f.write(f"| {r['model']} | {r['precision']} | {r['threads']} | "
                       f"{r['latency_p50']:.2f} | {r['latency_p99']:.2f} | "
                       f"{r['throughput_fps']:.2f} | {accuracy_str} | {r['cosine_similarity']:.6f} |\n")

        # 性能对比总结
        f.write("\n## 性能对比总结\n\n")
        f.write("| 后端 | 平均 P50(ms) | 平均 P99(ms) | 平均吞吐量(FPS) |\n")
        f.write("|------|--------------|--------------|-----------------|\n")

        for backend in backends:
            backend_results = [r for r in all_results if r['backend'] == backend]
            avg_p50 = sum(r['latency_p50'] for r in backend_results) / len(backend_results)
            avg_p99 = sum(r['latency_p99'] for r in backend_results) / len(backend_results)
            avg_fps = sum(r['throughput_fps'] for r in backend_results) / len(backend_results)
            f.write(f"| {backend} | {avg_p50:.2f} | {avg_p99:.2f} | {avg_fps:.2f} |\n")

        # 准确性验证总结
        f.write("\n## 准确性验证\n\n")
        passed = sum(1 for r in all_results if r['accuracy_passed'])
        total = len(all_results)
        f.write(f"准确性通过率: {passed}/{total} ({passed/total*100:.1f}%)\n\n")

        if passed < total:
            f.write("### 失败的测试\n\n")
            for r in all_results:
                if not r['accuracy_passed']:
                    f.write(f"- {r['backend']} {r['model']} {r['precision']} {r['threads']} threads "
                           f"(cosine similarity: {r['cosine_similarity']:.6f})\n")

    print(f"报告已生成: {report_path}")
    return report_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 generate_report.py <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]
    generate_report(results_dir)
