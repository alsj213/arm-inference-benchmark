#!/usr/bin/env python3
"""
验证基准测试结果脚本
检查所有框架的准确性、检测异常结果
"""

import sys
import os
import re
import glob

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
        'latency_p99': 0.0,
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

        p99_match = re.search(r'P99:\s*([\d.]+)\s*ms', content)
        if p99_match:
            results['latency_p99'] = float(p99_match.group(1))

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


def validate_results(results_dir):
    """验证所有测试结果"""
    print("=" * 60)
    print("验证基准测试结果")
    print("=" * 60)

    # 查找所有日志文件
    log_files = glob.glob(os.path.join(results_dir, "*.log"))

    if not log_files:
        print("错误: 未找到日志文件")
        return False

    all_results = []
    accuracy_failures = []
    anomaly_results = []

    for log_path in log_files:
        results = parse_log_file(log_path)
        all_results.append(results)

        # 检查准确性
        if not results['accuracy_passed']:
            accuracy_failures.append(results)

        # 检测异常结果（延迟过高或吞吐量过低）
        if results['latency_p99'] > 1000:  # P99 延迟超过 1 秒
            anomaly_results.append(results)

    # 输出验证结果
    print(f"\n总测试数: {len(all_results)}")
    print(f"准确性通过: {len(all_results) - len(accuracy_failures)}")
    print(f"准确性失败: {len(accuracy_failures)}")
    print(f"异常结果: {len(anomaly_results)}")

    # 输出准确性失败的测试
    if accuracy_failures:
        print("\n" + "=" * 60)
        print("准确性失败的测试:")
        print("=" * 60)
        for r in accuracy_failures:
            print(f"  {r['backend']} {r['model']} {r['precision']} {r['threads']} threads")
            print(f"    余弦相似度: {r['cosine_similarity']:.6f}")

    # 输出异常结果
    if anomaly_results:
        print("\n" + "=" * 60)
        print("异常结果（延迟过高）:")
        print("=" * 60)
        for r in anomaly_results:
            print(f"  {r['backend']} {r['model']} {r['precision']} {r['threads']} threads")
            print(f"    P99 延迟: {r['latency_p99']:.2f} ms")

    # 总体验证结果
    print("\n" + "=" * 60)
    if len(accuracy_failures) == 0 and len(anomaly_results) == 0:
        print("✅ 所有测试通过验证!")
        return True
    else:
        print("⚠️  部分测试存在问题，请检查上述失败和异常结果")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 validate_results.py <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]
    success = validate_results(results_dir)
    sys.exit(0 if success else 1)
