#!/usr/bin/env python3
"""
分析单算子性能基准测试结果
生成各框架对比报告和可视化图表
"""

import sys
import os
import csv
from collections import defaultdict

def load_results(csv_file):
    """加载 CSV 结果文件"""
    results = defaultdict(dict)
    backends = set()

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_name = row['op_name']
            backend = row['backend']
            backends.add(backend)
            results[op_name][backend] = {
                'mean_ms': float(row['mean_ms']),
                'min_ms': float(row['min_ms']),
                'max_ms': float(row['max_ms']),
                'std_ms': float(row['std_ms'])
            }

    return results, sorted(backends)

def print_summary_table(results, backends):
    """打印汇总表格"""
    print("\n" + "=" * 120)
    print(f"{'Operator':<30}", end='')
    for b in backends:
        print(f"{b:>15}", end='')
    print(f"{'Best':>10}")
    print("-" * 120)

    total = {b: 0.0 for b in backends}
    wins = {b: 0 for b in backends}

    for op in sorted(results.keys()):
        print(f"{op:<30}", end='')
        values = []
        for b in backends:
            if b in results[op]:
                v = results[op][b]['mean_ms']
                values.append((b, v))
                total[b] += v
                print(f"{v:>14.3f}ms", end='')
            else:
                print(f"{'N/A':>15}", end='')

        if values:
            best = min(values, key=lambda x: x[1])
            wins[best[0]] += 1
            print(f"{best[0]:>10}")
        else:
            print()

    print("-" * 120)
    print(f"{'TOTAL':<30}", end='')
    for b in backends:
        print(f"{total[b]:>14.3f}ms", end='')
    print()

    print("\n=== Wins by Backend ===")
    for b in sorted(wins.keys(), key=lambda x: wins[x], reverse=True):
        print(f"  {b:<20}: {wins[b]} operators fastest")


def print_category_breakdown(results, backends):
    """按算子类别分类统计"""
    categories = {
        'Convolution': ['Conv2d', 'ConvTranspose', 'Depthwise', 'DWConv'],
        'Pooling': ['Pool', 'AdaptiveAvg'],
        'Activation': ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'PReLU', 'GELU', 'Hardswish'],
        'Normalization': ['BatchNorm', 'InstanceNorm', 'LayerNorm'],
        'Linear': ['Linear'],
        'ElementWise': ['Add', 'Mul', 'Div', 'Pow', 'Sqrt', 'Exp'],
        'ShapeOp': ['Upsample', 'Pad', 'Flatten', 'Reshape'],
        'Reduce': ['ReduceMean', 'ReduceSum'],
        'Softmax': ['Softmax'],
        'Fused': ['Conv_BN']
    }

    print("\n" + "=" * 120)
    print("Performance by Operator Category")
    print("=" * 120)

    for category, patterns in categories.items():
        cat_ops = []
        for op in results.keys():
            for p in patterns:
                if p in op:
                    cat_ops.append(op)
                    break

        if not cat_ops:
            continue

        print(f"\n--- {category} ({len(cat_ops)} ops) ---")
        print(f"{'Backend':<15} {'Total ms':>12} {'Avg ms':>12}")
        print("-" * 40)

        for b in backends:
            total = sum(results[op][b]['mean_ms'] for op in cat_ops if b in results[op])
            avg = total / len([op for op in cat_ops if b in results[op]]) if cat_ops else 0
            print(f"{b:<15} {total:>12.3f} {avg:>12.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_single_op_results.py <results_csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    print(f"Analyzing results from: {csv_file}")
    results, backends = load_results(csv_file)

    print(f"Loaded {len(results)} operators across {len(backends)} backends")

    print_summary_table(results, backends)
    print_category_breakdown(results, backends)

    print("\n" + "=" * 120)
    print("Analysis complete!")
    print("=" * 120)


if __name__ == '__main__':
    main()
