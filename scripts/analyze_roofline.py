#!/usr/bin/env python3
"""Roofline 模型分析脚本 — 对骁龙 865 SM8250

分析单算子 Benchmark 的实测延迟，标注每个算子的 Bound 类型
(Compute-Bound vs Memory-Bound)，并计算硬件利用率。

参考: Williams, Waterman, Patterson (2009) "Roofline: An Insightful Visual
Performance Model for Multicore Architectures"
"""

import math
import re
import sys
import os

# =============================================================================
# 骁龙 865 (SM8250) 硬件参数
# =============================================================================
# CPU: 1×A77 @ 2.84GHz + 3×A77 @ 2.42GHz + 4×A55 @ 1.8GHz
# 锁频大核后使用 4×A77 @ 2.42GHz
# 每个 A77: 4 FMA/cycle (2×128-bit NEON → 4 FP32 ops/cycle)
# 峰值: 4 cores × 2.42 GHz × 4 FMA = 38.72 GFLOPS FP32
# 实测 GNURoofline: ~25-30 GFLOPS (MNN 优化 + NEON intrinsics)
# 保守估计: 25 GFLOPS (考虑调度和效率损失)
PEAK_GFLOPS_FP32 = 25.0      # MNN 优化 NEON 实测可达
PEAK_BANDWIDTH_GBS = 34.0    # LPDDR5 理论 34 GB/s，实测可用 ~25-30 GB/s
ROOFLINE_SLOPE = PEAK_GFLOPS_FP32 / PEAK_BANDWIDTH_GBS  # ~0.74 FLOPS/Byte

# =============================================================================
# 算子参数字典 — 从测例名解析 (M=空间点数, C_in/C_out=通道, K=kernel size)
# =============================================================================
OP_PARAMS = {
    # Conv1x1 形状分桶
    'Conv1x1_M49_C32_K64':       {'type': 'conv', 'M': 49,   'C_in': 32,  'C_out': 64,  'K': 1},
    'Conv1x1_M49_C256_K512':     {'type': 'conv', 'M': 49,   'C_in': 256, 'C_out': 512, 'K': 1},
    'Conv1x1_M784_C32_K64':      {'type': 'conv', 'M': 784,  'C_in': 32,  'C_out': 64,  'K': 1},
    'Conv1x1_M3136_C64_K128':    {'type': 'conv', 'M': 3136, 'C_in': 64,  'C_out': 128, 'K': 1},
    'Conv1x1_K16_C64_M784':      {'type': 'conv', 'M': 784,  'C_in': 64,  'C_out': 16,  'K': 1},
    'Conv1x1_K1024_C256_M784':   {'type': 'conv', 'M': 784,  'C_in': 256, 'C_out': 1024,'K': 1},
    # NCHW4c 对齐退化
    'Conv1x1_Misaligned_C31_K64': {'type': 'conv', 'M': 3136, 'C_in': 31,  'C_out': 64,  'K': 1},
    'Conv1x1_Misaligned_C33_K64': {'type': 'conv', 'M': 3136, 'C_in': 33,  'C_out': 64,  'K': 1},
    # DWConv 极端通道
    'DWConv_C960_3x3':           {'type': 'conv', 'M': 49,   'C_in': 960, 'C_out': 960,'K': 3, 'groups': 960},
    'DWConv_C16_3x3':            {'type': 'conv', 'M': 12544,'C_in': 16,  'C_out': 16, 'K': 3, 'groups': 16},
    # MatMul 方阵 vs 长矩阵
    'MatMul_768x768x768':        {'type': 'matmul', 'M': 1, 'N': 768, 'K': 768},
    'MatMul_512x512x512':        {'type': 'matmul', 'M': 1, 'N': 512, 'K': 512},
    'MatMul_768x3072':           {'type': 'matmul', 'M': 1, 'N': 3072, 'K': 768},
    'MatMul_3072x768':           {'type': 'matmul', 'M': 1, 'N': 768, 'K': 3072},
    # 基础测例（来自 Phase 0）
    'Conv2d_1x1_s1':            {'type': 'conv', 'M': 3136, 'C_in': 64,  'C_out': 128, 'K': 1},
    'Conv2d_3x3_s1':            {'type': 'conv', 'M': 3136, 'C_in': 64,  'C_out': 128, 'K': 3},
    'DepthwiseConv_3x3_s1':     {'type': 'conv', 'M': 3136, 'C_in': 64,  'C_out': 64, 'K': 3, 'groups': 64},
    'Linear_512_512':           {'type': 'matmul', 'M': 1, 'N': 512, 'K': 512},
    'Linear_1024_1024':         {'type': 'matmul', 'M': 1, 'N': 1024, 'K': 1024},
}


def compute_conv_flops_and_bytes(M, C_in, C_out, K, groups=1):
    """计算 Conv 的 FLOPS 和访存量 (FP16)

    M: 空间点数 (H*W)
    C_in: 输入通道数
    C_out: 输出通道数
    K: kernel size
    groups: 分组数 (DWConv 时 groups=C_in=C_out)

    返回: (flops, bytes_read)
    """
    # FLOPS: 每个输出元素 = 2*(C_in/groups)*K*K MACs
    flops = 2 * M * (C_in // groups) * C_out * K * K

    # 访存量: input (M*C_in*2B FP16) + weight (C_in*C_out/groups*K*K*2B)
    bytes_read = 2 * (M * C_in + C_in * C_out // groups * K * K)

    return flops, bytes_read


def compute_matmul_flops_and_bytes(M, N, K):
    """计算 MatMul(M,K) × (K,N) 的 FLOPS 和访存量 (FP16)

    返回: (flops, bytes_read)
    """
    flops = 2 * M * N * K
    bytes_read = 2 * (M * K + K * N)
    return flops, bytes_read


def roofline_analysis(op_name, flops, bytes_read, measured_seconds):
    """Roofline 分析: 标注算子 Bound 类型并计算利用率

    返回: dict with op, ai, achieved_gflops, bound, utilization_pct
    """
    ai = flops / max(bytes_read, 1)  # Arithmetic Intensity (FLOPS/Byte)

    achieved_gflops = (flops / 1e9) / max(measured_seconds, 1e-9)

    if ai >= ROOFLINE_SLOPE:
        bound = "Compute-Bound"
        utilization = achieved_gflops / PEAK_GFLOPS_FP32 * 100
    else:
        bound = "Memory-Bound"
        peak_possible = PEAK_BANDWIDTH_GBS * ai
        utilization = achieved_gflops / max(peak_possible, 1e-3) * 100

    return {
        'op': op_name,
        'ai': ai,
        'achieved_gflops': achieved_gflops,
        'bound': bound,
        'utilization_pct': utilization
    }


def extract_latency_from_log(log_file):
    """从 benchmark 日志中提取平均延迟 (秒)

    日志格式: Mean: X.XX ms
    如果提取失败，返回 None
    """
    if not os.path.exists(log_file):
        print(f"⚠️  Log not found: {log_file}", file=sys.stderr)
        return None

    with open(log_file, 'r') as f:
        text = f.read()

    # 匹配 "mean=X.XXms" (当前 single_op_benchmark 格式)
    m = re.search(r'mean=([\d.]+)ms', text)
    if m:
        return float(m.group(1)) / 1000.0

    # 备选: "Mean: X.XX ms"
    m = re.search(r'Mean:\s+([\d.]+)\s*ms', text)
    if m:
        return float(m.group(1)) / 1000.0

    # 备选: "avg = X.XX ms"
    m = re.search(r'avg\s*=\s*([\d.]+)\s*ms', text)
    if m:
        return float(m.group(1)) / 1000.0

    return None


def extract_latency_and_threads(log_file):
    """从日志中提取延迟和线程数"""
    t = extract_latency_from_log(log_file)
    if t is None:
        return None, None

    with open(log_file, 'r') as f:
        text = f.read()

    m = re.search(r'threads[:\s=]+(\d+)', text, re.IGNORECASE)
    threads = int(m.group(1)) if m else 1

    return t, threads


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/phase2_hotspot_benchmark'

    if not os.path.isdir(log_dir):
        print(f"Error: log directory '{log_dir}' not found", file=sys.stderr)
        sys.exit(1)

    results = []
    thread_results = {}

    for op_name, params in OP_PARAMS.items():
        log_file = os.path.join(log_dir, f'{op_name}_mnn.log')

        if not os.path.exists(log_file):
            # 尝试不带 _mnn 后缀
            alt_log = os.path.join(log_dir, f'{op_name}.log')
            if os.path.exists(alt_log):
                log_file = alt_log
            else:
                continue

        t = extract_latency_from_log(log_file)
        if t is None:
            print(f"⚠️  Cannot parse latency from: {log_file}", file=sys.stderr)
            continue

        # 计算 FLOPS 和访存量
        if params['type'] == 'conv':
            groups = params.get('groups', 1)
            flops, br = compute_conv_flops_and_bytes(
                params['M'], params['C_in'], params['C_out'], params['K'],
                groups=groups
            )
        elif params['type'] == 'matmul':
            flops, br = compute_matmul_flops_and_bytes(
                params['M'], params['N'], params['K']
            )
        else:
            continue

        r = roofline_analysis(op_name, flops, br, t)
        results.append(r)

        # 检查是否是多线程测例
        thr = 1
        with open(log_file, 'r') as f:
            text = f.read()
        m = re.search(r'threads[:\s=]+(\d+)', text, re.IGNORECASE)
        if m:
            thr = int(m.group(1))

        key = f'{op_name}_t{thr}'
        thread_results[key] = {
            'op': op_name,
            'threads': thr,
            'latency_ms': t * 1000,
            'achieved_gflops': r['achieved_gflops'],
            'bound': r['bound']
        }

    # =========================================================================
    # 输出分析表
    # =========================================================================
    print("\n# Phase 2 Roofline 分析结果\n")
    print(f"## 硬件参数\n")
    print(f"- **峰值算力**: {PEAK_GFLOPS_FP32:.0f} GFLOPS (FP32, MNN NEON 优化实测)")
    print(f"- **内存带宽**: {PEAK_BANDWIDTH_GBS:.0f} GB/s (LPDDR5)")
    print(f"- **Roofline 斜面**: {ROOFLINE_SLOPE:.2f} FLOPS/Byte")
    print(f"- **芯片**: 骁龙 865 (SM8250), 4×A77 @ 2.42GHz\n")

    print("## 算子 Roofline 分析\n")
    print("| 算子 | 算术强度 | GFLOPS | Bound 类型 | 利用率(%) |")
    print("|------|---------|--------|-----------|----------|")

    for r in sorted(results, key=lambda x: x['achieved_gflops'], reverse=True):
        bound_icon = "🔴" if r['bound'] == 'Compute-Bound' else "🟡"
        print(f"| {r['op']} | {r['ai']:.1f} | {r['achieved_gflops']:.2f} | "
              f"{bound_icon} {r['bound']} | {r['utilization_pct']:.1f}% |")

    # 分类汇总
    compute_bound = [r for r in results if r['bound'] == 'Compute-Bound']
    memory_bound = [r for r in results if r['bound'] == 'Memory-Bound']

    print(f"\n## 分类汇总\n")
    print(f"| Bound 类型 | 算子数 | 平均利用率 | 最高利用率 |")
    print(f"|-----------|--------|----------|----------|")
    if compute_bound:
        avg_util_cb = sum(r['utilization_pct'] for r in compute_bound) / len(compute_bound)
        max_util_cb = max(r['utilization_pct'] for r in compute_bound)
        print(f"| Compute-Bound | {len(compute_bound)} | {avg_util_cb:.1f}% | {max_util_cb:.1f}% |")
    if memory_bound:
        avg_util_mb = sum(r['utilization_pct'] for r in memory_bound) / len(memory_bound)
        max_util_mb = max(r['utilization_pct'] for r in memory_bound)
        print(f"| Memory-Bound | {len(memory_bound)} | {avg_util_mb:.1f}% | {max_util_mb:.1f}% |")

    # 多线程加速比
    print(f"\n## 多线程加速比\n")
    print("| 算子 | 线程数 | 延迟 (ms) | GFLOPS | Bound |")
    print("|------|--------|----------|--------|-------|")
    for key, r in sorted(thread_results.items()):
        print(f"| {r['op']} | {r['threads']} | {r['latency_ms']:.4f} | "
              f"{r['achieved_gflops']:.2f} | {r['bound']} |")

    print()


if __name__ == '__main__':
    main()
