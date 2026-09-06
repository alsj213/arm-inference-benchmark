#!/usr/bin/env python3
"""Multi-run statistics for benchmark measurements.

Aggregates repeated measurements (>=5 runs recommended) across one or more
benchmark log files, groups them by (backend, model, precision, threads), and
reports mean / std / CV / P50 / P90 / P99. Flags groups whose P50 CV exceeds
5% as high variance.

Reuses parse_log.py for log parsing.

Usage:
    python3 scripts/stats_summary.py results/benchmark_*.log
    python3 scripts/stats_summary.py --json results/*.log
"""
import sys
import os
import json
import glob
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_log import parse_log  # noqa: E402

CV_WARN_THRESHOLD = 5.0  # % - CV above this flags high variance
MIN_RUNS = 5  # spec: formal measurement SHALL run at least 5 rounds


def percentile(sorted_vals, pct):
    """Linear-interpolated percentile (0..100) over a sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def summarize(values):
    """Return dict of mean/std/cv/p50/p90/p99 for a list of floats."""
    if not values:
        return None
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) >= 2 else 0.0
    cv = (std / mean * 100.0) if mean else 0.0
    s = sorted(values)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cv": round(cv, 2),
        "p50": round(percentile(s, 50), 4),
        "p90": round(percentile(s, 90), 4),
        "p99": round(percentile(s, 99), 4),
    }


def collect_samples(inputs):
    """Expand input paths (files or dirs) into parsed sample dicts."""
    samples = []
    for path in inputs:
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.log")))
        else:
            files = [path]
        for f in files:
            try:
                results = parse_log(f)
            except Exception as exc:
                print(f"Warning: cannot parse {f}: {exc}", file=sys.stderr)
                continue
            for r in results:
                if r.get("backend") is None:
                    continue
                samples.append({**r, "_log": f})
    return samples


def group_key(r):
    return (r.get("backend"), r.get("model"), r.get("precision"), r.get("threads"))


def group_samples(samples):
    """Group samples by config key, preserving insertion order."""
    groups = {}
    for s in samples:
        key = group_key(s)
        groups.setdefault(key, []).append(s)
    return groups


def build_rows(groups, cv_threshold=CV_WARN_THRESHOLD):
    """Build per-group statistics rows."""
    rows = []
    for (backend, model, precision, threads), rs in groups.items():
        p50 = summarize([r.get("latency", {}).get("p50") or 0 for r in rs])
        p99 = summarize([r.get("latency", {}).get("p99") or 0 for r in rs])
        fps = summarize([r.get("throughput_fps") or 0 for r in rs])
        row = {
            "backend": backend,
            "model": model,
            "precision": precision,
            "threads": threads,
            "n": len(rs),
            "samples_min": len(rs) >= MIN_RUNS,
            "latency_p50": p50,
            "latency_p99": p99,
            "fps": fps,
            "high_variance": bool(p50 and p50["cv"] > cv_threshold),
        }
        rows.append(row)
    return rows


def render_markdown(rows):
    lines = ["## 多轮统计结果", ""]
    lines.append("| backend | model | precision | threads | n | P50 mean(ms) | P50 std | CV% | P90(ms) | P99(ms) | FPS mean | 方差 |")
    lines.append("|---------|-------|-----------|---------|---|--------------|---------|-----|---------|---------|----------|------|")
    for r in rows:
        p50 = r["latency_p50"] or {}
        p99 = r["latency_p99"] or {}
        fps = r["fps"] or {}
        warn = "**HIGH**" if r["high_variance"] else ("insufficient" if not r["samples_min"] else "ok")
        lines.append(
            f"| {r['backend']} | {r['model']} | {r['precision']} | {r['threads']} | {r['n']} "
            f"| {p50.get('mean', 0):.2f} | {p50.get('std', 0):.2f} | {p50.get('cv', 0):.1f} "
            f"| {p50.get('p90', 0):.2f} | {p99.get('p99', 0):.2f} | {fps.get('mean', 0):.2f} | {warn} |"
        )
    lines.append("")
    for r in rows:
        if r["high_variance"]:
            lines.append(
                f"> ⚠️ **High variance (CV={r['latency_p50']['cv']:.1f}%)** "
                f"for {r['backend']} {r['model']} (n={r['n']}). "
                f"Check thermal throttling / background processes / env control."
            )
        elif not r["samples_min"]:
            lines.append(
                f"> ℹ️ {r['backend']} {r['model']}: only {r['n']} run(s); "
                f"spec requires >= {MIN_RUNS} for formal statistics."
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="log files or directories containing *.log")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    parser.add_argument("--cv-threshold", type=float, default=CV_WARN_THRESHOLD,
                        help=f"CV%% warning threshold (default {CV_WARN_THRESHOLD})")
    args = parser.parse_args()

    samples = collect_samples(args.inputs)
    if not samples:
        print("Error: no valid benchmark samples found.", file=sys.stderr)
        sys.exit(1)

    rows = build_rows(group_samples(samples), cv_threshold=args.cv_threshold)

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(rows))


if __name__ == "__main__":
    main()
