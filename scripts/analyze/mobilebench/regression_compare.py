#!/usr/bin/env python3
"""Baseline snapshot and performance regression comparison.

Saves a baseline.json (per-config P50 samples + commit + env snapshot fields),
then compares a later run against it, flagging statistically significant
regressions where P50 degrades more than the threshold (default 5%).

Significance testing uses scipy (Mann-Whitney U, or paired t-test with
--paired). When scipy is unavailable, falls back to descriptive comparison
and reports the lack of a significance test.

Usage:
    # Save a baseline
    python3 scripts/regression_compare.py save \
        --results results/base/*.log --out baseline.json \
        --env results/env_snapshot_*.json
    # Compare current run against the baseline
    python3 scripts/regression_compare.py compare \
        --baseline baseline.json --results results/current/*.log \
        --env results/env_snapshot_*.json
"""
import sys
import os
import json
import glob
import argparse
import subprocess
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_summary import collect_samples, group_samples  # noqa: E402

REGRESSION_THRESHOLD = 0.05  # 5% P50 degradation
ALPHA = 0.05
MIN_RUNS_FOR_TEST = 5

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def git_commit():
    """Return current git commit hash or None."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def load_env(path_pattern):
    """Load the newest matching env snapshot JSON, or None."""
    paths = sorted(glob.glob(path_pattern))
    if not paths:
        return None
    try:
        with open(paths[-1]) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def summarize_env(env):
    """Reduce a full env snapshot to comparable identity fields."""
    if not env:
        return {}
    device = env.get("device", {})
    code = env.get("code", {})
    return {
        "device_id": (device.get("device_id") or {}).get("value"),
        "model": (device.get("model") or {}).get("value"),
        "os": (device.get("os_version") or {}).get("value"),
        "commit": (code.get("commit") or {}).get("value"),
    }


def config_key_str(key):
    """Serialize a config key tuple to the baseline.json key string."""
    return "|".join(str(part) for part in key)


def expand_results(paths):
    """Parse result samples grouped by config key, in baseline shape.

    Returns {key_str: {"p50_samples": [float, ...]}} matching baseline.json's
    per-config structure, so compare can treat both uniformly.
    """
    grouped = group_samples(collect_samples(paths))
    return {
        config_key_str(key): {"p50_samples": [s["latency"]["p50"] for s in samples]}
        for key, samples in grouped.items()
    }


def save_baseline(results, out_path, env):
    """Persist a baseline.json from grouped P50 samples."""
    baseline = {
        "type": "mobile-bench-baseline",
        "version": 1,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit": git_commit(),
        "env": summarize_env(env),
        "configs": {
            key: {
                "key": key,
                "p50_samples": samples["p50_samples"],
            }
            for key, samples in results.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"Baseline saved: {out_path} ({len(results)} configs)")


def compare_runs(baseline, current, paired=False):
    """Compare baseline vs current configs; return list of result dicts."""
    base_cfgs = baseline.get("configs", {})
    rows = []
    for key, cur in current.items():
        base = base_cfgs.get(key)
        if base is None:
            rows.append({"key": key, "status": "NEW", "base_n": 0, "cur_n": len(cur["p50_samples"])})
            continue
        base_samples = base["p50_samples"]
        cur_samples = cur["p50_samples"]
        base_mean = sum(base_samples) / len(base_samples) if base_samples else 0
        cur_mean = sum(cur_samples) / len(cur_samples) if cur_samples else 0
        delta = (cur_mean - base_mean) / base_mean if base_mean else 0.0

        row = {
            "key": key,
            "status": "COMPARE",
            "base_n": len(base_samples),
            "cur_n": len(cur_samples),
            "base_mean_p50": round(base_mean, 4),
            "cur_mean_p50": round(cur_mean, 4),
            "delta_pct": round(delta * 100.0, 2),
            "p_value": None,
            "significant": None,
            "test": None,
        }
        row.update(test_significance(base_samples, cur_samples, paired))
        row["regression"] = bool(
            delta > REGRESSION_THRESHOLD
            and (row["significant"] is True or row["significant"] is None)
        )
        rows.append(row)
    return rows


def test_significance(base_samples, cur_samples, paired=False):
    """Run significance test; return {p_value, significant, test}."""
    if not HAS_SCIPY:
        return {"p_value": None, "significant": None,
                "test": "none (scipy unavailable)"}
    if len(base_samples) < MIN_RUNS_FOR_TEST or len(cur_samples) < MIN_RUNS_FOR_TEST:
        return {"p_value": None, "significant": None,
                "test": f"insufficient samples (< {MIN_RUNS_FOR_TEST})"}
    try:
        if paired:
            n = min(len(base_samples), len(cur_samples))
            stat, p = scipy_stats.ttest_rel(base_samples[:n], cur_samples[:n])
            test = "paired t-test"
        else:
            stat, p = scipy_stats.mannwhitneyu(base_samples, cur_samples, alternative="two-sided")
            test = "Mann-Whitney U"
    except ValueError:
        return {"p_value": None, "significant": None, "test": "test failed"}
    return {"p_value": round(float(p), 4), "significant": bool(p < ALPHA), "test": test}


def render_markdown(rows):
    lines = ["## 性能回归对比", ""]
    lines.append("| 配置 | base n | cur n | base P50(ms) | cur P50(ms) | Δ% | 检验 | p值 | 判定 |")
    lines.append("|------|--------|-------|--------------|--------------|-----|------|-----|------|")
    for r in rows:
        if r["status"] == "NEW":
            lines.append(f"| {r['key']} | - | {r['cur_n']} | - | - | - | - | - | **NEW** |")
            continue
        sig = "显著" if r["significant"] is True else ("不显著" if r["significant"] is False else "未检验")
        verdict = "⚠️ **REGRESSION**" if r["regression"] else ("✅" if r["delta_pct"] <= 0 else "ok")
        pv = r["p_value"] if r["p_value"] is not None else "-"
        lines.append(
            f"| {r['key']} | {r['base_n']} | {r['cur_n']} | {r['base_mean_p50']:.2f} "
            f"| {r['cur_mean_p50']:.2f} | {r['delta_pct']:+.2f} | {r['test'] or '-'} | {pv} | {verdict} |"
        )
    lines.append("")
    for r in rows:
        if r["status"] == "NEW":
            lines.append(f"> 🆕 {r['key']}: 基线中不存在该配置,仅当前测量有 {r['cur_n']} 轮")
            continue
        if r["regression"]:
            lines.append(
                f"> 🔴 **REGRESSION DETECTED**: {r['key']} Δ=+{r['delta_pct']:.2f}% "
                f"(base {r['base_mean_p50']:.2f} → cur {r['cur_mean_p50']:.2f} ms)"
            )
        elif r.get("test") == "none (scipy unavailable)":
            lines.append(f"> ℹ️ scipy 未安装,{r['key']} 未做显著性检验;pip install scipy 后重跑")
        elif r.get("significant") is None:
            lines.append(f"> ℹ️ {r['key']} 样本不足(< {MIN_RUNS_FOR_TEST}),判定为描述性对比")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    save = sub.add_parser("save", help="save a baseline")
    save.add_argument("--results", nargs="+", required=True, help="log files/dirs for the baseline run")
    save.add_argument("--out", required=True, help="baseline.json output path")
    save.add_argument("--env", default=None, help="env snapshot glob for baseline env fields")

    cmp = sub.add_parser("compare", help="compare current run against baseline")
    cmp.add_argument("--baseline", required=True, help="baseline.json path")
    cmp.add_argument("--results", nargs="+", required=True, help="log files/dirs for the current run")
    cmp.add_argument("--env", default=None, help="env snapshot glob for current env fields")
    cmp.add_argument("--paired", action="store_true", help="use paired t-test instead of Mann-Whitney U")
    cmp.add_argument("--json", action="store_true", help="emit JSON instead of markdown")

    args = parser.parse_args()

    if args.command == "save":
        results = expand_results(args.results)
        env = load_env(args.env) if args.env else None
        save_baseline(results, args.out, env)
        return

    # --compare
    try:
        with open(args.baseline) as f:
            baseline = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"Error: cannot read baseline: {exc}", file=sys.stderr)
        sys.exit(1)

    current = expand_results(args.results)
    if not current:
        print("Error: no valid samples in current run.", file=sys.stderr)
        sys.exit(1)

    rows = compare_runs(baseline, current, paired=args.paired)

    # env consistency note
    cur_env = summarize_env(load_env(args.env)) if args.env else {}
    base_env = baseline.get("env", {})
    env_diff = {k: (base_env.get(k), cur_env.get(k))
                for k in set(base_env) | set(cur_env)
                if base_env.get(k) != cur_env.get(k)}

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(rows))
        if env_diff:
            print("> ⚠️ 环境差异(对比可能失真):")
            for k, (b, c) in env_diff.items():
                print(f">   {k}: {b} → {c}")

    # nonzero exit when any regression found, for CI integration
    if any(r.get("regression") for r in rows):
        sys.exit(2)


if __name__ == "__main__":
    main()
