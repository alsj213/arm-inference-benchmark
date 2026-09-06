"""Unit tests for scripts/regression_compare.py."""
import json
from pathlib import Path

import pytest

from regression_compare import (config_key_str, compare_runs, expand_results,
                                save_baseline, test_significance as significance_func)


def write_log(path, backend, model, p50, threads=4):
    path.write_text(
        f"Backend: {backend}\nModel: {model}\nPrecision: fp32\nThreads: {threads}\n"
        f"P50: {p50:.2f} ms\nThroughput: 100.0 FPS\n",
        encoding="utf-8",
    )


def make_logs(tmp_path, sub, p50s):
    d = tmp_path / sub
    d.mkdir(parents=True)
    for i, p in enumerate(p50s):
        write_log(d / f"run_{i}.log", "mnn", "mobilenetv2", p)
    return [str(f) for f in sorted(d.glob("*.log"))]


def test_config_key_str():
    assert config_key_str(("mnn", "mobilenetv2", "fp32", 4)) == "mnn|mobilenetv2|fp32|4"


def test_expand_results_shape(tmp_path):
    paths = make_logs(tmp_path, "base", [8.4, 8.5, 8.6])
    res = expand_results(paths)
    key = "mnn|mobilenetv2|fp32|4"
    assert key in res
    assert len(res[key]["p50_samples"]) == 3


def test_save_baseline(tmp_path):
    paths = make_logs(tmp_path, "base", [8.4, 8.5, 8.6])
    out = tmp_path / "baseline.json"
    # full env_snapshot structure, as produced by scripts/env_snapshot.py
    env = {
        "device": {"device_id": {"value": "x", "source": "cfg"},
                   "model": {"value": "M", "source": "adb"}},
        "code": {"commit": {"value": "abc", "source": "git"}},
    }
    save_baseline(expand_results(paths), str(out), env=env)
    data = json.loads(out.read_text())
    assert data["type"] == "mobile-bench-baseline"
    assert data["env"]["device_id"] == "x"
    assert "mnn|mobilenetv2|fp32|4" in data["configs"]


def test_compare_detects_regression(tmp_path):
    base = expand_results(make_logs(tmp_path, "base", [8.4] * 6))
    cur = expand_results(make_logs(tmp_path, "cur", [9.3] * 6))
    baseline = {"configs": base}
    rows = compare_runs(baseline, cur)
    r = rows[0]
    assert r["regression"] is True
    assert r["delta_pct"] > 5.0
    assert r["p_value"] is not None


def test_compare_no_regression(tmp_path):
    base = expand_results(make_logs(tmp_path, "base", [8.4] * 6))
    cur = expand_results(make_logs(tmp_path, "cur", [8.4] * 6))
    baseline = {"configs": base}
    rows = compare_runs(baseline, cur)
    assert rows[0]["regression"] is False


def test_compare_new_config(tmp_path):
    base = expand_results(make_logs(tmp_path, "base", [8.4] * 5))
    make_logs(tmp_path, "cur", [8.4] * 5)
    # add a new config only present in current
    (tmp_path / "cur" / "new.log").write_text(
        "Backend: tvm\nModel: yolo\nPrecision: fp32\nThreads: 1\n"
        "P50: 12.0 ms\nThroughput: 80 FPS\n", encoding="utf-8")
    cur = expand_results([str(f) for f in (tmp_path / "cur").glob("*.log")])
    baseline = {"configs": base}
    rows = compare_runs(baseline, cur)
    statuses = {r["status"] for r in rows}
    assert "NEW" in statuses


def test_significance_insufficient():
    r = significance_func([8.4], [9.3])
    assert r["significant"] is None
    assert "insufficient" in r["test"]


def test_significance_scipy(tmp_path):
    r = significance_func([8.4] * 6, [9.3] * 6)
    assert r["p_value"] is not None
