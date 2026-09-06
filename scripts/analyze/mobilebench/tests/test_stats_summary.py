"""Unit tests for scripts/stats_summary.py."""
from pathlib import Path

import pytest

from stats_summary import (MIN_RUNS, CV_WARN_THRESHOLD, build_rows,
                           collect_samples, group_samples, percentile,
                           summarize)


def write_log(path, backend, model, p50, precision="fp32", threads=4, fps=100.0):
    path.write_text(
        f"Backend: {backend}\nModel: {model}\nPrecision: {precision}\n"
        f"Threads: {threads}\nP50: {p50:.2f} ms\nP90: 9.5 ms\nP99: 9.9 ms\n"
        f"Throughput: {fps} FPS\n",
        encoding="utf-8",
    )


@pytest.fixture
def low_variance_dir(tmp_path: Path):
    """5 runs of the same config with tight P50 spread."""
    for i in range(5):
        write_log(tmp_path / f"run_{i}.log", "mnn", "mobilenetv2", 8.4 + i * 0.05)
    return tmp_path


@pytest.fixture
def high_variance_dir(tmp_path: Path):
    write_log(tmp_path / "a.log", "ncnn", "resnet50", 10.0, threads=2)
    write_log(tmp_path / "b.log", "ncnn", "resnet50", 15.0, threads=2)
    return tmp_path


def test_percentile_basic():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert percentile([1.0], 90) == 1.0
    assert percentile([], 50) == 0.0


def test_summarize_stats():
    s = summarize([8.0, 8.1, 8.2, 8.3, 8.4])
    assert s["mean"] == pytest.approx(8.2, abs=0.01)
    assert s["p50"] == pytest.approx(8.2, abs=0.01)
    assert s["cv"] == pytest.approx(1.72, abs=0.1)


def test_summarize_empty():
    assert summarize([]) is None


def test_collect_and_group(low_variance_dir):
    samples = collect_samples([str(low_variance_dir)])
    assert len(samples) == 5
    groups = group_samples(samples)
    assert len(groups) == 1
    key = ("mnn", "mobilenetv2", "fp32", 4)
    assert key in groups
    assert len(groups[key]) == 5


def test_low_variance_no_flag(low_variance_dir):
    rows = build_rows(group_samples(collect_samples([str(low_variance_dir)])))
    assert len(rows) == 1
    assert rows[0]["high_variance"] is False
    assert rows[0]["samples_min"] is True


def test_high_variance_flag(high_variance_dir):
    rows = build_rows(group_samples(collect_samples([str(high_variance_dir)])))
    assert rows[0]["high_variance"] is True
    assert rows[0]["latency_p50"]["cv"] > CV_WARN_THRESHOLD


def test_custom_threshold(high_variance_dir):
    # CV ~17%, threshold 20 -> no flag
    rows = build_rows(group_samples(collect_samples([str(high_variance_dir)])),
                      cv_threshold=20.0)
    assert rows[0]["high_variance"] is False


def test_insufficient_samples(tmp_path):
    write_log(tmp_path / "only.log", "tvm", "yolov8n", 12.0)
    rows = build_rows(group_samples(collect_samples([str(tmp_path)])))
    assert rows[0]["samples_min"] is False
    assert rows[0]["n"] == 1 < MIN_RUNS


def test_group_by_config(tmp_path):
    write_log(tmp_path / "a.log", "mnn", "mobilenetv2", 8.0, threads=4)
    write_log(tmp_path / "b.log", "mnn", "mobilenetv2", 8.1, threads=1)
    groups = group_samples(collect_samples([str(tmp_path)]))
    assert len(groups) == 2
