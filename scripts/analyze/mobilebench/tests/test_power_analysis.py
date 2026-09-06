"""Unit tests for scripts/power_analysis.py."""
from pathlib import Path

import pytest

from power_analysis import (battery_delta_energy, compute_metrics,
                            parse_batterystats, parse_perfetto)


def write(path, content):
    Path(path).write_text(content, encoding="utf-8")
    return str(path)


def test_parse_perfetto_energy_j(tmp_path):
    p = write(tmp_path / "e.json", '{"energy_j": 120.0}')
    assert parse_perfetto(p) == {"energy_j": 120.0}


def test_parse_perfetto_energy_uws(tmp_path):
    p = write(tmp_path / "e.json", '{"energy_uws": 120000000}')
    assert parse_perfetto(p) == {"energy_j": 120.0}


def test_parse_perfetto_power_rails(tmp_path):
    p = write(tmp_path / "e.json",
              '{"power_rails": [{"name": "cpu", "energy_uj": 50000000}]}')
    assert parse_perfetto(p) == {"energy_j": 50.0}


def test_parse_perfetto_no_energy(tmp_path):
    p = write(tmp_path / "e.json", '{"foo": 1}')
    assert "error" in parse_perfetto(p)


def test_parse_batterystats_same_line(tmp_path):
    p = write(tmp_path / "b.txt", "Estimated power use (mAh): 1080.0\n")
    assert parse_batterystats(p) == {"mah": 1080.0}


def test_parse_batterystats_next_line_bare(tmp_path):
    p = write(tmp_path / "b.txt", "Estimated power use (mAh):\n  1080.0\n")
    assert parse_batterystats(p) == {"mah": 1080.0}


def test_parse_batterystats_drain_positive(tmp_path):
    p = write(tmp_path / "b.txt",
              "Estimated power use (mAh):\n"
              "  Capacity: 5000, Computed drain: 42.5, actual drain: 40.0\n")
    assert parse_batterystats(p) == {"mah": 42.5}


def test_parse_batterystats_capacity_is_not_drain(tmp_path):
    """Regression: Capacity must never be treated as power draw."""
    p = write(tmp_path / "b.txt",
              "Estimated power use (mAh):\n"
              "  Capacity: 5000, Computed drain: 0, actual drain: 0\n")
    assert "error" in parse_batterystats(p)


def test_parse_batterystats_falls_back_to_discharge(tmp_path):
    """Real dumps may have computed drain 0 but a Discharge signal."""
    p = write(tmp_path / "b.txt",
              "Estimated power use (mAh):\n"
              "  Capacity: 4089, Computed drain: 0, actual drain: 0\n"
              "Discharge: 0.409 mAh\n")
    assert parse_batterystats(p) == {"mah": 0.409}


def test_parse_batterystats_zero_drain_unavailable(tmp_path):
    p = write(tmp_path / "b.txt",
              "Estimated power use (mAh):\n"
              "  Capacity: 5000, Computed drain: 0, actual drain: 0\n")
    assert "error" in parse_batterystats(p)


def test_battery_delta_energy():
    r = battery_delta_energy(80, 79, 4500, 3.85)
    assert r["energy_j"] == pytest.approx(0.01 * 4.5 * 3.85 * 3600, rel=0.01)


def test_battery_delta_negative():
    assert "error" in battery_delta_energy(79, 80, 4500, 3.85)


def test_compute_metrics_basic():
    m = compute_metrics(120.0, 100, duration_s=60, throughput_fps=120.0)
    assert m["available"] is True
    assert m["energy_per_inference_j"] == pytest.approx(1.2)
    assert m["avg_power_w"] == pytest.approx(2.0)
    assert m["perf_per_watt"] == pytest.approx(60.0)


def test_compute_metrics_no_duration():
    m = compute_metrics(120.0, 100)
    assert m["perf_per_watt"] is None


def test_compute_metrics_zero_energy():
    assert compute_metrics(0, 100)["available"] is False


def test_cli_perfetto_smoke(tmp_path, capsys):
    """CLI smoke: perfetto source produces metrics and no error."""
    import subprocess
    import sys
    p = tmp_path / "e.json"
    p.write_text('{"energy_uws": 120000000}', encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "scripts/power_analysis.py", "--source", "perfetto",
         "--input", str(p), "--inference-count", "100",
         "--duration-s", "60", "--throughput-fps", "120"],
        capture_output=True, text=True, cwd="/home/liu/project/mobile-bench")
    assert r.returncode == 0
    assert "每推理能耗" in r.stdout


def test_cli_batterystats_capacity_unavailable(tmp_path):
    """CLI smoke: Capacity-only dump must exit nonzero with unavailable."""
    import subprocess
    import sys
    p = tmp_path / "b.txt"
    p.write_text(
        "Estimated power use (mAh):\n"
        "  Capacity: 5000, Computed drain: 0, actual drain: 0\n",
        encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "scripts/power_analysis.py", "--source", "batterystats",
         "--input", str(p), "--inference-count", "100"],
        capture_output=True, text=True, cwd="/home/liu/project/mobile-bench")
    assert r.returncode != 0
    assert "Power data unavailable" in r.stdout or "Power data unavailable" in r.stderr
