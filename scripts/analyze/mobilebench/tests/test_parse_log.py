"""Unit tests for scripts/parse_log.py, including real device --json output."""
import json
from pathlib import Path

from parse_log import normalize_real_json, parse_json_lines, parse_log


def write(path, content):
    Path(path).write_text(content, encoding="utf-8")
    return str(path)


REAL_JSON = json.dumps({
    "accuracy": {"cosine_similarity": 0.0, "max_absolute_error": 0.0,
                 "mean_absolute_error": 0.0, "mean_relative_error": 0.0,
                 "passed": False},
    "framework": "MNN",
    "git_commit": "c684be1",
    "init_time_ms": 109.29651,
    "metrics": {"max_ms": 18.659791, "mean_ms": 18.520847,
                "min_ms": 18.306615, "p50_ms": 18.520781,
                "p90_ms": 18.621406, "p99_ms": 18.659791,
                "std_dev": 0.074831, "throughput_fps": 53.9932},
    "model": "mobilenetv2",
    "peak_memory_kb": 18446744073709551496,
    "precision": "fp32",
    "run_id": "",
    "test_runs": 100,
    "threads": 1,
    "timestamp": "",
    "track": "cnn",
    "warmup_runs": 10,
})


def test_parse_bare_json_line():
    """Real device output: bare JSON object, no JSON_RESULT: prefix."""
    results = parse_json_lines(REAL_JSON)
    assert len(results) == 1
    r = results[0]
    assert r["backend"] == "MNN"
    assert r["model"] == "mobilenetv2"
    assert r["latency"]["p50"] == 18.520781
    assert r["throughput_fps"] == 53.9932
    assert r["accuracy_passed"] is False


def test_parse_json_result_prefix():
    """Legacy documented format still supported."""
    results = parse_json_lines("JSON_RESULT: " + REAL_JSON)
    assert len(results) == 1
    assert results[0]["backend"] == "MNN"


def test_parse_log_real_file(tmp_path):
    p = write(tmp_path / "bench.log",
              "some preamble\n>> Testing mnn on mobilenetv2...\n" + REAL_JSON + "\nDone\n")
    results = parse_log(p)
    assert len(results) == 1
    assert results[0]["model"] == "mobilenetv2"


def test_parse_log_text_fallback(tmp_path):
    p = write(tmp_path / "text.log",
              "Backend: ncnn\nModel: resnet50\nP50: 12.5 ms\nThroughput: 80 FPS\n")
    results = parse_log(p)
    assert len(results) == 1
    assert results[0]["backend"] == "ncnn"
    assert results[0]["latency"]["p50"] == 12.5


def test_normalize_real_json():
    r = normalize_real_json(json.loads(REAL_JSON))
    assert r["latency"]["p90"] == 18.621406
    assert r["init_time_ms"] == 109.29651


def test_parse_empty():
    assert parse_json_lines("") == []
    assert parse_json_lines("no json here\njust text\n") == []
