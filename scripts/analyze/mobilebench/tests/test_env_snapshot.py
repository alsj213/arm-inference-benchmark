"""Unit tests for scripts/env_snapshot.py."""
from pathlib import Path

import pytest

import env_snapshot


def write(path, content):
    Path(path).write_text(content, encoding="utf-8")
    return str(path)


def test_run_cmd_missing_command():
    r = env_snapshot.run_cmd(["/nonexistent/definitely-missing", "--x"])
    assert r["value"] is None
    assert "error" in r and r["error"]
    assert r["source"]


def test_run_cmd_success():
    r = env_snapshot.run_cmd(["echo", "hello"])
    assert r["value"] == "hello"
    assert r["error"] is None


def test_load_config_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        env_snapshot.load_config(str(tmp_path / "missing.yml"))


def test_load_config_invalid_yaml(tmp_path):
    p = write(tmp_path / "bad.yml", "device: [unclosed\n")
    with pytest.raises(SystemExit):
        env_snapshot.load_config(p)


def test_load_config_ok(tmp_path):
    p = write(tmp_path / "ok.yml", "device:\n  adb: /x/adb\n")
    cfg = env_snapshot.load_config(p)
    assert cfg["device"]["adb"] == "/x/adb"


def test_collect_device_no_adb(tmp_path):
    r = env_snapshot.collect_device({"device": {}})
    assert "error" in r["adb"]


def test_collect_temperature_adds_unit(tmp_path, monkeypatch):
    calls = []

    def fake_run_cmd(cmd):
        calls.append(cmd)
        if "type" in cmd[-1]:
            return {"value": "cpu-0-usr", "source": " ".join(cmd)}
        return {"value": "35200", "source": " ".join(cmd)}

    monkeypatch.setattr(env_snapshot, "run_cmd", fake_run_cmd)
    r = env_snapshot.collect_temperature(["adb", "-s", "x"])
    assert r["value"] == "35200"
    assert r["unit"] == "millidegree_celsius"


def test_md5_of(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    import hashlib
    assert env_snapshot.md5_of(str(f)) == hashlib.md5(b"abc").hexdigest()


def test_md5_of_missing(tmp_path):
    assert "error" in env_snapshot.md5_of(str(tmp_path / "missing.bin"))


def test_collect_binary_glob(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "bench").write_bytes(b"bin")
    cfg = {"project": {"build_dir": str(tmp_path / "build"), "binary": "bench"}}
    r = env_snapshot.collect_binary(cfg)
    assert r["path"]["value"].endswith("bench")
    assert r["size"]["value"] == 3
    assert r["md5"]["value"]


def test_collect_binary_missing(tmp_path):
    cfg = {"project": {"build_dir": str(tmp_path), "binary": "nope"}}
    assert "error" in env_snapshot.collect_binary(cfg)


def test_collect_models(tmp_path):
    (tmp_path / "models" / "classification" / "m").mkdir(parents=True)
    (tmp_path / "models" / "classification" / "m" / "m.onnx").write_bytes(b"m")
    cfg = {"project": {"models_dir": str(tmp_path / "models")}}
    entries = env_snapshot.collect_models(cfg)
    assert len(entries) == 1
    assert entries[0]["name"].endswith("m.onnx")
    assert entries[0]["md5"]
