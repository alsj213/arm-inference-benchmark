#!/usr/bin/env python3
"""Collect a reproducible environment snapshot for a benchmark run.

Reads .benchmarkrc.yml, then gathers device / OS / code / binary / model
metadata so a benchmark result can be reproduced or compared later. Every
field carries its source command (or an explicit error when unavailable),
matching the traceability requirement of the benchmark-methodology spec.

Usage:
    python3 scripts/env_snapshot.py            # write results/env_snapshot_<ts>.json
    python3 scripts/env_snapshot.py --out path.json
"""
import sys
import os
import json
import glob
import hashlib
import argparse
import subprocess
import datetime

try:
    import yaml
except ImportError:
    print("Error: pyyaml required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)

MODEL_EXTENSIONS = (".onnx", ".mnn", ".ort", ".bin", ".param", ".gguf", ".ms", ".so")
CMD_TIMEOUT = 30


def run_cmd(cmd, timeout=CMD_TIMEOUT):
    """Run a shell command; return {value, source, error}."""
    source = " ".join(cmd)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout.strip()
        if proc.returncode != 0:
            return {"value": None, "source": source, "error": proc.stderr.strip() or f"exit {proc.returncode}"}
        return {"value": out, "source": source, "error": None}
    except subprocess.TimeoutExpired:
        return {"value": None, "source": source, "error": f"timeout after {timeout}s"}
    except FileNotFoundError:
        return {"value": None, "source": source, "error": "command not found"}


def load_config(config_path=".benchmarkrc.yml"):
    """Load benchmark configuration, returning a dict or exiting on error."""
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.", file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        print(f"Error: invalid YAML in {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    if "device" not in cfg:
        print("Error: .benchmarkrc.yml missing 'device' section.", file=sys.stderr)
        sys.exit(1)
    return cfg


def collect_device(cfg):
    """Gather device identity, OS, kernel, temperature via adb."""
    adb = cfg.get("device", {}).get("adb")
    dev_id = cfg.get("device", {}).get("id")
    if not adb:
        return {"adb": {"value": None, "source": ".benchmarkrc.yml device.adb", "error": "missing"}}
    base = [adb]
    if dev_id:
        base += ["-s", dev_id]
    result = {
        "device_id": {"value": dev_id or None, "source": ".benchmarkrc.yml device.id"},
        "model": run_cmd(base + ["shell", "getprop", "ro.product.model"]),
        "os_version": run_cmd(base + ["shell", "getprop", "ro.build.version.release"]),
        "os_sdk": run_cmd(base + ["shell", "getprop", "ro.build.version.sdk"]),
        "kernel": run_cmd(base + ["shell", "uname", "-r"]),
        "cpu_cores": run_cmd(base + ["shell", "nproc"]),
        "temp": collect_temperature(base),
    }
    return result


def collect_temperature(base):
    """Read CPU thermal zone temperature in millidegrees Celsius (mC).

    Android thermal_zone temp values are in millidegrees Celsius (e.g. 35200
    == 35.2 C), matching the agent's >45000 mC throttling threshold. The raw
    value is kept for traceability and the unit is annotated on the field.
    """
    for z in range(8):
        zone_type = run_cmd(base + ["shell", "cat", f"/sys/class/thermal/thermal_zone{z}/type"])
        if zone_type.get("value") and "cpu" in zone_type.get("value", "").lower():
            raw = run_cmd(base + ["shell", "cat", f"/sys/class/thermal/thermal_zone{z}/temp"])
            if raw.get("value") is not None:
                raw["unit"] = "millidegree_celsius"
            return raw
    return {"value": None, "source": "thermal_zone*/type", "error": "no CPU thermal zone found"}


def collect_code():
    """Git commit / branch / framework submodule versions."""
    result = {
        "commit": run_cmd(["git", "log", "--oneline", "-1"]),
        "branch": run_cmd(["git", "branch", "--show-current"]),
    }
    sub_status = run_cmd(["git", "submodule", "status"])
    frameworks = []
    if sub_status.get("value"):
        for line in sub_status["value"].splitlines():
            parts = line.split()
            if len(parts) >= 2:
                frameworks.append({
                    "name": parts[1],
                    "commit": parts[0].lstrip("-+"),
                    "dirty": parts[0].startswith("+"),
                    "source": "git submodule status",
                })
    result["frameworks"] = {"value": frameworks, "source": "git submodule status"}
    return result


def md5_of(path):
    """Compute md5 hex digest of a file, streaming to bound memory."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        return {"error": str(exc)}


def collect_binary(cfg):
    """Binary path, size and md5 from build_dir + binary name."""
    proj = cfg.get("project", {})
    build_dir = proj.get("build_dir")
    binary = proj.get("binary")
    if not build_dir or not binary:
        return {"error": ".benchmarkrc.yml project.build_dir/binary missing"}
    candidates = glob.glob(os.path.join(build_dir, "**", binary), recursive=True)
    if not candidates:
        return {"error": f"binary not found under {build_dir}/"}
    # M7 修复:多候选(不同 ABI/构建类型并存)按最新修改时间取,避免字典序误选
    path = max(candidates, key=os.path.getmtime)
    size = os.path.getsize(path)
    digest = md5_of(path)
    return {
        "path": {"value": path, "source": "glob build_dir/**/" + binary},
        "size": {"value": size, "source": f"stat {path}"},
        "md5": {"value": digest if isinstance(digest, str) else None,
                "source": f"md5sum {path}",
                "error": digest.get("error") if not isinstance(digest, str) else None},
    }


def collect_models(cfg):
    """List model files with size and md5 from models_dir."""
    models_dir = cfg.get("project", {}).get("models_dir")
    if not models_dir or not os.path.isdir(models_dir):
        return [{"error": "models_dir not found or missing in config"}]
    entries = []
    for f in sorted(glob.glob(os.path.join(models_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(f) or not f.endswith(MODEL_EXTENSIONS):
            continue
        rel = os.path.relpath(f, models_dir)
        digest = md5_of(f)
        entries.append({
            "name": rel,
            "size": os.path.getsize(f),
            "md5": digest if isinstance(digest, str) else None,
            "error": digest.get("error") if not isinstance(digest, str) else None,
            "source": f"md5sum {f}",
        })
    return entries


def build_snapshot(cfg):
    """Assemble the full snapshot dict."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": {"value": now, "source": "date"},
        "device": collect_device(cfg),
        "code": collect_code(),
        "binary": collect_binary(cfg),
        "models": collect_models(cfg),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=".benchmarkrc.yml", help="path to .benchmarkrc.yml")
    parser.add_argument("--out", default=None, help="output JSON path (default results/env_snapshot_<ts>.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    snapshot = build_snapshot(cfg)

    out_path = args.out
    if not out_path:
        results_dir = cfg.get("project", {}).get("results_dir") or "results"
        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(results_dir, f"env_snapshot_{ts}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"Env snapshot: {out_path}")
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
