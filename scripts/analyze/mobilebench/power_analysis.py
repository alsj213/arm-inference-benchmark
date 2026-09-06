#!/usr/bin/env python3
"""Compute energy and energy-efficiency metrics from power measurement data.

Supports three sources, tried in priority order:
  1. Perfetto power rails (--source perfetto, JSON with energy fields)
  2. dumpsys batterystats (--source batterystats, text dump)
  3. Battery delta estimate (--source battery-delta, pct drop x capacity)

Outputs energy per inference (J) and perf/Watt. When no real measurement is
available it reports "Power data unavailable" and never fabricates a number,
per the benchmark-power spec.

Usage:
    python3 scripts/power_analysis.py --source perfetto --input trace.json \
        --inference-count 100 --duration-s 60 --throughput-fps 114.94
    python3 scripts/power_analysis.py --source battery-delta \
        --start-pct 80 --end-pct 79 --battery-capacity-mah 4500 \
        --inference-count 100 --duration-s 60
"""
import sys
import json
import re
import argparse
from pathlib import Path

DEFAULT_VOLTAGE_V = 3.85
JOULES_PER_WH = 3600.0
MICROJOULES_PER_JOULE = 1e6


def parse_perfetto(input_path):
    """Extract total energy (J) from a perfetto-style JSON.

    Accepts {"energy_j": X}, {"energy_uws": X}, or a power_rails array of
    {"name": ..., "energy_uj": ...}. Returns energy in J or None.
    """
    try:
        data = json.loads(Path(input_path).read_text())
    except (OSError, ValueError) as exc:
        return {"error": f"cannot parse perfetto json: {exc}"}

    if isinstance(data, dict) and "energy_j" in data:
        return {"energy_j": float(data["energy_j"])}
    if isinstance(data, dict) and "energy_uws" in data:
        return {"energy_j": float(data["energy_uws"]) / MICROJOULES_PER_JOULE}
    if isinstance(data, dict) and "power_rails" in data:
        total = 0.0
        for rail in data["power_rails"]:
            if "energy_uj" in rail:
                total += float(rail["energy_uj"]) / MICROJOULES_PER_JOULE
        if total > 0:
            return {"energy_j": total}
    return {"error": "no energy data found (expected energy_j/energy_uws/power_rails)"}


def parse_batterystats(input_path):
    """Extract estimated power (mAh) from a dumpsys batterystats text dump.

    Priority order, to avoid misreading battery Capacity as the drain value:
      1. "Computed drain" / "actual drain" right after the header, if > 0
      2. "Discharge: N mAh" (real battery drain signal)
      3. "Estimated power use (mAh): N" (value on the same line)
      4. "Estimated power use (mAh):" followed by a bare number on the next line

    Capacity is never treated as power draw. A zero drain (e.g. fresh --reset
    with no load) is reported as unavailable, not as 0 J, to avoid implying a
    valid measurement.
    """
    try:
        text = Path(input_path).read_text()
    except OSError as exc:
        return {"error": f"cannot read batterystats dump: {exc}"}

    # 1. Drain values right after the Estimated power use header.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "estimated power use" not in line.lower():
            continue
        window = lines[i:i + 4]
        window_text = " ".join(window)
        drain_matches = re.findall(
            r"(?:computed|actual)\s*drain:\s*([\d.]+)", window_text, re.IGNORECASE)
        if drain_matches:
            drains = [float(v) for v in drain_matches]
            if max(drains) > 0:
                return {"mah": max(drains)}
        break

    # 2. Real discharge signal (falls back when computed drain is 0).
    discharge_match = re.search(r"Discharge:\s*([\d.]+)\s*mAh", text, re.IGNORECASE)
    if discharge_match:
        mah = float(discharge_match.group(1))
        if mah > 0:
            return {"mah": mah}

    # 3. Header value on the same line: "Estimated power use (mAh): 1080.0".
    for i, line in enumerate(lines):
        if "estimated power use" not in line.lower():
            continue
        tokens = line.replace(",", "").split()
        for j, tok in enumerate(tokens):
            if tok in ("(mah)", "mah") or tok.endswith("mah"):
                prev = tokens[j - 1] if j > 0 else None
                if prev:
                    try:
                        return {"mah": float(prev)}
                    except ValueError:
                        pass
        # Bare number directly after "(mAh):" on the same line.
        if "(mah):" in line.lower():
            m = re.search(r"\(mah\):\s*([\d.]+)", line, re.IGNORECASE)
            if m:
                return {"mah": float(m.group(1))}
        # 4. Value alone on the NEXT line (no field labels):
        #    "Estimated power use (mAh):\n 1080.0"
        if i + 1 < len(lines) and re.match(r"^\s*[\d.]+\s*$", lines[i + 1]):
            return {"mah": float(lines[i + 1].strip())}
        break

    return {"error": "no usable power drain found in batterystats dump"}


def battery_delta_energy(start_pct, end_pct, capacity_mah, voltage_v):
    """Energy in J from a battery percentage drop over a test window."""
    delta_ah = (start_pct - end_pct) / 100.0 * capacity_mah / 1000.0
    if delta_ah < 0:
        return {"error": "end_pct must be <= start_pct (charging during test?)"}
    wh = delta_ah * voltage_v
    return {"energy_j": wh * JOULES_PER_WH}


def compute_metrics(energy_j, inference_count, duration_s=None, throughput_fps=None):
    """Derive energy-per-inference and perf/Watt from total energy."""
    if not energy_j or energy_j <= 0:
        return {"available": False, "reason": "no energy data"}
    if not inference_count or inference_count <= 0:
        return {"available": False, "reason": "inference-count must be > 0"}

    metrics = {
        "energy_j": round(energy_j, 4),
        "energy_per_inference_j": round(energy_j / inference_count, 4),
        "inference_count": inference_count,
    }
    if duration_s:
        metrics["avg_power_w"] = round(energy_j / duration_s, 4)
        if throughput_fps:
            metrics["perf_per_watt"] = round(throughput_fps / (energy_j / duration_s), 4)
        else:
            metrics["perf_per_watt"] = None
    else:
        metrics["avg_power_w"] = None
        metrics["perf_per_watt"] = None
    metrics["available"] = True
    return metrics


def render_markdown(metrics):
    if not metrics.get("available"):
        return "Power data unavailable: " + metrics.get("reason", "no data")
    lines = [
        "## 功耗与能效结果",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 总能耗 | {metrics['energy_j']:.2f} J |",
        f"| 每推理能耗 | {metrics['energy_per_inference_j']:.4f} J |",
    ]
    if metrics.get("avg_power_w") is not None:
        lines.append(f"| 平均功率 | {metrics['avg_power_w']:.2f} W |")
    if metrics.get("perf_per_watt") is not None:
        lines.append(f"| 能效 | {metrics['perf_per_watt']:.2f} FPS/W |")
    else:
        lines.append("| 能效 | N/A (需 duration-s 与 throughput-fps) |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["perfetto", "batterystats", "battery-delta"], required=True)
    parser.add_argument("--input", help="input file for perfetto/batterystats sources")
    parser.add_argument("--inference-count", type=int, required=True, help="number of inferences in the test window")
    parser.add_argument("--duration-s", type=float, help="test window duration in seconds")
    parser.add_argument("--throughput-fps", type=float, help="measured throughput (FPS) for perf/Watt")
    # battery-delta only
    parser.add_argument("--start-pct", type=float)
    parser.add_argument("--end-pct", type=float)
    parser.add_argument("--battery-capacity-mah", type=float, help="battery capacity in mAh")
    parser.add_argument("--voltage", type=float, default=DEFAULT_VOLTAGE_V, help=f"nominal voltage (default {DEFAULT_VOLTAGE_V}V)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args()

    if args.source == "perfetto":
        if not args.input:
            print("Error: --input required for --source perfetto", file=sys.stderr)
            sys.exit(1)
        energy = parse_perfetto(args.input)
    elif args.source == "batterystats":
        if not args.input:
            print("Error: --input required for --source batterystats", file=sys.stderr)
            sys.exit(1)
        parsed = parse_batterystats(args.input)
        if "mah" in parsed:
            if parsed["mah"] <= 0:
                energy = {"error": "batterystats drain is 0 (no power recorded since reset)"}
            else:
                energy = {"energy_j": parsed["mah"] / 1000.0 * args.voltage * JOULES_PER_WH}
        else:
            energy = parsed
    else:  # battery-delta
        if None in (args.start_pct, args.end_pct, args.battery_capacity_mah):
            print("Error: --start-pct, --end-pct, --battery-capacity-mah required for battery-delta", file=sys.stderr)
            sys.exit(1)
        energy = battery_delta_energy(args.start_pct, args.end_pct, args.battery_capacity_mah, args.voltage)

    if "error" in energy:
        result = {"available": False, "reason": energy["error"]}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Power data unavailable:", energy["error"])
        sys.exit(1)

    metrics = compute_metrics(energy["energy_j"], args.inference_count, args.duration_s, args.throughput_fps)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(render_markdown(metrics))


if __name__ == "__main__":
    main()
