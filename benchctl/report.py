"""报告生成 — JSON 导出 + HTML 渲染."""
import json
from pathlib import Path
from typing import Optional
from .db import Database

PROJECT_ROOT = Path(__file__).parent.parent


def export_json(frameworks: list[str], model: str,
                output: Optional[Path] = None) -> Path:
    """导出横向对比数据为 JSON."""
    db = Database()
    results = db.compare(frameworks, model)
    db.close()

    if output is None:
        output = PROJECT_ROOT / "results" / f"compare_{model}.json"

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    return output


def generate_html_compare(frameworks: list[str], model: str,
                          output: Optional[Path] = None) -> Path:
    """生成横向对比 HTML 报告，含可视化条形图."""
    db = Database()
    results = db.compare(frameworks, model)
    db.close()

    if output is None:
        output = PROJECT_ROOT / "results" / f"report_{model}.html"
    if not results:
        return output

    # 解析指标
    parsed = []
    for r in results:
        m = json.loads(r["metrics_json"])
        parsed.append({
            "framework": r["framework"].upper(),
            "p50": m.get("p50_ms", 0),
            "p90": m.get("p90_ms", 0),
            "p99": m.get("p99_ms", 0),
            "mean": m.get("mean_ms", 0),
            "fps": m.get("throughput_fps", 0),
            "memory": r.get("peak_memory_kb", 0),
            "precision": r.get("precision", "fp32"),
            "threads": r.get("threads", 4),
            "temp": r.get("device_temp"),
        })

    best_p50 = min(p["p50"] for p in parsed)
    slowest_p50 = max(p["p50"] for p in parsed)

    # 条形图: 以最慢为 100%
    def bar_width(val):
        return max(5, (val / slowest_p50) * 100)

    # 设备信息
    device = results[0].get("device_model", "Unknown")
    timestamp = results[0].get("timestamp", "")

    # 表格行
    rows_html = ""
    for p in parsed:
        is_best = abs(p["p50"] - best_p50) < 0.01
        row_class = ' class="best"' if is_best else ""
        speedup = ""
        if not is_best:
            speedup = f'<span style="color:#999;font-size:0.85em">{best_p50/p["p50"]*100:.0f}%</span>'
        else:
            speedup = '<span style="color:#28a745;font-weight:bold">BASELINE</span>'
        w = bar_width(p["p50"])
        rows_html += f"""
        <tr{row_class}>
            <td style="text-align:left;font-weight:bold">{p['framework']}</td>
            <td>
                <div style="display:flex;align-items:center;gap:8px">
                    <div style="background:{'#28a745' if is_best else '#6c757d'};height:18px;width:{w:.0f}%;border-radius:3px;min-width:20px"></div>
                    <span>{p['p50']:.2f} ms</span>
                </div>
            </td>
            <td>{p['p99']:.2f}</td>
            <td>{p['fps']:.1f}</td>
            <td>{p['memory']}</td>
            <td>{speedup}</td>
        </tr>"""

    temp_str = f" | 🌡 {parsed[0]['temp']}°C" if parsed[0].get("temp") else ""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark: {model}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #f8f9fa; color: #212529; padding: 2em; }}
        .container {{ max-width: 960px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.1);
                padding: 24px; margin-bottom: 20px; }}
        h1 {{ font-size: 1.5em; margin-bottom: 4px; }}
        .meta {{ color: #6c757d; font-size: 0.85em; margin-bottom: 16px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th {{ background: #f1f3f5; text-align: left; padding: 10px 12px;
              font-size: 0.8em; text-transform: uppercase; letter-spacing: .05em; color: #495057; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #e9ecef; }}
        tr:hover {{ background: #f8f9fa; }}
        .best {{ background: #d4edda55; }}
        .bar-container {{ flex: 1; min-width: 120px; background: #e9ecef; border-radius: 3px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 16px; }}
        .kpi {{ background: white; border-radius: 8px; padding: 16px; text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
        .kpi-value {{ font-size: 1.8em; font-weight: 700; color: #212529; }}
        .kpi-label {{ font-size: 0.75em; color: #6c757d; text-transform: uppercase; margin-top: 4px; }}
        .speedup-badge {{ display: inline-block; background: #28a745; color: white;
                         padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }}
        footer {{ text-align: center; color: #adb5bd; font-size: 0.8em; margin-top: 24px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>📊 {model} Benchmark Report</h1>
        <div class="meta">
            🔴 {device} (SM8250){temp_str} · FP32 · {parsed[0]['threads']}线程 · warmup=10 runs=50
            <br><code>{timestamp}</code>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi">
            <div class="kpi-value">{best_p50:.1f}<small style="font-size:0.5em">ms</small></div>
            <div class="kpi-label">🏆 最快 P50</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">{slowest_p50/best_p50:.1f}x</div>
            <div class="kpi-label">⚡ 性能差距</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">{parsed[0]['fps']:.0f}</div>
            <div class="kpi-label">🔥 最高 FPS</div>
        </div>
    </div>

    <div class="card">
        <h2 style="font-size:1.1em;margin-bottom:12px">Latency 对比 (P50)</h2>
        <table>
            <thead>
                <tr><th>框架</th><th>P50 延迟</th><th>P99</th><th>FPS</th><th>内存(KB)</th><th>相对性能</th></tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>

    <footer>
        ARM Inference Benchmark · benchctl · {timestamp[:10] if timestamp else 'N/A'}
    </footer>
</div>
</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(html)
    return output
