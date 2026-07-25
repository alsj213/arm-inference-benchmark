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
    """生成横向对比 HTML 报告."""
    db = Database()
    results = db.compare(frameworks, model)
    db.close()

    if output is None:
        output = PROJECT_ROOT / "results" / f"report_{model}.html"

    # 简单内联 HTML (后续可用 Jinja2 模板)
    best_p50 = min(
        json.loads(r["metrics_json"]).get("p50_ms", float("inf"))
        for r in results
    ) if results else 0

    rows_html = ""
    for r in results:
        m = json.loads(r["metrics_json"])
        p50 = m.get("p50_ms", 0)
        p99 = m.get("p99_ms", 0)
        fps = m.get("throughput_fps", 0)
        row_class = ' class="best"' if (best_p50 > 0 and abs(p50 - best_p50) < 0.01) else ''
        rows_html += f"""
        <tr{row_class}>
            <td>{r['framework']}</td>
            <td>{p50:.2f}</td>
            <td>{p99:.2f}</td>
            <td>{fps:.1f}</td>
            <td>{r.get('precision', 'fp32')}</td>
            <td>{r.get('threads', 4)}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>Benchmark: {model}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
        th {{ background: #f5f5f5; }}
        .best {{ background: #d4edda; font-weight: bold; }}
        .header {{ display: flex; justify-content: space-between; align-items: baseline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Benchmark: {model}</h1>
        <span>生成时间: <code>{results[0]['timestamp'] if results else 'N/A'}</code></span>
    </div>
    <table>
        <thead>
            <tr>
                <th>框架</th><th>P50 (ms)</th><th>P99 (ms)</th><th>FPS</th>
                <th>精度</th><th>线程</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    <p style="color:#666; margin-top:1em;">
        <small>测试设备: 红米 K30 Pro / 骁龙 865 (SM8250)</small>
    </p>
</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(html)
    return output
