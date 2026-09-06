#!/usr/bin/env python3
"""
Generate benchmark reports in Markdown and HTML format from log files.
"""
import sys
import os
import re
import glob
import json
from datetime import datetime
import html

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_log import parse_log  # noqa: E402


def _fmt(v, nd=2):
    """None -> 'N/A'(M2:未测量不伪装 0)"""
    return 'N/A' if v is None else f'{float(v):.{nd}f}'


def _cos(v):
    return '-' if v is None else f'{float(v):.6f}'


def _esc_md(v):
    s = str(v) if v is not None else ''
    return s.replace('|', '\\|').replace('\n', ' ')


def _h(v):
    return html.escape(str(v)) if v is not None else ''


def _to_flat(parsed):
    """Flatten a parse_log.py result (nested latency) into the legacy shape.

    Downstream report generators expect flat keys like latency_p50; parse_log
    returns nested {latency: {p50, p90, p99, mean}}. Adapt here so both real
    JSON output (bare {"framework": ...}) and text logs parse identically.
    """
    latency = parsed.get("latency") or {}
    return {
        'backend': parsed.get('backend'),
        'model': parsed.get('model'),
        'precision': parsed.get('precision'),
        'threads': parsed.get('threads'),
        'accuracy_passed': bool(parsed.get('accuracy_passed')),
        'cosine_similarity': parsed.get('cosine_similarity'),
        'latency_p50': latency.get('p50'),
        'latency_p90': latency.get('p90'),
        'latency_p99': latency.get('p99'),
        'latency_mean': latency.get('mean'),
        'throughput_fps': parsed.get('throughput_fps'),
        'memory_kb': parsed.get('memory_kb'),
        'init_time_ms': parsed.get('init_time_ms'),
    }


def parse_log_file(log_path):
    """Parse a benchmark log file for structured results.

    Delegates to parse_log.py so both bare-JSON (real device --json output)
    and text logs are handled by a single parser. Returns a LIST of legacy flat
    records — one per result found in the file (a single log may hold many
    JSON_RESULT lines, e.g. multi-backend 3way runs), each tagged with its
    source file for traceability. Empty/invalid files yield [].
    """
    try:
        parsed_list = parse_log(log_path)
    except Exception as e:
        print(f"Warning: cannot parse file {log_path}: {e}", file=sys.stderr)
        return []
    records = []
    for parsed in parsed_list or []:
        flat = _to_flat(parsed)
        if flat.get('backend') is None:
            continue  # 跳过无有效 backend 的占位记录
        flat['source'] = os.path.basename(log_path)
        records.append(flat)
    return records


def generate_markdown_report(all_results, results_dir):
    """Generate a Markdown report from parsed results."""
    report_path = os.path.join(results_dir, "benchmark_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Benchmark Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Directory: {results_dir}\n\n")
        f.write(f"Total tests: {len(all_results)}\n\n")

        backends = sorted(set(r['backend'] for r in all_results if r['backend']))
        for backend in backends:
            f.write(f"\n## {backend.upper()}\n\n")
            f.write("| Model | Precision | Threads | P50(ms) | P99(ms) | Throughput(FPS) | Accuracy | Cosine Similarity |\n")
            f.write("|------|-----------|---------|---------|---------|-----------------|----------|-------------------|\n")

            backend_results = [r for r in all_results if r['backend'] == backend]
            for r in sorted(backend_results, key=lambda x: (x['model'] or '', x['precision'] or '', x['threads'] or 0)):
                accuracy_str = "PASS" if r['accuracy_passed'] else "FAIL"
                f.write(f"| {_esc_md(r['model'])} | {_esc_md(r['precision'])} | {r['threads']} | "
                       f"{_fmt(r['latency_p50'])} | {_fmt(r['latency_p99'])} | "
                       f"{_fmt(r['throughput_fps'])} | {accuracy_str} | {_cos(r['cosine_similarity'])} |\n")

        # Summary
        f.write("\n## Performance Summary\n\n")
        f.write("| Backend | Avg P50(ms) | Avg P99(ms) | Avg Throughput(FPS) |\n")
        f.write("|---------|-------------|-------------|---------------------|\n")
        for backend in backends:
            backend_results = [r for r in all_results if r['backend'] == backend]
            if not backend_results:
                continue
            def _avg(k):
                vals = [r[k] for r in backend_results if r[k] is not None]
                return sum(vals) / len(vals) if vals else None
            avg_p50, avg_p99, avg_fps = _avg('latency_p50'), _avg('latency_p99'), _avg('throughput_fps')
            f.write(f"| {_esc_md(backend)} | {_fmt(avg_p50)} | {_fmt(avg_p99)} | {_fmt(avg_fps)} |\n")

        # Accuracy
        f.write("\n## Accuracy Verification\n\n")
        passed = sum(1 for r in all_results if r['accuracy_passed'])
        total = len(all_results)
        f.write(f"Pass rate: {passed}/{total} ({passed / total * 100:.1f}%)\n\n")
        if passed < total:
            f.write("### Failed Tests\n\n")
            for r in all_results:
                if not r['accuracy_passed']:
                    f.write(f"- {r['backend']} {r['model']} {r['precision']} {r['threads']} threads "
                           f"(cosine similarity: {_cos(r['cosine_similarity'])})\n")

    print(f"Markdown report: {report_path}")
    return report_path


def generate_html_report(all_results, results_dir):
    """Generate a standalone single-file HTML report with inline CSS."""
    report_path = os.path.join(results_dir, "benchmark_report.html")

    backends = sorted(set(r['backend'] for r in all_results if r['backend']))
    passed = sum(1 for r in all_results if r['accuracy_passed'])
    total = len(all_results)

    # Build table rows per backend
    backend_tables = ""
    for backend in backends:
        backend_results = [r for r in all_results if r['backend'] == backend]
        rows = ""
        for r in sorted(backend_results, key=lambda x: (x['model'] or '', x['precision'] or '', x['threads'] or 0)):
            accuracy_class = "pass" if r['accuracy_passed'] else "fail"
            rows += f"""          <tr>
            <td>{_h(r['model'])}</td>
            <td>{_h(r['precision'])}</td>
            <td>{r['threads']}</td>
            <td class="num">{_fmt(r['latency_p50'])}</td>
            <td class="num">{_fmt(r['latency_p99'])}</td>
            <td class="num">{_fmt(r['throughput_fps'])}</td>
            <td class="{accuracy_class}">{'PASS' if r['accuracy_passed'] else 'FAIL'}</td>
            <td class="num">{_cos(r['cosine_similarity'])}</td>
          </tr>\n"""

        # Summary stats
        def _avg(k):
            vals = [r[k] for r in backend_results if r[k] is not None]
            return sum(vals) / len(vals) if vals else None
        avg_p50, avg_p99, avg_fps = _avg('latency_p50'), _avg('latency_p99'), _avg('throughput_fps')

        backend_tables += f"""
      <h2>{backend.upper()}</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Precision</th>
            <th>Threads</th>
            <th>P50(ms)</th>
            <th>P99(ms)</th>
            <th>Throughput(FPS)</th>
            <th>Accuracy</th>
            <th>Cosine Similarity</th>
          </tr>
        </thead>
        <tbody>
{rows}        </tbody>
        <tfoot>
          <tr class="summary">
            <td colspan="3"><strong>Avg</strong></td>
            <td class="num">{_fmt(avg_p50)}</td>
            <td class="num">{_fmt(avg_p99)}</td>
            <td class="num">{_fmt(avg_fps)}</td>
            <td colspan="2"></td>
          </tr>
        </tfoot>
      </table>"""

    # Failed tests rows
    failed_rows = ""
    for r in all_results:
        if not r['accuracy_passed']:
            failed_rows += f"""          <tr>
            <td>{_h(r['backend'])}</td>
            <td>{_h(r['model'])}</td>
            <td>{_h(r['precision'])}</td>
            <td>{r['threads']} threads</td>
            <td class="num">{_cos(r['cosine_similarity'])}</td>
          </tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         color: #333; background: #f5f7fa; padding: 24px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #1a1a2e; margin-bottom: 8px; }}
  h2 {{ color: #16213e; margin: 24px 0 12px; padding-bottom: 6px; border-bottom: 2px solid #e0e0e0; }}
  .meta {{ color: #666; font-size: 14px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px;
          background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #1a1a2e; color: #fff; font-weight: 600; font-size: 13px; text-transform: uppercase; }}
  tr:hover {{ background: #f0f4ff; }}
  td.num {{ text-align: right; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }}
  .pass {{ color: #155724; background: #d4edda; text-align: center; font-weight: 600; border-radius: 4px; }}
  .fail {{ color: #721c24; background: #f8d7da; text-align: center; font-weight: 600; border-radius: 4px; }}
  .summary {{ background: #eef2f7; font-weight: 600; }}
  .summary td {{ border-bottom: none; }}
  .summary-bar {{ display: flex; gap: 16px; margin: 16px 0; }}
  .summary-card {{ flex: 1; background: #fff; padding: 20px; border-radius: 8px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .summary-card .value {{ font-size: 28px; font-weight: 700; color: #1a1a2e; }}
  .summary-card .label {{ font-size: 12px; color: #888; text-transform: uppercase; margin-top: 4px; }}
  .summary-card.pass-card .value {{ color: #155724; }}
  .summary-card.fail-card .value {{ color: #721c24; }}
</style>
</head>
<body>
<div class="container">
  <h1>Benchmark Report</h1>
  <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &mdash; Directory: {results_dir} &mdash; Tests: {total}</p>

  <div class="summary-bar">
    <div class="summary-card">
      <div class="value">{total}</div>
      <div class="label">Total Tests</div>
    </div>
    <div class="summary-card pass-card">
      <div class="value">{passed}</div>
      <div class="label">Passed</div>
    </div>
    <div class="summary-card fail-card">
      <div class="value">{total - passed}</div>
      <div class="label">Failed</div>
    </div>
    <div class="summary-card">
      <div class="value">{passed / total * 100:.1f}%</div>
      <div class="label">Pass Rate</div>
    </div>
  </div>

{backend_tables}

  <h2>Accuracy Verification</h2>
  <p>Pass rate: {passed}/{total} ({passed / total * 100:.1f}%)</p>
"""

    if passed < total:
        html += f"""
  <h3>Failed Tests</h3>
  <table>
    <thead>
      <tr>
        <th>Backend</th>
        <th>Model</th>
        <th>Precision</th>
        <th>Threads</th>
        <th>Cosine Similarity</th>
      </tr>
    </thead>
    <tbody>
{failed_rows}    </tbody>
  </table>"""

    html += """
</div>
</body>
</html>"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report: {report_path}")
    return report_path


def generate_report(results_dir):
    """Main entry: parse logs, generate both Markdown and HTML reports."""
    log_files = glob.glob(os.path.join(results_dir, "*.log"))
    if not log_files:
        print("Error: no log files found in", results_dir, file=sys.stderr)
        return

    all_results = []
    for f in log_files:
        all_results.extend(parse_log_file(f))

    # M6:过滤无 CNN 实测指标记录(LLM 形状/空),避免全零 FAIL 行
    valid = [r for r in all_results
             if r.get('latency_p50') is not None or r.get('latency_p99') is not None
                or r.get('throughput_fps') is not None]
    if len(valid) != len(all_results):
        print(f"[warn] 跳过 {len(all_results) - len(valid)} 条无 CNN 实测指标记录"
              f"(LLM/空日志)", file=sys.stderr)
    all_results = valid

    if not all_results:
        print("Error: no valid benchmark results found in logs.", file=sys.stderr)
        return

    md_path = generate_markdown_report(all_results, results_dir)
    html_path = generate_html_report(all_results, results_dir)
    print(f"\nReports generated: {md_path}, {html_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: generate_report.py <results_directory>")
        sys.exit(1)

    generate_report(sys.argv[1])
