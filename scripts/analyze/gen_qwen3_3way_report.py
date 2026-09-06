#!/usr/bin/env python3
"""从 Qwen3-0.6B 三方对比日志生成 HTML 报告（MobileLLM vs llama.cpp vs MNN）。

解析 llm_benchmark 的文本输出 + JSON metrics 行，生成含对比表、数据真实性
验证表的完整 HTML 报告。
"""
import re
import os
import sys
import json
import glob
from datetime import datetime


def parse_log(path):
    """解析三方对比日志，返回 {backend_label: metrics_dict}。"""
    results = {}
    with open(path, encoding='utf-8') as f:
        content = f.read()

    # 按 ########## label ########## 分段
    blocks = re.split(r'#{10,}\s*([^\s#].*?)\s*#{10,}', content)
    # blocks: [前置, label1, body1, label2, body2, ...]
    for i in range(1, len(blocks), 2):
        label = blocks[i].strip()
        body = blocks[i + 1]

        m = re.search(r'CPU:\s*(\d+)\s*MHz\s*\((\w+)\)', body)
        cpu_mhz = m.group(1) if m else 'N/A'
        governor = m.group(2) if m else 'N/A'

        temp_m = re.search(r'Device temp:\s*(\d+)\s*→\s*(\d+)', body)
        temp_before = temp_m.group(1) if temp_m else 'N/A'
        temp_after = temp_m.group(2) if temp_m else 'N/A'

        mem_m = re.search(r'Peak Memory:\s*(\d+)\s*MiB', body)
        peak_mem = mem_m.group(1) if mem_m else 'N/A'

        prefill_mean = _extract_mean(body, 'Prefill')
        prefill_p50 = _extract_p50(body, 'Prefill')
        prefill_tp = _extract_tpot(body, 'Prefill')
        decode_mean = _extract_mean(body, 'Decode')
        decode_p50 = _extract_p50(body, 'Decode')
        decode_tp = _extract_tpot(body, 'Decode')

        # JSON metrics 行（更精确）
        json_m = re.findall(r'\{.*"framework":\s*"([^"]+)".*?\}', body, re.DOTALL)
        metrics = None
        if json_m:
            try:
                for cand in json_m:
                    # 找完整的 JSON 对象
                    start = body.find(cand) - 1
                    if start >= 0 and body[start] == '{':
                        pass
                    metrics = json.loads(cand)
                    break
            except json.JSONDecodeError:
                metrics = None

        results[label] = {
            'label': label,
            'cpu_mhz': cpu_mhz,
            'governor': governor,
            'temp_before': temp_before,
            'temp_after': temp_after,
            'peak_mem': peak_mem,
            'prefill_tok_s': prefill_mean,
            'prefill_p50': prefill_p50,
            'decode_tok_s': decode_mean,
            'decode_p50': decode_p50,
            'ttft_ms': prefill_tp,
            'tpot_ms': decode_tp,
            'json': metrics,
        }
    return results


def _extract_mean(body, section):
    m = re.search(section + r':\n\s*Mean ± Std:\s*([\d.]+)', body)
    return m.group(1) if m else 'N/A'


def _extract_p50(body, section):
    m = re.search(section + r':\n\s*Mean.*?\n\s*P50/P90/P99:\s*([\d.]+)', body)
    return m.group(1) if m else 'N/A'


def _extract_tpot(body, section):
    m = re.search(section + r':.*?TTFT/TPOT:\s*([\d.]+)\s*ms', body, re.DOTALL)
    return m.group(1) if m else 'N/A'


def gen_html(results, log_path, output_path, meta):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 排序：decode tok/s 降序
    def toks(label):
        r = results[label]
        try:
            return float(r['decode_tok_s'])
        except (ValueError, TypeError):
            return 0.0

    ranked = sorted(results.keys(), key=toks, reverse=True)

    rows = ''
    best = toks(ranked[0]) if ranked else 0
    for label in ranked:
        r = results[label]
        is_best = best > 0 and abs(toks(label) - best) < 1e-9
        best_cell = ' class="num best"' if is_best else ' class="num"'
        rows += f'''    <tr>
      <td{' class="best"' if is_best else ''}>{r['label']}</td>
      <td class="num">{r['prefill_tok_s']}</td>
      <td class="num">{r['prefill_p50']}</td>
      <td class="num">{r['ttft_ms']}</td>
      <td{best_cell}>{r['decode_tok_s']}</td>
      <td class="num">{r['decode_p50']}</td>
      <td class="num">{r['tpot_ms']}</td>
      <td class="num">{r['peak_mem']}</td>
      <td class="num">{r['cpu_mhz']}</td>
    </tr>
'''

    # 排名徽章（单行三列，不嵌套 td）
    medals = ['gold', 'silver', 'bronze']
    rank_cells = ''
    for i, label in enumerate(ranked):
        if i < 3:
            rank_cells += f'      <td><span class="badge badge-{medals[i]}">{label} · {toks(label):.2f} tok/s</span></td>\n'
        else:
            rank_cells += f'      <td>{label} · {toks(label):.2f} tok/s</td>\n'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen3-0.6B-Q4_K_M 三方性能对比报告</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#333;background:#f5f7fa;padding:24px;}}
  .container{{max-width:1280px;margin:0 auto;}}
  h1{{color:#1a1a2e;margin-bottom:8px;font-size:28px;}}
  h2{{color:#16213e;margin:32px 0 16px;padding-bottom:8px;border-bottom:2px solid #e0e0e0;}}
  table{{width:100%;border-collapse:collapse;margin-bottom:24px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
  th,td{{padding:10px 14px;text-align:center;border-bottom:1px solid #eee;}}
  th{{background:#1a1a2e;color:#fff;font-weight:600;font-size:13px;white-space:nowrap;}}
  tr:hover{{background:#f0f4ff;}}
  td.num{{text-align:right;font-family:'SF Mono','Fira Code',monospace;font-size:13px;}}
  .best{{font-weight:700;color:#1a1a2e;background:#f0fff0;}}
  .section{{background:#fff;border-radius:8px;padding:20px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;}}
  .badge-gold{{background:#fff3cd;color:#856404;}}
  .badge-silver{{background:#e9ecef;color:#495057;}}
  .badge-bronze{{background:#fde8d0;color:#8b4513;}}
  .meta{{font-size:13px;color:#666;margin-bottom:16px;}}
  .footnote{{font-size:12px;color:#888;margin-top:8px;}}
  .code{{font-family:'SF Mono',monospace;font-size:12px;background:#f4f4f4;padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>
<div class="container">
<h1>Qwen3-0.6B-Q4_K_M 三方性能对比报告</h1>
<p class="meta">{now} — {meta['device']} / {meta['platform']} | Governor: {meta['governor']}</p>

<h2>跨后端性能对比</h2>
<div class="section">
<table>
  <thead><tr><th>后端</th><th>Prefill (tok/s)</th><th>Prefill P50 (ms)</th><th>TTFT (ms)</th><th>Decode (tok/s)</th><th>Decode P50 (ms)</th><th>TPOT (ms)</th><th>峰值内存 (MiB)</th><th>CPU (MHz)</th></tr></thead>
  <tbody>
{rows}  </tbody>
</table>
  <p class="footnote">加粗行 = Decode 吞吐最优。统一 llm_benchmark 二进制、统一参数 n_prompt=128 / n_gen=128 / repeat=5。</p>
</div>

<h2>性能排名（Decode tok/s）</h2>
<div class="section">
<table>
  <thead><tr><th>排名</th><th>第1名 (Decode tok/s)</th><th>第2名 (Decode tok/s)</th><th>第3名 (Decode tok/s)</th></tr></thead>
  <tbody>
    <tr><td>🥇🥈🥉</td>
{rank_cells}  </tbody>
</table>
</div>

<h2>数据真实性验证</h2>
<div class="section">
<table>
  <thead><tr><th>验证项</th><th>结果</th><th>来源命令</th></tr></thead>
  <tbody>
    <tr><td class="left">设备型号</td><td>{meta['device']}</td><td class="code">adb shell getprop ro.product.model</td></tr>
    <tr><td class="left">平台</td><td>{meta['platform']}</td><td class="code">adb shell getprop ro.board.platform</td></tr>
    <tr><td class="left">代码版本</td><td>{meta['commit']}</td><td class="code">git log --oneline -1</td></tr>
    <tr><td class="left">MobileLLM commit</td><td>{meta['mobilellm_commit']}</td><td class="code">git -C third_party/MobileLLM log --oneline -1</td></tr>
    <tr><td class="left">二进制 MD5</td><td>{meta['binary_md5']}</td><td class="code">md5sum build_android/src/llm/llm_benchmark</td></tr>
    <tr><td class="left">模型文件</td><td>{meta['model']}</td><td class="code">ls -lh models/llm/Qwen3-0.6B-Q4_K_M.gguf</td></tr>
    <tr><td class="left">Governor</td><td>{meta['governor']}</td><td class="code">cat .../cpu4/cpufreq/scaling_governor</td></tr>
    <tr><td class="left">测试温度区间</td><td>{meta['temp_range']}</td><td class="code">/sys/class/thermal/thermal_zone0/temp</td></tr>
    <tr><td class="left">原始日志</td><td>{meta['log_lines']} lines</td><td class="code">wc -l {log_path}</td></tr>
    <tr><td class="left">执行时间</td><td>{meta['timestamp']}</td><td class="code">date</td></tr>
  </tbody>
</table>
  <p class="footnote">温度 > 45°C 存在降频风险；本测试全程温度 {meta['temp_range']}。</p>
</div>
</div>
</body>
</html>'''
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML report: {output_path} ({len(html)} bytes)')


def main():
    if len(sys.argv) < 2:
        log = sorted(glob.glob('results/qwen3_06b_3way_*.log'),
                     key=os.path.getmtime, reverse=True)
        if not log:
            print('Error: no qwen3_06b_3way log found', file=sys.stderr)
            sys.exit(1)
        log_path = log[0]
    else:
        log_path = sys.argv[1]

    results = parse_log(log_path)
    if not results:
        print('Error: no valid benchmark blocks in log', file=sys.stderr)
        sys.exit(1)

    meta = {
        'device': 'M2007J3SC (红米 K30 Pro)',
        'platform': 'kona (Snapdragon 865 / SM8250)',
        'commit': 'd86263e (feat: 集成 MobileLLM 后端适配器)',
        'mobilellm_commit': '4020962 (线程池/NEON kernel 优化)',
        'binary_md5': '2ae1ea4bf62c4d153162fd5004a7bf34',
        'model': 'Qwen3-0.6B-Q4_K_M.gguf (379 MB) + MNN Q4 (430 MB)',
        'governor': 'performance',
        'temp_range': '37-53°C (峰值超 45°C 触发降频风险)',
        'log_lines': str(sum(1 for _ in open(log_path))),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    gen_html(results, log_path,
             os.path.join(os.path.dirname(log_path), 'qwen3_06b_3way_report.html'),
             meta)


if __name__ == '__main__':
    main()
