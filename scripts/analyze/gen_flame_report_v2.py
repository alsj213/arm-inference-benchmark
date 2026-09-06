#!/usr/bin/env python3
"""生成 MobileLLM 优化前后对比分析报告（11e10d6 vs 4020962）。

内联旧版/新版火焰图，对比线程池优化效果与剩余瓶颈。
"""
import os

BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'profiling',
                    'mobilellm_qwen3_06b')
OUT = os.path.join(BASE, 'analysis_report_v2.html')

svg_old = open(os.path.join(BASE, '_svg_mobilellm.txt')).read().strip()
svg_new = open(os.path.join(BASE, '_svg_mobilellm_v2.txt')).read().strip()

html = '''<title>MobileLLM 优化前后性能对比 — Qwen3-0.6B decode (11e10d6 → 4020962)</title>
<style>
  :root {
    --bg: #f5f7fa; --card: #ffffff; --ink: #333; --ink-strong: #1a1a2e;
    --border: #e5e7eb; --good: #155724; --bad: #721c24;
    --good-bg: #d4edda; --bad-bg: #f8d7da; --warn: #856404; --warn-bg: #fff3cd;
    --up: #155724; --up-bg: #d4edda; --down: #721c24; --down-bg: #f8d7da;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--ink); padding: 24px; }
  .container { max-width: 1280px; margin: 0 auto; }
  h1 { color: var(--ink-strong); font-size: 24px; margin-bottom: 6px; }
  .meta { font-size: 13px; color: #666; margin-bottom: 20px; }
  h2 { color: var(--ink-strong); margin: 28px 0 12px; padding-bottom: 8px;
       border-bottom: 2px solid var(--border); font-size: 19px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0 20px; background: var(--card);
          border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th, td { padding: 9px 12px; text-align: center; border-bottom: 1px solid var(--border); }
  th { background: var(--ink-strong); color: #fff; font-weight: 600; font-size: 13px; }
  tr:hover { background: #f0f4ff; }
  td.num { text-align: right; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }
  td.left { text-align: left; }
  .up { color: var(--up); font-weight: 700; }
  .down { color: var(--down); font-weight: 700; }
  .tag { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .tag-up { background: var(--up-bg); color: var(--up); }
  .tag-warn { background: var(--warn-bg); color: var(--warn); }
  .tag-down { background: var(--down-bg); color: var(--down); }
  .card { background: var(--card); border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .card h3 { margin-bottom: 8px; color: var(--ink-strong); }
  .flame-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 6px;
                background: #fff; padding: 8px; }
  .flame-wrap svg { min-width: 900px; display: block; }
  .code { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; background: #f4f4f4;
          padding: 1px 5px; border-radius: 3px; }
  ol { padding-left: 22px; margin: 8px 0; line-height: 1.7; }
  .footnote { font-size: 12px; color: #888; margin-top: 8px; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #111827; --card: #1f2937; --ink: #e5e7eb; --ink-strong: #f3f4f6;
      --border: #374151; --up: #86efac; --down: #fca5a5;
      --up-bg: #14532d; --down-bg: #7f1d1d; --warn: #fcd34d; --warn-bg: #78350f;
    }
  }
  :root[data-theme="dark"] {
    --bg: #111827; --card: #1f2937; --ink: #e5e7eb; --ink-strong: #f3f4f6;
    --border: #374151; --up: #86efac; --down: #fca5a5;
    --up-bg: #14532d; --down-bg: #7f1d1d; --warn: #fcd34d; --warn-bg: #78350f;
  }
</style>
<div class="container">
<h1>MobileLLM 优化前后性能对比</h1>
<p class="meta">Qwen3-0.6B-Q4_K_M decode · 设备: M2007J3SC (骁龙865) · 锁频 performance · simpleperf 采样 4kHz</p>

<div class="card">
  <h3>吞吐对比 <span class="tag tag-up">decode +6.3%</span></h3>
  <table>
    <thead><tr><th>指标</th><th>旧版 11e10d6</th><th>新版 4020962</th><th>变化</th></tr></thead>
    <tbody>
      <tr><td>Decode (tok/s)</td><td class="num">11.68</td><td class="num up">12.42</td><td class="up">+6.3%</td></tr>
      <tr><td>Prefill (tok/s)</td><td class="num">17.13</td><td class="num">17.21</td><td>+0.5%</td></tr>
      <tr><td>TPOT (ms)</td><td class="num">85.6</td><td class="num up">80.5</td><td class="up">-6.0%</td></tr>
      <tr><td>峰值内存 (MiB)</td><td class="num">427</td><td class="num up">445</td><td>基本持平</td></tr>
    </tbody>
  </table>
</div>

<div class="card">
  <h3>热点开销对比 — 线程池优化显著生效</h3>
  <table>
    <thead><tr><th>热点</th><th>旧版 11e10d6</th><th>新版 4020962</th><th>变化</th></tr></thead>
    <tbody>
      <tr><td class="left code">_worker_thread (×3)</td><td class="num down">45.1%</td><td class="num up">0.43%</td><td class="up">✅ -98.9%</td></tr>
      <tr><td class="left code">mblm_thread_pool_submit</td><td class="num down">10.9%</td><td class="num up">1.25% (submit_main)</td><td class="up">✅ -88%</td></tr>
      <tr><td class="left code">_partition_rows (calloc)</td><td class="num down">含堆分配</td><td class="num up">0.01%</td><td class="up">✅ 栈数组生效</td></tr>
      <tr><td class="left code">_vec_dot_q4_K_q8_K (计算)</td><td class="num">17.1%</td><td class="num">21.6%</td><td>↑ 计算占比提升</td></tr>
      <tr><td class="left code">mblm_kernel_flash_attn</td><td class="num">3.4%</td><td class="num down">18.2%</td><td class="down">⚠️ 新瓶颈</td></tr>
    </tbody>
  </table>
  <p class="footnote">优化提交：<span class="code">ca4d6ad</span> worker 空闲自适应退避 · <span class="code">72c07d5</span> _partition_rows 栈数组 + submit_main · <span class="code">4020962</span> Q6_K NEON 寄存器优化</p>
</div>

<h2>火焰图对比</h2>
<div class="card">
  <h3>新版 4020962 — 线程池开销消失，计算主导</h3>
  <div class="flame-wrap">''' + svg_new + '''</div>
  <p class="footnote">注意 <span class="code">_worker_thread</span> 宽条已消失；<span class="code">_vec_dot_q4_K_q8_K</span> 和 <span class="code">mblm_kernel_flash_attn</span> 成为主导。</p>
</div>
<div class="card">
  <h3>旧版 11e10d6 — 巨大 worker 自旋空转条</h3>
  <div class="flame-wrap">''' + svg_old + '''</div>
</div>

<div class="card">
  <h3>调用链对比</h3>
  <table>
    <thead><tr><th>路径</th><th>旧版 11e10d6</th><th>新版 4020962</th></tr></thead>
    <tbody>
      <tr><td class="left">mblm_matmul_q4k 内主线程直接计算</td><td class="num">3.8%</td><td class="num up">27.9%</td></tr>
      <tr><td class="left">mblm_matmul_q4k → thread_pool_submit</td><td class="num down">45.1%</td><td class="num up">35.6% (含 QK 实际计算)</td></tr>
      <tr><td class="left">mblm_kernel_flash_attn</td><td class="num">3.4%</td><td class="num">18.2%</td></tr>
    </tbody>
  </table>
</div>

<h2>结论：剩余瓶颈在 kernel 计算效率</h2>
<div class="card">
  <ol>
    <li><span class="tag tag-up">已解决</span> <b>线程池自旋空转</b>（45%→0.4%）与 <b>每次 calloc</b>（→栈数组）。优化提交 <span class="code">ca4d6ad</span>/<span class="code">72c07d5</span> 精确命中上次分析结论。</li>
    <li><span class="tag tag-warn">剩余瓶颈</span> <b>decode 仅 +6.3%</b>，说明线程池并非 decode 慢的全部。优化后计算成为主导，MobileLLM 的 <span class="code">_vec_dot_q4_K_q8_K</span> 单核 NEON 效率仍低于 llama.cpp 的 <span class="code">ggml_vec_dot_q4_K</span>（16-way unroll + <span class="code">vdotq_s32</span>）。</li>
    <li><span class="tag tag-warn">新暴露瓶颈</span> <b><span class="code">mblm_kernel_flash_attn</span> 占 18.2%</b>（llama.cpp 的 flash_attn_ext 仅 2.8%）。MobileLLM 的 Flash Attention 实现效率低，是下一个优化重点。</li>
    <li><span class="tag">建议</span> 优化方向：① Q4_K/Q6_K NEON dot kernel 对齐 llama.cpp；② flash_attn 向量化/分块优化；③ 对比 MNN (decode 39 tok/s) 的 kernel 策略。</li>
  </ol>
</div>
</div>'''

out = os.path.abspath(OUT)
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'对比分析报告: {out} ({len(html)} bytes)')
