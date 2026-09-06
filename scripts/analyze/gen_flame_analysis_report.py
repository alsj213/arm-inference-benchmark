#!/usr/bin/env python3
"""生成 MobileLLM vs llama.cpp 火焰图分析报告（HTML，内联双火焰图 SVG）。

从 simpleperf 采样数据 + 源码分析生成可交互性能分析报告。
"""
import os

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'profiling',
                   'mobilellm_qwen3_06b', 'analysis_report.html')
SVG_M = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'profiling',
                     'mobilellm_qwen3_06b', '_svg_mobilellm.txt')
SVG_L = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'profiling',
                     'mobilellm_qwen3_06b', '_svg_llamacpp.txt')

svg_mobilellm = open(SVG_M).read().strip()
svg_llamacpp = open(SVG_L).read().strip()

html = '''<title>MobileLLM 性能瓶颈分析 — Qwen3-0.6B decode</title>
<style>
  :root {
    --bg: #f5f7fa; --card: #ffffff; --ink: #333; --ink-strong: #1a1a2e;
    --accent: #2563eb; --border: #e5e7eb; --good: #155724; --bad: #721c24;
    --good-bg: #d4edda; --bad-bg: #f8d7da; --warn: #856404; --warn-bg: #fff3cd;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--ink); padding: 24px; }
  .container { max-width: 1280px; margin: 0 auto; }
  h1 { color: var(--ink-strong); font-size: 26px; margin-bottom: 6px; }
  .meta { font-size: 13px; color: #666; margin-bottom: 20px; }
  h2 { color: var(--ink-strong); margin: 28px 0 12px; padding-bottom: 8px;
       border-bottom: 2px solid var(--border); font-size: 20px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0 20px; background: var(--card);
          border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th, td { padding: 10px 14px; text-align: center; border-bottom: 1px solid var(--border); }
  th { background: var(--ink-strong); color: #fff; font-weight: 600; font-size: 13px; }
  tr:hover { background: #f0f4ff; }
  td.num { text-align: right; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }
  td.left { text-align: left; }
  .bad { color: var(--bad); font-weight: 700; }
  .good { color: var(--good); font-weight: 700; }
  .tag { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .tag-bad { background: var(--bad-bg); color: var(--bad); }
  .tag-good { background: var(--good-bg); color: var(--good); }
  .tag-warn { background: var(--warn-bg); color: var(--warn); }
  .card { background: var(--card); border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .card h3 { margin-bottom: 8px; color: var(--ink-strong); }
  .flame-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 6px;
                background: #fff; padding: 8px; }
  .flame-wrap svg { min-width: 900px; display: block; }
  .code { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; background: #f4f4f4;
          padding: 1px 5px; border-radius: 3px; }
  ul, ol { padding-left: 22px; margin: 8px 0; line-height: 1.7; }
  .footnote { font-size: 12px; color: #888; margin-top: 8px; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #111827; --card: #1f2937; --ink: #e5e7eb; --ink-strong: #f3f4f6;
      --border: #374151; --good: #86efac; --bad: #fca5a5;
      --good-bg: #14532d; --bad-bg: #7f1d1d; --warn: #fcd34d; --warn-bg: #78350f;
    }
  }
  :root[data-theme="dark"] {
    --bg: #111827; --card: #1f2937; --ink: #e5e7eb; --ink-strong: #f3f4f6;
    --border: #374151; --good: #86efac; --bad: #fca5a5;
    --good-bg: #14532d; --bad-bg: #7f1d1d; --warn: #fcd34d; --warn-bg: #78350f;
  }
</style>
<div class="container">
<h1>MobileLLM 性能瓶颈分析</h1>
<p class="meta">Qwen3-0.6B-Q4_K_M decode · 设备: M2007J3SC (kona / 骁龙865) · 锁频 performance · simpleperf CPU 采样 (4kHz, call-graph fp)</p>

<div class="card">
  <h3>性能差距概览 <span class="tag tag-bad">MobileLLM 慢 3.04x</span></h3>
  <table>
    <thead><tr><th>后端</th><th>Decode (tok/s)</th><th>TPOT (ms)</th><th>Q4_K+Q6_K 实际计算占比</th><th>线程同步/管理占比</th></tr></thead>
    <tbody>
      <tr><td>MNN LLM</td><td class="num">41.27</td><td class="num">23.9</td><td class="num">-</td><td class="num">-</td></tr>
      <tr><td>llama.cpp</td><td class="num good">35.54</td><td class="num">28.1</td><td class="num good">82.1%</td><td class="num good">3.3%</td></tr>
      <tr><td>MobileLLM</td><td class="num bad">11.68</td><td class="num bad">85.6</td><td class="num bad">36.8%</td><td class="num bad">~56%</td></tr>
    </tbody>
  </table>
  <p class="footnote">计算占比 = 采样落在 Q4_K/Q6_K 向量点积的时间比例；线程占比 = worker/thread_pool_submit 采样比例。MobileLLM 约 56% 的 CPU 时间耗在线程管理而非计算。</p>
</div>

<h2>火焰图对比（点击放大查看调用链）</h2>
<div class="card">
  <h3>MobileLLM — 热点: <span class="code">mblm_thread_pool_submit</span> (45%)</h3>
  <div class="flame-wrap">''' + svg_mobilellm + '''</div>
  <p class="footnote">红色/暖色 = 高开销。注意巨大的 <span class="code">_worker_thread</span> 宽条（自旋空转）和 <span class="code">mblm_thread_pool_submit</span> → <span class="code">_partition_rows</span> → <span class="code">calloc/free</span> 调用链。</p>
</div>
<div class="card">
  <h3>llama.cpp — 热点: <span class="code">ggml_vec_dot_q4_K_q8_K</span> (49%, 4线程均分)</h3>
  <div class="flame-wrap">''' + svg_llamacpp + '''</div>
  <p class="footnote">宽条集中在实际的量化点积 kernel（真正的计算），线程同步仅 <span class="code">kmp_flag_64::wait</span> 3.3%（OpenMP 空闲休眠）。</p>
</div>

<h2>热点函数对比</h2>
<div class="card">
  <table>
    <thead><tr><th>MobileLLM 热点</th><th>占比</th><th>llama.cpp 热点</th><th>占比</th></tr></thead>
    <tbody>
      <tr><td class="left code">_worker_thread (×3)</td><td class="num bad">45.1%</td><td class="left code">ggml_vec_dot_q4_K_q8_K (×4)</td><td class="num good">48.9%</td></tr>
      <tr><td class="left code">mblm_thread_pool_submit</td><td class="num bad">10.9%</td><td class="left code">ggml_vec_dot_q6_K_q8_K (×4)</td><td class="num good">33.1%</td></tr>
      <tr><td class="left code">_vec_dot_q4_K_q8_K</td><td class="num">17.1%</td><td class="left code">ggml_compute_forward_flash_attn_ext</td><td class="num">2.8%</td></tr>
      <tr><td class="left code">_vec_dot_q6_K_q8_K</td><td class="num">19.7%</td><td class="left code">kmp_flag_64::wait (OpenMP)</td><td class="num good">3.3%</td></tr>
      <tr><td class="left code">mblm_kernel_flash_attn</td><td class="num">3.4%</td><td class="left code">ggml_vec_dot_f16</td><td class="num">0.6%</td></tr>
    </tbody>
  </table>
</div>

<h2>根因分析（调用链 + 源码证据）</h2>
<div class="card">
  <h3>调用链：GEMM 是主导，但线程池吃掉了近一半</h3>
  <pre class="code">mblm_decode_internal (84%)
 └─ mblm_graph_compute (98%)
     └─ mblm_matmul_q4k (82.5%)          ← GEMM 占 decode 绝大多数时间
         ├─ mblm_thread_pool_submit (45%) ← 线程池提交！
         │   └─ _partition_rows
         │       ├─ calloc (33%)          ← 每次提交分配 weights 数组
         │       └─ je_free (17%)         ← 每次释放
         └─ _vec_dot_q4_K_q8_K (28.5%)    ← 实际 Q4_K 计算</pre>
</div>
<div class="card">
  <h3>源码证据（third_party/MobileLLM）</h3>
  <ol>
    <li><b>worker 纯自旋空转</b> — <span class="code">src/engine/mblm_thread.c:83-85</span>：
      <pre class="code">while (__atomic_load_n(&pool->work_gen, ...) == my_gen) {
    _cpu_relax();   // 无任务也持续空转 CPU，永不 sleep
}</pre>
      3 个 worker 线程空闲时也 100% 占核 → 采样 <span class="code">_worker_thread</span> 45% 大部分是空转。</li>
    <li><b>每次 submit 分配内存</b> — <span class="code">mblm_thread.c:216</span>：
      <pre class="code">float *weights = (float *)calloc(pool->n_threads, sizeof(float));
...
free(weights);   // 每次 matmul 提交触发 calloc + free</pre></li>
    <li><b>主线程 spin 等待</b> — <span class="code">mblm_thread.c:284</span>：
      <pre class="code">while (__atomic_load_n(&pool->n_busy, ...) > 0) { _cpu_relax(); }</pre></li>
    <li><b>decode 单 token 仍开线程池</b> — <span class="code">mblm_matmul_q4k.c:813</span> 每次 QK 都 <span class="code">mblm_thread_pool_submit</span>，M=1 的 GEMM 并行收益 < 同步开销。</li>
  </ol>
</div>

<h2>优化建议</h2>
<div class="card">
  <ol>
    <li><span class="tag tag-warn">高优先级</span> <b>worker 空闲休眠</b>：把 <span class="code">_cpu_relax()</span> 自旋改为 <span class="code">pthread_cond_wait</span>/semaphore 阻塞，或加条件分支在无任务时 <span class="code">sched_yield</span>/休眠。可回收约 40%+ CPU（对应 decode 显著提速）。</li>
    <li><span class="tag tag-warn">高优先级</span> <b>消除每次 calloc</b>：<span class="code">_partition_rows</span> 的 <span class="code">weights</span> 数组改为线程池内复用缓冲（或整数权重计算），decode 路径零堆分配。</li>
    <li><span class="tag">中优先级</span> <b>M=1 解码避免线程池</b>：单 token decode 时 GEMM 规模小（M=1），串行 + NEON 比 4 线程 + 同步更快。在 <span class="code">mblm_matmul_q4k</span> 对 M≤阈值走串行路径。</li>
    <li><span class="tag">中优先级</span> <b>优化 Q4_K NEON kernel</b>：参考 llama.cpp <span class="code">ggml_vec_dot_q4_K</span> 的 16-way unroll + <span class="code">vdotq_s32</span> 实现，MobileLLM 的 <span class="code">_vec_dot_q4_K_q8_K</span> 单核占比仍低于 llama.cpp 对应 kernel。</li>
    <li><span class="tag">参考</span> <b>线程策略</b>：llama.cpp 用持久化 OpenMP 线程 + barrier，空闲休眠；MNN 用底层 MatMul 深度优化。MobileLLM 可对齐此调度模型。</li>
  </ol>
</div>
</div>'''

out = os.path.abspath(OUT)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'分析报告: {out} ({len(html)} bytes)')
