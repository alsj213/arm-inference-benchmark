#include "mobilellm_backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "../common/utils.h"

// MobileLLM 纯 C ABI（头文件自带 extern "C" 保护）
#include "mblm/mblm.h"

// ── 生命周期 ──
MobileLlmBackend::~MobileLlmBackend() { deinit(); }

bool MobileLlmBackend::init(const BenchmarkConfig& config) {
  (void)config;
  if (mblm_backend_init() != MBLM_OK) {
    printf("MobileLLM: backend_init failed\n");
    return false;
  }
  return true;
}

void MobileLlmBackend::deinit() {
  if (ctx_) {
    mblm_context_free((mblm_context_t*)ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    mblm_model_free((mblm_model_t*)model_);
    model_ = nullptr;
  }
  mblm_backend_free();
  model_loaded_ = false;
}

bool MobileLlmBackend::load_model(const std::string& model_path,
                                  int n_ctx, int n_batch) {
  // run_mobilellm 仅调用 load_model（不调用 init），在此确保后端已初始化
  if (mblm_backend_init() != MBLM_OK) {
    printf("MobileLLM: backend_init failed\n");
    return false;
  }
  mblm_model_params_t mp = mblm_model_params_default();
  mp.model_path = model_path.c_str();
  mp.n_ctx      = n_ctx;
  mp.n_batch    = n_batch;
  mp.n_threads  = n_threads_;
  mp.backend    = MBLM_BACKEND_CPU;

  mblm_model_t* m = mblm_model_load(mp);
  if (!m) {
    printf("MobileLLM: model load failed: %s\n", model_path.c_str());
    return false;
  }
  model_ = m;
  model_loaded_ = true;

  // 预建一个 context（后续 benchmark 每次重复重建，避免 KV cache 累积）
  mblm_context_params_t cp = mblm_context_params_default();
  cp.n_ctx     = n_ctx;
  cp.n_batch   = n_batch;
  cp.n_threads = n_threads_;
  ctx_ = mblm_context_create(m, cp);
  if (!ctx_) {
    printf("MobileLLM: context create failed\n");
    return false;
  }
  return true;
}

// ── 交互式生成 ──
std::string MobileLlmBackend::generate(const std::string& prompt,
                                       int max_tokens) {
  if (!model_) return "";
  mblm_context_params_t cp = mblm_context_params_default();
  cp.n_threads = n_threads_;
  mblm_sampler_params_t sp = mblm_sampler_params_default();
  sp.temp = 0.7f;
  int32_t n_gen = 0;
  char* out = mblm_generate((mblm_model_t*)model_, prompt.c_str(),
                            cp, sp, max_tokens, &n_gen);
  if (!out) return "";
  std::string result(out);
  free(out);
  return result;
}

// ── Benchmark（对齐 mblm_benchmark.c measure_once 语义）──
MobileLlmBackend::LlmBenchResult MobileLlmBackend::benchmark(
    int n_prompt, int n_generate, int n_repeat) {
  LlmBenchResult result = {};
  if (!model_loaded_ || !model_) return result;
  result.n_prompt = n_prompt;
  result.n_generate = n_generate;

  size_t mem_before = utils::get_memory_usage_kb() / 1024;  // MiB
  result.peak_memory_mib = mem_before;

  const mblm_token_t kTok = 1;  // 固定 token，仅测吞吐（同 mblm_benchmark）

  for (int r = 0; r < n_repeat; r++) {
    // 每次重复新建 context：与 mblm_benchmark 一致（KV cache 不跨重复累积）
    mblm_context_params_t cp = mblm_context_params_default();
    cp.n_ctx     = n_prompt + n_generate + 64;
    cp.n_batch   = 512;
    cp.n_threads = n_threads_;
    mblm_context_t* ctx = mblm_context_create((mblm_model_t*)model_, cp);
    if (!ctx) continue;

    std::vector<mblm_token_t> prompt_tokens(n_prompt, kTok);

    // ── Prefill ──
    auto t0c = std::chrono::high_resolution_clock::now();
    int rc = mblm_decode(ctx, prompt_tokens.data(), n_prompt, 0);
    auto t1c = std::chrono::high_resolution_clock::now();
    if (rc != MBLM_OK) {
      mblm_context_free(ctx);
      continue;
    }
    double prefill_s =
        std::chrono::duration<double>(t1c - t0c).count();
    result.prefill_per_iter.push_back(
        prefill_s > 0 ? (double)n_prompt / prefill_s : 0.0);
    result.prefill_ms.push_back(prefill_s * 1000.0);

    // ── Decode ──
    mblm_token_t tok = kTok;
    auto t2c = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_generate; i++) {
      mblm_decode(ctx, &tok, 1, n_prompt + i);
      tok = kTok;
    }
    auto t3c = std::chrono::high_resolution_clock::now();
    double decode_s =
        std::chrono::duration<double>(t3c - t2c).count();
    result.decode_per_iter.push_back(
        decode_s > 0 ? (double)n_generate / decode_s : 0.0);
    result.decode_ms.push_back(decode_s * 1000.0);

    mblm_context_free(ctx);

    // 采样峰值 RSS
    size_t cur = utils::get_memory_usage_kb() / 1024;
    if (cur > result.peak_memory_mib) result.peak_memory_mib = cur;
  }

  if (!result.prefill_per_iter.empty()) {
    double sum = 0;
    for (double v : result.prefill_per_iter) sum += v;
    result.prefill_tok_per_s = sum / result.prefill_per_iter.size();
    result.ttft_ms = result.prefill_ms[0];  // 首次 prefill 延迟 ≈ TTFT
  }
  if (!result.decode_per_iter.empty()) {
    double sum = 0;
    for (double v : result.decode_per_iter) sum += v;
    result.decode_tok_per_s = sum / result.decode_per_iter.size();
    result.tpot_ms = 1000.0 / (result.decode_tok_per_s > 0
                                   ? result.decode_tok_per_s : 1e-9);
  }
  return result;
}
