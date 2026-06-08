#include "mnn_llm_backend.h"

#ifdef BENCHMARK_MNN
#include "llm/llm.hpp"
#include <MNN/AutoTime.hpp>
#include <chrono>
#include <sstream>
#include <cstdlib>

// ── init / load ──

bool MnnLlmBackend::init(const BenchmarkConfig& config) {
    if (config.model_path.empty()) return true;  // deferred loading
    return load_model(config.model_path);
}

bool MnnLlmBackend::load_model(const std::string& config_path) {
    printf("MNN_LLM: loading config: %s\n", config_path.c_str());
    auto start = std::chrono::high_resolution_clock::now();

    llm_ = MNN::Transformer::Llm::createLLM(config_path);
    if (!llm_) {
        printf("MNN_LLM: createLLM failed\n");
        return false;
    }

    bool ok = llm_->load();
    auto end = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration<double>(end - start).count();

    if (!ok) {
        printf("MNN_LLM: load failed\n");
        return false;
    }

    printf("MNN_LLM: loaded in %.2f s\n", t);
    model_loaded_ = true;
    return true;
}

// ── generate (text in → text out) ──

std::string MnnLlmBackend::generate(const std::string& prompt, int max_tokens) {
    if (!model_loaded_) return "";

    std::ostringstream oss;
    llm_->response(prompt, &oss, nullptr, max_tokens);
    return oss.str();
}

void MnnLlmBackend::reset() {
    if (llm_) llm_->reset();
}

// ── benchmark ──
// Uses Llm::getContext()->prefill_us / decode_us — same hooks as official llm_bench

MnnLlmBackend::LlmBenchResult MnnLlmBackend::benchmark(
        int n_prompt, int n_generate, int n_repeat) {

    LlmBenchResult result = {};
    if (!model_loaded_) return result;

    result.n_prompt = n_prompt;
    result.n_generate = n_generate;

    std::vector<int> prompt_ids(n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        prompt_ids[i] = (rand() % 10000) + 100;
    }

    std::vector<double> prefill_speeds, decode_speeds;

    for (int r = 0; r < n_repeat + 1; r++) {  // +1 warmup (skip first, like official llm_bench)
        llm_->reset();

        // Read baseline before generation
        auto* ctx = llm_->getContext();
        int64_t prefill_before = ctx->prefill_us;
        int64_t decode_before  = ctx->decode_us;

        // Full run: prefill + decode
        auto output_ids = llm_->generate(prompt_ids, n_generate);

        // Read after — take delta (context accumulates across calls)
        int64_t prefill_delta = ctx->prefill_us - prefill_before;
        int64_t decode_delta  = ctx->decode_us  - decode_before;

        if (r == 0) continue;  // skip warmup (same as official llm_bench)

        double prefill_s = prefill_delta / 1e6;
        double decode_s  = decode_delta  / 1e6;

        double prefill_tok_s = (prefill_s > 0) ? (double)n_prompt / prefill_s : 0;
        double decode_tok_s  = (decode_s > 0)  ? (double)output_ids.size() / decode_s : 0;

        prefill_speeds.push_back(prefill_tok_s);
        decode_speeds.push_back(decode_tok_s);
    }

    double avg_prefill = 0, avg_decode = 0;
    for (auto v : prefill_speeds) avg_prefill += v;
    for (auto v : decode_speeds)  avg_decode += v;
    result.prefill_tok_per_s = avg_prefill / prefill_speeds.size();
    result.decode_tok_per_s  = avg_decode  / decode_speeds.size();

    return result;
}

// ── deinit ──

void MnnLlmBackend::deinit() {
    delete llm_;
    llm_ = nullptr;
    model_loaded_ = false;
}

MnnLlmBackend::~MnnLlmBackend() {
    deinit();
}

#else
// Stub when MNN not enabled
bool MnnLlmBackend::init(const BenchmarkConfig&) { return false; }
bool MnnLlmBackend::load_model(const std::string&) { return false; }
std::string MnnLlmBackend::generate(const std::string&, int) { return ""; }
void MnnLlmBackend::reset() {}
MnnLlmBackend::LlmBenchResult MnnLlmBackend::benchmark(int, int, int) { return {}; }
void MnnLlmBackend::deinit() {}
MnnLlmBackend::~MnnLlmBackend() {}
#endif
