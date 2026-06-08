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
// NOTE: MNN's public generate() API bundles prefill+decode into one call.
// Prefill runs ~4x faster than decode per token, so combined speed overstates
// real decode speed. The official llm_bench separates them via internal hooks.
// Use combined speed for relative comparison between runs; for absolute
// decode-only speed, reference the official llm_bench output.

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

    std::vector<double> total_speeds, decode_speeds;

    for (int r = 0; r < n_repeat; r++) {
        llm_->reset();

        // Phase 1: generate 1 token to measure prefill-included cost
        auto t0 = std::chrono::high_resolution_clock::now();
        auto first = llm_->generate(prompt_ids, 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        double prefill_s = std::chrono::duration<double>(t1 - t0).count();
        double prefill_tok_s = (double)n_prompt / prefill_s;  // prompt processing speed

        // Phase 2: full run prefill+decode (public API limitation)
        llm_->reset();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto output_ids = llm_->generate(prompt_ids, n_generate);
        auto t3 = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t3 - t2).count();

        int total_tokens = n_prompt + (int)output_ids.size();
        total_speeds.push_back(total_tokens / total_s);

        // Approximate decode: subtract prefill time (from phase 1)
        double decode_s = total_s - prefill_s;
        double decode_speed = (output_ids.size() > 0) ? (double)output_ids.size() / decode_s : 0;
        decode_speeds.push_back(decode_speed);
    }

    // Average across repeats
    double avg_total = 0, avg_decode = 0;
    for (auto v : total_speeds) avg_total += v;
    for (auto v : decode_speeds) avg_decode += v;
    result.decode_tok_per_s = avg_decode / decode_speeds.size();
    result.prefill_tok_per_s = 0;

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
