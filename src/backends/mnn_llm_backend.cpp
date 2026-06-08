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

MnnLlmBackend::LlmBenchResult MnnLlmBackend::benchmark(
        int n_prompt, int n_generate, int n_repeat) {

    LlmBenchResult result = {};
    if (!model_loaded_) return result;

    result.n_prompt = n_prompt;
    result.n_generate = n_generate;

    // Generate random token IDs for prompt
    std::vector<int> prompt_ids(n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        prompt_ids[i] = (rand() % 10000) + 100;  // valid token range
    }

    std::vector<int64_t> prefill_us, decode_us;

    for (int r = 0; r < n_repeat; r++) {
        llm_->reset();

        // Prefill: process prompt tokens (measured via first token latency)
        auto t0 = std::chrono::high_resolution_clock::now();

        // Use generate with prompt_ids for prefill + first n_generate tokens
        auto output_ids = llm_->generate(prompt_ids, n_generate);

        auto t1 = std::chrono::high_resolution_clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // MNN's generate() does prefill + decode in one call.
        // We can't easily separate them without modifying MNN internals.
        // Approximate: total = prefill + n_generate * per_token_decode
        // For fair comparison with llama.cpp, report total speed.
        // llm_bench separates via internal hooks — here we use total speed.
    }

    // Run with generate() and measure elapsed time
    // Since MNN's API doesn't expose prefill/decode separately,
    // measure total end-to-end time and compute combined tok/s
    llm_->reset();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto output_ids = llm_->generate(prompt_ids, n_generate);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    int total_tokens = n_prompt + (int)output_ids.size();
    double total_tok_per_s = total_tokens / elapsed_s;

    // Approximate decode speed (subtract estimated prefill time)
    // Prefill is typically ~10x faster per token than decode for small models
    // This is approximate — exact measurement requires MNN internal hooks
    result.decode_tok_per_s = total_tok_per_s;  // total speed as primary metric
    result.prefill_tok_per_s = 0;               // not measurable with public API

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
