#include "mnn_llm_backend.h"

#ifdef BENCHMARK_MNN
#include "llm/llm.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExprCreator.hpp>
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

// ── VL generate ──

std::string MnnLlmBackend::generate_vl(const MultimodalInput& input) {
    if (!model_loaded_ || !llm_) return "";

    // Create VARP from raw RGB data
    using namespace MNN::Express;
    auto varp = _Const(input.image_data.data(),
        {input.height, input.width, 3}, NCHW, halide_type_of<uint8_t>());
    varp = _Cast<float>(varp) * _Const(1.0f / 255.0f);

    // Build MultimodalPrompt (skip thinking mode for direct answer)
    MNN::Transformer::MultimodalPrompt mm_prompt;
    mm_prompt.prompt_template = "<|im_start|>system\nYou are a helpful assistant. Answer directly without thinking.\n<|im_end|>\n<|im_start|>user\n<img>img1</img>" +
        input.prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n";
    mm_prompt.images["img1"] = {varp, input.width, input.height};

    // Run inference
    std::ostringstream oss;
    llm_->response(mm_prompt, &oss, nullptr, input.max_tokens);
    return oss.str();
}

// ── VL benchmark ──

MnnLlmBackend::LlmBenchResult MnnLlmBackend::benchmark_vl(
        int n_prompt, int n_gen, int n_repeat) {
    LlmBenchResult result = {};
    if (!model_loaded_) return result;

    // Create a dummy image (420x420 gradient like test pattern)
    int w = 420, h = 420;
    std::vector<uint8_t> dummy_img(w * h * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            dummy_img[(y*w+x)*3+0] = (uint8_t)(255 * x / w);
            dummy_img[(y*w+x)*3+1] = (uint8_t)(255 * y / h);
            dummy_img[(y*w+x)*3+2] = 128;
        }

    MultimodalInput input;
    input.image_data = dummy_img;
    input.width = w;
    input.height = h;
    input.prompt = "describe the image";
    input.max_tokens = n_gen;

    std::vector<double> prefill_speeds, decode_speeds;

    for (int r = 0; r < n_repeat + 1; r++) {
        llm_->reset();

        auto* ctx = llm_->getContext();
        int64_t vision_before = ctx->vision_us;
        int64_t prefill_before = ctx->prefill_us;
        int64_t decode_before = ctx->decode_us;

        generate_vl(input);

        int64_t vision_delta = ctx->vision_us - vision_before;
        int64_t prefill_delta = ctx->prefill_us - prefill_before;
        int64_t decode_delta = ctx->decode_us - decode_before;

        if (r == 0) continue;  // warmup

        double prefill_s = prefill_delta / 1e6;
        double decode_s = decode_delta / 1e6;

        prefill_speeds.push_back((prefill_s > 0) ? (double)n_prompt / prefill_s : 0);
        decode_speeds.push_back((decode_s > 0) ? (double)n_gen / decode_s : 0);
    }

    double avg_prefill = 0, avg_decode = 0;
    for (auto v : prefill_speeds) avg_prefill += v;
    for (auto v : decode_speeds)  avg_decode += v;
    result.prefill_tok_per_s = avg_prefill / prefill_speeds.size();
    result.decode_tok_per_s  = avg_decode  / decode_speeds.size();

    return result;
}

// ── logits access ──

std::vector<float> MnnLlmBackend::get_last_logits() const {
    // TODO: extract from llm_->getContext() when logits field is available
    return last_logits_;
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
std::string MnnLlmBackend::generate_vl(const MultimodalInput&) { return ""; }
MnnLlmBackend::LlmBenchResult MnnLlmBackend::benchmark_vl(int, int, int) { return {}; }
std::vector<float> MnnLlmBackend::get_last_logits() const { return {}; }
void MnnLlmBackend::deinit() {}
MnnLlmBackend::~MnnLlmBackend() {}
#endif
