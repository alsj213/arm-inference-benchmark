#include "mnn_llm_backend.h"
#include "common/utils.h"
#include "common/precision.h"
#include "json.hpp"

#ifdef BENCHMARK_MNN
#include "llm/llm.hpp"
#include "MNN_generated.h"       // MNN flatbuffer schema（读 llm.mnn 的量化参数）
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <map>

// ── 精度检测 ──
// MNN 模型的量化位权威记录在 llm.mnn 图文件的 Convolution.op.main.quanParameter.aMaxOrBits
// （线性层/权重主体量化）。注意：llm_config.json 的 tie_embeddings[3] 反映的是嵌入层量化，
// 与主体线性层可能不同（如官方 Qwen3-0.6B：线性层 4bit、嵌入层 8bit），不能作为整体量化依据。
// 优先级：
//   1. llm.mnn flatbuffer 解析 → aMaxOrBits 众数（模型实际状态，最权威）
//   2. export_args.json 的 quant_bit（llmexport 导出参数，回退）
//   3. config.json 的 "precision" 字段（"high"=fp16，"low"=已量化、bit 未知，最后兜底）

// 解析 llm.mnn（MNN 图文件）中 Convolution 的 quanParameter.aMaxOrBits 众数。
// 返回 0 表示解析失败。
static int detect_mnn_linear_bits(const std::string& config_path) {
    using json = nlohmann::json;
    auto dir = config_path.substr(0, config_path.find_last_of('/') + 1);

    // llm_model 文件名来自 config.json（默认 llm.mnn）
    std::string mnn_name = "llm.mnn";
    {
        std::ifstream f(config_path);
        if (f.good()) {
            try { json j; f >> j; mnn_name = j.value("llm_model", "llm.mnn"); }
            catch (...) { /* fall through */ }
        }
    }

    FILE* f = fopen((dir + mnn_name).c_str(), "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return 0; }
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    size_t n = fread(buf.data(), 1, buf.size(), f);
    fclose(f);
    if (n != buf.size()) return 0;

    flatbuffers::Verifier verifier(buf.data(), buf.size());
    if (!MNN::VerifyNetBuffer(verifier)) return 0;
    auto netT = MNN::GetNet(buf.data())->UnPack();

    std::map<int, int> counts;
    for (auto& op : netT->oplists) {
        if (op->type != MNN::OpType_Convolution) continue;
        if (op->main.type != MNN::OpParameter_Convolution2D) continue;
        auto conv = op->main.AsConvolution2D();
        if (!conv || !conv->quanParameter) continue;
        counts[conv->quanParameter->aMaxOrBits]++;
    }
    int best_bit = 0, best_cnt = 0;
    for (auto& [b, c] : counts) {
        if (c > best_cnt) { best_cnt = c; best_bit = b; }
    }
    return best_bit;
}

static precision::Info detect_mnn_precision(const std::string& config_path) {
    using json = nlohmann::json;
    auto dir = config_path.substr(0, config_path.find_last_of('/') + 1);

    // 1. llm.mnn flatbuffer：线性层 aMaxOrBits 众数
    int bits = detect_mnn_linear_bits(config_path);
    if (bits > 0) return precision::from_mnn_quant_bit(bits);

    // 2. export_args.json quant_bit（回退）
    {
        std::ifstream f(dir + "export_args.json");
        if (f.good()) {
            try {
                json j; f >> j;
                if (j.contains("quant_bit") && !j["quant_bit"].is_null()) {
                    return precision::from_mnn_quant_bit(j["quant_bit"].get<int>());
                }
            } catch (...) { /* fall through */ }
        }
    }
    // 3. config.json precision 字段（最后兜底）
    {
        std::ifstream f(config_path);
        if (f.good()) {
            try {
                json j; f >> j;
                std::string prec = j.value("precision", "low");
                if (prec == "high") return {"f16", "fp16"};
                if (prec == "low")  return {"other", "quantized"};
            } catch (...) { /* fall through */ }
        }
    }
    return {"", "unknown"};
}

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

    // ── Match native buildLLM() set_config calls (llm_bench.cpp:1087-1158) ──
    // Config keys NOT already in config.json are set explicitly.
    llm_->set_config(R"({"async":false})");                // ensure GPU/CPU sync
    llm_->set_config(R"({"reuse_kv":false})");             // reset KV cache each run
    llm_->set_config(R"({"power":"normal"})");             // power mode (default: normal)
    llm_->set_config(R"({"dynamic_option":0})");           // scheduling (0 for n_prompt<=300)
    llm_->set_config(R"({"attention_mode":8})");           // flash_attention=1, quant_kv=0
    llm_->set_config(R"({"use_mmap":false})");             // no mmap (match native)
    llm_->set_config(R"({"tmp_path":"tmp"})");             // intermediate buffer path

    bool ok = llm_->load();
    if (!ok) {
        printf("MNN_LLM: load failed\n");
        return false;
    }

    // ── Match native tuning_prepare() (llm_bench.cpp:1161-1162) ──
    llm_->tuning(MNN::Transformer::OP_ENCODER_NUMBER,
                 {1, 5, 10, 20, 30, 50, 100});

    auto end = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration<double>(end - start).count();

    // 精度检测
    auto pinfo = detect_mnn_precision(config_path);
    precision_level_ = pinfo.level;
    precision_label_ = pinfo.label;
    printf("MNN_LLM: Precision: %s (level=%s)\n",
           precision_label_.c_str(), precision_level_.c_str());

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
// Exact match to native llm_bench kv_cache=false mode (llm_bench.cpp:1291-1317):
//   - prefill: response(all_prompt_tokens, max=1)  → speed = n_prompt / prefill_us
//   - decode:  response(single_token, max=n_gen)   → speed = n_gen / decode_us
// Each response() call resets prefill_us/decode_us internally (reuse_kv=false).

MnnLlmBackend::LlmBenchResult MnnLlmBackend::benchmark(
        int n_prompt, int n_generate, int n_repeat) {

    LlmBenchResult result = {};
    if (!model_loaded_) return result;

    result.n_prompt = n_prompt;
    result.n_generate = n_generate;

    // Track peak RSS
    size_t mem_before = utils::get_memory_usage_kb() / 1024;  // MiB
    result.peak_memory_mib = mem_before;

    // Match native: fixed token value 16 (llm_bench.cpp:1293)
    const int tok = 16;
    std::vector<int> prompt_tokens(n_prompt, tok);
    std::vector<int> single_token(1, tok);

    auto* ctx = llm_->getContext();

    double prefill_sum = 0, decode_sum = 0;

    for (int r = 0; r < n_repeat + 1; r++) {  // +1 warmup
        int64_t prefill_us = 0, decode_us = 0;

        // ── Prefill phase ──
        if (n_prompt > 0) {
            llm_->response(prompt_tokens, nullptr, nullptr, 1);
            prefill_us = ctx->prefill_us;
        }

        // ── Decode phase ──
        if (n_generate > 0) {
            llm_->response(single_token, nullptr, nullptr, n_generate);
            decode_us = ctx->decode_us;
        }

        if (r == 0) continue;  // skip warmup

        double prefill_ms = prefill_us / 1000.0;
        double decode_ms  = decode_us / 1000.0;
        double prefill_spd = (n_prompt > 0 && prefill_us > 0)
            ? 1e6 * n_prompt / prefill_us : 0;
        double decode_spd  = (n_generate > 0 && decode_us > 0)
            ? 1e6 * n_generate / decode_us : 0;

        result.prefill_ms.push_back(prefill_ms);
        result.decode_ms.push_back(decode_ms);
        result.prefill_per_iter.push_back(prefill_spd);
        result.decode_per_iter.push_back(decode_spd);
        prefill_sum += prefill_spd;
        decode_sum  += decode_spd;

        size_t cur = utils::get_memory_usage_kb() / 1024;
        if (cur > result.peak_memory_mib) result.peak_memory_mib = cur;
    }

    result.prefill_tok_per_s = prefill_sum / n_repeat;
    result.decode_tok_per_s  = decode_sum  / n_repeat;
    result.ttft_ms = result.prefill_ms.empty() ? 0 : result.prefill_ms[0];
    result.tpot_ms = (n_generate > 0 && !result.decode_ms.empty())
        ? result.decode_ms[0] / n_generate : 0;

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
