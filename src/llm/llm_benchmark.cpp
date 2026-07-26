/*
 * LLM Benchmark — unified entry for llama.cpp + MNN LLM
 * Usage: ./llm_benchmark [--backend <llamacpp|mnn_llm>] [--model <path>]
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

#include "common/benchmark.h"
#include "common/utils.h"

// Read device temperature (first thermal zone > 0, in °C)
static int read_temp() {
    for (int i = 0; i < 20; i++) {
        char path[64];
        snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", i);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        int val = 0;
        fscanf(f, "%d", &val);
        fclose(f);
        if (val > 0) return val / 1000;
    }
    return -1;
}

// Print benchmark statistics with mean±std, percentiles, TTFT, TPOT
static void print_llm_stats(const std::string& label,
                            double mean_tok_s,
                            const std::vector<double>& per_iter,
                            const std::vector<double>& per_ms,
                            double ttft_or_tpot_ms) {
    printf("  %s:\n", label.c_str());
    printf("    Mean ± Std:  %.2f ± ", mean_tok_s);
    if (per_iter.size() >= 2) {
        double sum = 0;
        for (auto v : per_iter) sum += v;
        double mean = sum / per_iter.size();
        double var = 0;
        for (auto v : per_iter) var += (v - mean) * (v - mean);
        var /= per_iter.size();
        printf("%.2f tok/s\n", std::sqrt(var));
    } else {
        printf("N/A\n");
    }
    if (!per_ms.empty()) {
        auto stats = utils::calculate_stats(per_ms);
        printf("    P50/P90/P99: %.2f / %.2f / %.2f ms\n",
               stats.p50_ms, stats.p90_ms, stats.p99_ms);
        printf("    Min / Max:   %.2f / %.2f ms\n", stats.min_ms, stats.max_ms);
    }
    if (ttft_or_tpot_ms > 0) {
        printf("    TTFT/TPOT:   %.2f ms\n", ttft_or_tpot_ms);
    }
}

#ifdef BENCHMARK_LLAMACPP
#include "backends/llamacpp_backend.h"
#endif
#ifdef BENCHMARK_MNN
#include "backends/mnn_llm_backend.h"
#endif

struct Args {
    std::string backend = "llamacpp";
    std::string model;
    int max_tokens = 128;
    int n_ctx = 256;  // llama-bench: n_prompt + n_gen
    int n_prompt = 128;
    int n_repeat = 5;
    bool benchmark_only = false;
    // VL mode
    std::string image_path;       // --image <path>
    int image_width = 0;          // --image-size <w> <h>
    int image_height = 0;
    std::string accuracy_ref;     // --accuracy <mnn|llamacpp>
    int seed = 42;                // --seed <n>
    std::string prompt_text;      // --prompt <text>
    bool json_output = false;     // --json
};

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --backend <llamacpp|mnn_llm>   LLM backend (default: llamacpp)\n");
    printf("  --model <path>                  Model file/config path\n");
    printf("  --max-tokens <n>                Max tokens to generate (default: 128)\n");
    printf("  --n-prompt <n>                  Prompt length for benchmark (default: 128)\n");
    printf("  --n-repeat <n>                  Benchmark repeats (default: 5)\n");
    printf("  --benchmark                     Benchmark mode (random tokens, perf only)\n");
    printf("  --image <path>                  Image file for VL inference\n");
    printf("  --image-size <w> <h>           RAW image dimensions\n");
    printf("  --accuracy <mnn|llamacpp>       Accuracy verification mode\n");
    printf("  --seed <n>                      Random seed (default: 42)\n");
    printf("  --prompt <text>                 Custom prompt text\n");
    printf("  --json                          Output results as JSON lines\n");
    printf("  --help                          Show this help\n");
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            args.backend = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-prompt") == 0 && i + 1 < argc) {
            args.n_prompt = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-repeat") == 0 && i + 1 < argc) {
            args.n_repeat = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            args.benchmark_only = true;
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if (strcmp(argv[i], "--image-size") == 0 && i + 2 < argc) {
            args.image_width = atoi(argv[++i]);
            args.image_height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--accuracy") == 0 && i + 1 < argc) {
            args.accuracy_ref = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args.prompt_text = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0) {
            args.json_output = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }
    return args;
}

// ── llama.cpp path ──
#ifdef BENCHMARK_LLAMACPP
static bool run_llamacpp(const Args& args) {
    printf("========================================\n");
    printf("   LLM Benchmark (llama.cpp)\n");
    printf("========================================\n\n");

    std::string model_path = args.model.empty()
        ? "models/llm/qwen2_0.5b/qwen2-0_5b-instruct-q4_k_m.gguf"
        : args.model;

    printf("Model: %s\n", model_path.c_str());
    printf("Max tokens: %d\n\n", args.max_tokens);

    LlamaCppBackend backend;
    BenchmarkConfig config;
    config.model_path = "";

    printf("Initializing llama.cpp...\n");
    if (!backend.init(config)) {
        printf("ERROR: init failed\n");
        return false;
    }

    printf("Loading model...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!backend.load_model(model_path, args.n_ctx, 512)) {
        printf("ERROR: load failed\n");
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_s = std::chrono::duration<double>(t1 - t0).count();
    printf("Loaded in %.2f s\n\n", load_s);

    if (args.benchmark_only) {
        int temp_before = read_temp();
        printf("--- Benchmark (n_prompt=%d, n_gen=%d, repeat=%d) ---\n",
               args.n_prompt, args.max_tokens, args.n_repeat);
        printf("    Device temp: %d°C\n", temp_before);
        auto r = backend.benchmark_decode(args.n_prompt, args.max_tokens, args.n_repeat);
        int temp_after = read_temp();

        print_llm_stats("Prefill", r.prefill_tok_per_s,
                        r.prefill_per_iter, r.prefill_ms, r.ttft_ms);
        print_llm_stats("Decode",  r.decode_tok_per_s,
                        r.decode_per_iter,  r.decode_ms,  r.tpot_ms);
        printf("  Peak Memory: %zu MiB\n", r.peak_memory_mib);
        printf("  Device temp: %d → %d°C\n", temp_before, temp_after);

        if (args.json_output) {
            json j;
            j["run_id"] = generate_run_id();
            j["timestamp"] = now_iso8601();
            j["git_commit"] = GIT_COMMIT_HASH;
            j["track"] = "llm";
            j["framework"] = "llama.cpp";
            j["model"] = model_path;
            j["mode"] = "benchmark";
            j["metrics"] = {
                {"prefill_tok_per_s", r.prefill_tok_per_s},
                {"decode_tok_per_s", r.decode_tok_per_s},
                {"ttft_ms", r.ttft_ms},
                {"tpot_ms", r.tpot_ms},
                {"peak_memory_mib", r.peak_memory_mib},
                {"temp_before", temp_before},
                {"temp_after", temp_after},
                {"n_prompt", args.n_prompt},
                {"n_gen", args.max_tokens},
                {"n_repeat", args.n_repeat}
            };
            // Per-iteration data
            j["metrics"]["prefill_per_iter"] = r.prefill_per_iter;
            j["metrics"]["decode_per_iter"]  = r.decode_per_iter;
            printf("%s\n", j.dump().c_str());
        }
    } else {
        // Interactive mode
        std::string prompt = "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nExplain what is machine learning in one sentence.\n\n### Response:\n";

        // Warm-up
        printf("--- Warm-up ---\n");
        std::string warmup = backend.generate(prompt, 16, 0.0f);
        printf("Warm-up: %s\n\n", warmup.c_str());

        // Generation
        printf("--- Generate ---\n");
        auto st = std::chrono::high_resolution_clock::now();
        std::string output = backend.generate(prompt, args.max_tokens, 0.7f);
        auto et = std::chrono::high_resolution_clock::now();
        double gen_s = std::chrono::duration<double>(et - st).count();

        printf("\n%s\n\n", output.c_str());
        printf("--- Results ---\n");
        printf("Generation time: %.2f s\n", gen_s);
        printf("Throughput: ~%.2f tok/s\n", args.max_tokens / gen_s);
    }

    backend.deinit();
    return true;
}
#else
static bool run_llamacpp(const Args&) {
    printf("llama.cpp backend not compiled (BENCHMARK_LLAMACPP=OFF)\n");
    return false;
}
#endif

// ── llama.cpp VL path ──
#ifdef BENCHMARK_LLAMACPP
static bool run_llamacpp_vl(const Args& args) {
    printf("========================================\n");
    printf("   VL Benchmark (llama.cpp)\n");
    printf("========================================\n\n");

    std::string model_path = args.model.empty()
        ? "models/qwen3-vl-4b-instruct-q4_k_m.gguf" : args.model;

    printf("Model: %s\n", model_path.c_str());
    printf("Image: %s\n", args.image_path.c_str());

    LlamaCppBackend backend;
    // Note: subprocess mode — no init/load_model needed

    std::string prompt = args.prompt_text.empty()
        ? "请用中文详细描述这张图片" : args.prompt_text;

    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n\n", args.max_tokens);

    if (args.benchmark_only) {
        // Simplified VL benchmark: warmup + measured run
        printf("--- VL Benchmark (simplified) ---\n");

        // Warmup
        printf("Warmup...\n");
        backend.generate_vl(args.image_path, prompt, 16);

        // Measured run
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = backend.generate_vl(args.image_path, prompt, args.max_tokens);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t1 - t0).count();

        printf("\n--- Results ---\n");
        printf("Vision time: %.2f s\n", result.vision_time_s);
        printf("Prefill time: %.2f s\n", result.prefill_time_s);
        printf("Decode time: %.2f s\n", result.decode_time_s);
        printf("Total tokens: %d\n", result.total_tokens);
        printf("Total time: %.2f s\n", total_s);
        printf("Decode speed: ~%.2f tok/s\n", args.max_tokens / total_s);

        if (args.json_output) {
            json j;
            j["run_id"] = generate_run_id();
            j["timestamp"] = now_iso8601();
            j["git_commit"] = GIT_COMMIT_HASH;
            j["track"] = "llm";
            j["framework"] = "llama.cpp";
            j["model"] = model_path;
            j["mode"] = "vl_benchmark";
            j["metrics"] = {
                {"vision_time_s", result.vision_time_s},
                {"prefill_time_s", result.prefill_time_s},
                {"decode_time_s", result.decode_time_s},
                {"total_tokens", result.total_tokens},
                {"total_time_s", total_s}
            };
            printf("%s\n", j.dump().c_str());
        }
    } else {
        printf("--- VL Generate ---\n");
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = backend.generate_vl(args.image_path, prompt, args.max_tokens);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t1 - t0).count();

        printf("\n%s\n\n", result.text.c_str());

        printf("--- Results ---\n");
        printf("Vision time: %.2f s\n", result.vision_time_s);
        printf("Prefill time: %.2f s\n", result.prefill_time_s);
        printf("Decode time: %.2f s\n", result.decode_time_s);
        printf("Total tokens: %d\n", result.total_tokens);
        printf("Total time: %.2f s\n", total_s);
    }

    return true;
}
#else
static bool run_llamacpp_vl(const Args&) {
    printf("llama.cpp backend not compiled (BENCHMARK_LLAMACPP=OFF)\n");
    return false;
}
#endif

// ── MNN LLM path ──
#ifdef BENCHMARK_MNN
static bool run_mnn_llm(const Args& args) {
    printf("========================================\n");
    printf("   LLM Benchmark (MNN LLM)\n");
    printf("========================================\n\n");

    std::string config_path = args.model.empty()
        ? "models/llm/qwen2_0.5b/mnn_llm/config.json"
        : args.model;

    printf("Config: %s\n", config_path.c_str());
    printf("Max tokens: %d\n\n", args.max_tokens);

    MnnLlmBackend backend;
    BenchmarkConfig config;
    config.model_path = config_path;

    printf("Loading MNN LLM model...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!backend.load_model(config_path)) {
        printf("ERROR: load failed\n");
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_s = std::chrono::duration<double>(t1 - t0).count();
    printf("Loaded in %.2f s\n\n", load_s);

    if (args.benchmark_only) {
        int temp_before = read_temp();
        printf("--- Benchmark (n_prompt=%d, n_gen=%d, repeat=%d) ---\n",
               args.n_prompt, args.max_tokens, args.n_repeat);
        printf("    Device temp: %d°C\n", temp_before);
        auto result = backend.benchmark(args.n_prompt, args.max_tokens, args.n_repeat);
        int temp_after = read_temp();

        print_llm_stats("Prefill", result.prefill_tok_per_s,
                        result.prefill_per_iter, result.prefill_ms, result.ttft_ms);
        print_llm_stats("Decode",  result.decode_tok_per_s,
                        result.decode_per_iter,  result.decode_ms,  result.tpot_ms);
        printf("  Peak Memory: %zu MiB\n", result.peak_memory_mib);
        printf("  Device temp: %d → %d°C\n", temp_before, temp_after);

        if (args.json_output) {
            json j;
            j["run_id"] = generate_run_id();
            j["timestamp"] = now_iso8601();
            j["git_commit"] = GIT_COMMIT_HASH;
            j["track"] = "llm";
            j["framework"] = "MNN_LLM";
            j["model"] = config_path;
            j["mode"] = "benchmark";
            j["metrics"] = {
                {"prefill_tok_per_s", result.prefill_tok_per_s},
                {"decode_tok_per_s", result.decode_tok_per_s},
                {"ttft_ms", result.ttft_ms},
                {"tpot_ms", result.tpot_ms},
                {"load_time_s", result.load_time_s},
                {"peak_memory_mib", result.peak_memory_mib},
                {"temp_before", temp_before},
                {"temp_after", temp_after},
                {"n_prompt", result.n_prompt},
                {"n_generate", result.n_generate}
            };
            j["metrics"]["prefill_per_iter"] = result.prefill_per_iter;
            j["metrics"]["decode_per_iter"]  = result.decode_per_iter;
            printf("%s\n", j.dump().c_str());
        }
    } else {
        // Interactive mode
        std::string prompt = "Hello, explain what machine learning is in one sentence.";

        printf("--- Warm-up ---\n");
        std::string warmup = backend.generate("Hello", 16);
        printf("Warm-up: %s\n\n", warmup.c_str());

        printf("--- Generate ---\n");
        auto st = std::chrono::high_resolution_clock::now();
        std::string output = backend.generate(prompt, args.max_tokens);
        auto et = std::chrono::high_resolution_clock::now();
        double gen_s = std::chrono::duration<double>(et - st).count();

        printf("\n%s\n\n", output.c_str());
        printf("--- Results ---\n");
        printf("Generation time: %.2f s\n", gen_s);
        printf("Throughput: ~%.2f tok/s\n", args.max_tokens / gen_s);
    }

    backend.deinit();
    return true;
}
#else
static bool run_mnn_llm(const Args&) {
    printf("MNN LLM backend not compiled (BENCHMARK_MNN=OFF)\n");
    return false;
}
#endif

// ── MNN LLM VL path ──
#ifdef BENCHMARK_MNN
static bool run_mnn_llm_vl(const Args& args) {
    printf("========================================\n");
    printf("   VL Benchmark (MNN LLM)\n");
    printf("========================================\n\n");

    std::string config_path = args.model.empty()
        ? "models/qwen3-vl-4b-mnn/config.json" : args.model;

    printf("Config: %s\n", config_path.c_str());

    MnnLlmBackend backend;
    BenchmarkConfig config;
    config.model_path = config_path;

    printf("Loading MNN VL model...\n");
    if (!backend.load_model(config_path)) {
        printf("ERROR: load failed\n");
        return false;
    }
    printf("Model loaded.\n\n");

    // Read image file (RAW RGB, no header)
    FILE* fp = fopen(args.image_path.c_str(), "rb");
    if (!fp) {
        printf("ERROR: cannot open image: %s\n", args.image_path.c_str());
        return false;
    }
    fseek(fp, 0, SEEK_END);
    size_t fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> img_data(fsize);
    size_t read_bytes = fread(img_data.data(), 1, fsize, fp);
    fclose(fp);
    if (read_bytes != fsize) {
        printf("ERROR: incomplete image read (%zu of %zu bytes)\n", read_bytes, fsize);
        return false;
    }
    printf("Read image: %zu bytes from %s\n", read_bytes, args.image_path.c_str());

    // Determine dimensions (420 default for Qwen3-VL)
    int w = args.image_width ? args.image_width : 420;
    int h = args.image_height ? args.image_height : 420;
    std::string prompt = args.prompt_text.empty()
        ? "请用中文详细描述这张图片" : args.prompt_text;

    printf("Image size: %dx%d\n", w, h);
    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n\n", args.max_tokens);

    MultimodalInput input;
    input.image_data = std::move(img_data);
    input.width = w;
    input.height = h;
    input.prompt = prompt;
    input.max_tokens = args.max_tokens;

    if (args.benchmark_only) {
        printf("--- VL Benchmark (n_prompt=%d, n_gen=%d, repeat=%d) ---\n",
               args.n_prompt, args.max_tokens, args.n_repeat);
        auto r = backend.benchmark_vl(args.n_prompt, args.max_tokens, args.n_repeat);
        printf("prefill: %.2f tok/s  |  decode: %.2f tok/s\n",
               r.prefill_tok_per_s, r.decode_tok_per_s);

        if (args.json_output) {
            json j;
            j["run_id"] = generate_run_id();
            j["timestamp"] = now_iso8601();
            j["git_commit"] = GIT_COMMIT_HASH;
            j["track"] = "llm";
            j["framework"] = "MNN_LLM";
            j["model"] = config_path;
            j["mode"] = "vl_benchmark";
            j["metrics"] = {
                {"prefill_tok_per_s", r.prefill_tok_per_s},
                {"decode_tok_per_s", r.decode_tok_per_s},
                {"n_prompt", r.n_prompt},
                {"n_generate", r.n_generate}
            };
            printf("%s\n", j.dump().c_str());
        }
    } else {
        printf("--- VL Generate ---\n");
        auto t0 = std::chrono::high_resolution_clock::now();
        std::string output = backend.generate_vl(input);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t1 - t0).count();
        printf("\n%s\n\n", output.c_str());
        printf("--- Results ---\n");
        printf("Total time: %.2f s\n", total_s);
    }

    backend.deinit();
    return true;
}
#else
static bool run_mnn_llm_vl(const Args&) {
    printf("MNN LLM backend not compiled (BENCHMARK_MNN=OFF)\n");
    return false;
}
#endif

// ── main ──
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    // Initialize random seed
    srand(args.seed);

    // --accuracy is parsed but not yet implemented
    if (!args.accuracy_ref.empty()) {
        printf("WARNING: --accuracy mode is not yet implemented, running in normal mode\n");
    }

    if (!args.image_path.empty()) {
        // VL mode
        if (args.backend == "mnn_llm" || args.backend == "mnn") {
            return run_mnn_llm_vl(args) ? 0 : 1;
        } else {
            return run_llamacpp_vl(args) ? 0 : 1;
        }
    }

    // Original text-only path
    if (args.backend == "mnn_llm" || args.backend == "mnn") {
        return run_mnn_llm(args) ? 0 : 1;
    } else {
        return run_llamacpp(args) ? 0 : 1;
    }
}
