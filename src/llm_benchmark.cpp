/*
 * LLM Benchmark — unified entry for llama.cpp + MNN LLM
 * Usage: ./llm_benchmark [--backend <llamacpp|mnn_llm>] [--model <path>]
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

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
    int n_ctx = 1024;
    int n_prompt = 128;
    int n_repeat = 5;
    bool benchmark_only = false;
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
        // Benchmark mode: random tokens, token-level API (same as llama-bench)
        printf("--- Benchmark (n_prompt=%d, n_gen=%d, repeat=%d) ---\n",
               args.n_prompt, args.max_tokens, args.n_repeat);
        auto r = backend.benchmark_decode(args.n_prompt, args.max_tokens, args.n_repeat);
        printf("prefill: %.2f tok/s  |  decode: %.2f tok/s\n",
               r.prefill_tok_per_s, r.decode_tok_per_s);
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
        // Benchmark mode: random token IDs, measure decode
        printf("--- Benchmark (n_prompt=%d, n_gen=%d, repeat=%d) ---\n",
               args.n_prompt, args.max_tokens, args.n_repeat);
        auto result = backend.benchmark(args.n_prompt, args.max_tokens, args.n_repeat);
        printf("prefill: %.2f tok/s  |  decode: %.2f tok/s\n",
               result.prefill_tok_per_s, result.decode_tok_per_s);
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

// ── main ──
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    if (args.backend == "mnn_llm" || args.backend == "mnn") {
        return run_mnn_llm(args) ? 0 : 1;
    } else {
        return run_llamacpp(args) ? 0 : 1;
    }
}
