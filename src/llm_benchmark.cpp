/*
 * LLM Benchmark for llama.cpp
 * Tests Qwen-0.5B and other models on mobile devices
 */

#include "backends/llamacpp_backend.h"
#include <chrono>
#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("   LLM Benchmark (llama.cpp backend)   \n");
    printf("========================================\n\n");

    std::string model_path = "models/nlp/qwen2_0.5b/qwen2-0_5b-instruct-q4_k_m.gguf";
    int max_tokens = 128;
    int n_ctx = 1024;

    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        max_tokens = atoi(argv[2]);
    }

    printf("Model: %s\n", model_path.c_str());
    printf("Max tokens: %d\n", max_tokens);
    printf("Context size: %d\n\n", n_ctx);

    // Initialize backend (don't auto-load model - we'll load with custom n_ctx)
    LlamaCppBackend backend;
    BenchmarkConfig config;
    // Leave model_path empty so init() only initializes the backend without loading
    config.model_path = "";

    printf("Initializing llama.cpp backend...\n");
    if (!backend.init(config)) {
        printf("ERROR: Failed to initialize llama.cpp backend\n");
        return 1;
    }

    // Load model
    printf("\nLoading model...\n");
    auto start_load = std::chrono::high_resolution_clock::now();

    if (!backend.load_model(model_path, n_ctx, 512)) {
        printf("ERROR: Failed to load model\n");
        return 1;
    }

    auto end_load = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(end_load - start_load).count();
    printf("Model loaded in %.2f seconds\n\n", load_time);

    // Test prompt
    std::string prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain what is machine learning in one sentence.\n\n### Response:\n";

    printf("Running generation test...\n");
    printf("Prompt: %s\n\n", prompt.substr(0, 80).c_str());

    // Warm-up generation
    printf("\n--- Warm-up generation ---\n");
    std::string warmup_output = backend.generate(prompt, 16, 0.0f);
    printf("Warm-up output: %s\n\n", warmup_output.c_str());

    // Benchmark generation
    printf("--- Performance Benchmark ---\n");
    auto start_gen = std::chrono::high_resolution_clock::now();

    std::string output = backend.generate(prompt, max_tokens, 0.7f);

    auto end_gen = std::chrono::high_resolution_clock::now();
    double gen_time = std::chrono::duration<double>(end_gen - start_gen).count();

    // Count tokens (approximate - we can improve this later)
    int output_tokens = std::min(max_tokens, (int)output.size() / 4);

    printf("\nGenerated output:\n%s\n\n", output.c_str());
    printf("========================================\n");
    printf("           Performance Results          \n");
    printf("========================================\n");
    printf("Model load time:    %.2f s\n", load_time);
    printf("Generation time:    %.2f s\n", gen_time);
    printf("Tokens generated:   ~%d\n", output_tokens);
    printf("Throughput:         %.2f tokens/sec\n", output_tokens / gen_time);
    printf("Latency per token:  %.2f ms\n", (gen_time / output_tokens) * 1000);
    printf("========================================\n");

    // Cleanup
    backend.deinit();
    printf("\nBenchmark completed!\n");

    return 0;
}
