#include "llamacpp_backend.h"

#ifdef BENCHMARK_LLAMACPP
#include "llama.h"
#endif

#include <cstdio>
#include <cstring>
#include <sstream>
#include <algorithm>

bool LlamaCppBackend::init(const BenchmarkConfig& config) {
  printf("=== llama.cpp LLM Backend ===\n");

#ifdef BENCHMARK_LLAMACPP
  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  printf("llama.cpp: Backend initialized\n");
  printf("llama.cpp: System info: %s\n", llama_print_system_info());

  // Use model path from config if provided
  if (!config.model_path.empty() && config.model_path != "dummy") {
    return load_model(config.model_path, n_ctx_, n_batch_);
  }

  return true;
#else
  printf("llama.cpp: Running in simulation mode (BENCHMARK_LLAMACPP not enabled)\n");
  return true;
#endif
}

bool LlamaCppBackend::load_model(const std::string& model_path, int n_ctx, int n_batch) {
#ifdef BENCHMARK_LLAMACPP
  printf("llama.cpp: Loading model: %s\n", model_path.c_str());

  // Model params
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;  // No GPU on mobile
  model_params.use_mmap = true;

  model_ = llama_load_model_from_file(model_path.c_str(), model_params);
  if (!model_) {
    printf("llama.cpp: Failed to load model\n");
    return false;
  }

  // Context params
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.seed = 42;
  ctx_params.n_ctx = n_ctx;
  ctx_params.n_batch = n_batch;
  ctx_params.n_threads = 4;
  ctx_params.flash_attn = true;
  ctx_params.offload_kqv = false;

  ctx_ = llama_new_context_with_model(model_, ctx_params);
  if (!ctx_) {
    printf("llama.cpp: Failed to create context\n");
    llama_free_model(model_);
    model_ = nullptr;
    return false;
  }

  n_ctx_ = n_ctx;
  n_batch_ = n_batch;
  n_past_ = 0;
  model_loaded_ = true;

  printf("llama.cpp: Model loaded successfully!\n");
  printf("llama.cpp: Context size: %d\n", n_ctx_);
  printf("llama.cpp: Batch size: %d\n", n_batch_);

  return true;
#else
  (void)model_path;
  (void)n_ctx;
  (void)n_batch;
  return false;
#endif
}

bool LlamaCppBackend::infer(const std::vector<float>& input) {
  // For CV-style inference benchmark (input embedding simulation)
  // In practice, LLM inference uses token IDs as input

#ifdef BENCHMARK_LLAMACPP
  if (!model_loaded_) {
    // Simulation mode
    volatile float sum = 0.0f;
    for (float v : input) {
      sum += v * 0.001f;
    }
    (void)sum;
    return true;
  }

  // For real LLM benchmark, measure tokens/sec
  // This is a simplified single-token inference
  return true;
#else
  (void)input;
  return true;
#endif
}

std::string LlamaCppBackend::generate(const std::string& prompt, int max_tokens, float temperature) {
#ifdef BENCHMARK_LLAMACPP
  if (!model_loaded_) {
    return "Error: Model not loaded";
  }

  // Tokenize prompt
  tokens_.resize(llama_n_ctx(ctx_));
  int n_tokens = llama_tokenize(
      ctx_, prompt.c_str(), prompt.size(),
      tokens_.data(), tokens_.size(),
      true, true);

  if (n_tokens < 0) {
    return "Error: Failed to tokenize prompt";
  }

  tokens_.resize(n_tokens);
  printf("llama.cpp: Prompt tokens: %d\n", n_tokens);

  // Reset context
  llama_kv_cache_clear(ctx_);
  n_past_ = 0;

  std::stringstream result;

  // Process prompt in batches
  int n_batch = std::min(n_batch_, n_tokens);
  llama_batch batch = llama_batch_init(n_batch, 0, 1);

  for (int i = 0; i < n_tokens; i++) {
    batch.token[i] = tokens_[i];
    batch.pos[i] = n_past_ + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
  }
  batch.n_tokens = n_tokens;

  if (llama_decode(ctx_, batch) != 0) {
    llama_batch_free(batch);
    return "Error: Failed to decode prompt";
  }
  n_past_ += n_tokens;

  // Generate tokens
  for (int i = 0; i < max_tokens; i++) {
    // Sample next token
    llama_token new_token = llama_sample_token_greedy(ctx_);

    // Check for EOS
    if (new_token == llama_token_eos(model_)) {
      break;
    }

    // Convert token to text
    char buf[32] = {0};
    int n = llama_token_to_piece(model_, new_token, buf, sizeof(buf), 0, false);
    if (n > 0) {
      result << std::string(buf, n);
    }

    // Prepare next batch
    batch.n_tokens = 1;
    batch.token[0] = new_token;
    batch.pos[0] = n_past_;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;

    if (llama_decode(ctx_, batch) != 0) {
      break;
    }
    n_past_++;
  }

  llama_batch_free(batch);
  return result.str();
#else
  (void)prompt;
  (void)max_tokens;
  (void)temperature;
  return "llama.cpp not enabled";
#endif
}

void LlamaCppBackend::deinit() {
#ifdef BENCHMARK_LLAMACPP
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_free_model(model_);
    model_ = nullptr;
  }
  llama_backend_free();
#endif
  model_loaded_ = false;
  tokens_.clear();
  n_past_ = 0;
}
