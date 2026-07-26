#include "llamacpp_backend.h"
#include "common/utils.h"

#ifdef BENCHMARK_LLAMACPP
#include "llama.h"
#endif

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <array>
#include <cctype>
#include <memory>

// Execute shell command and capture stdout/stderr via pipe.
// Returns {output, exit_code}.
static std::pair<std::string, int> exec_cmd(const std::string& cmd) {
    std::array<char, 4096> buffer;
    std::string result;
    FILE* raw = popen(cmd.c_str(), "r");
    if (!raw) return {"", -1};
    while (fgets(buffer.data(), buffer.size(), raw) != nullptr)
        result += buffer.data();
    int exit_code = pclose(raw);
    return {result, exit_code};
}

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
  if (model_loaded_) {
    printf("llama.cpp: Model already loaded, skipping\n");
    return true;
  }
  printf("llama.cpp: Loading model: %s\n", model_path.c_str());

  n_ctx_ = n_ctx;
  n_batch_ = n_batch;
  n_threads_ = 4;

  // Model params — match native llama-bench to_llama_mparams()
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;  // No GPU on mobile

  model_ = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!model_) {
    printf("llama.cpp: Failed to load model\n");
    return false;
  }

  vocab_ = llama_model_get_vocab(model_);

  // Context params — match native llama-bench to_llama_cparams()
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx       = n_ctx_;                  // llama-bench: n_prompt + n_gen + n_depth
  ctx_params.n_batch     = n_batch_;
  ctx_params.n_ubatch    = std::min(n_batch_, 512);
  ctx_params.type_k      = GGML_TYPE_F16;
  ctx_params.type_v      = GGML_TYPE_F16;
  ctx_params.offload_kqv = true;
  ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
  ctx_params.op_offload  = true;
  ctx_params.embeddings  = false;                   // match native default

  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_) {
    printf("llama.cpp: Failed to create context\n");
    llama_model_free(model_);
    model_ = nullptr;
    vocab_ = nullptr;
    return false;
  }

  // ── Threadpool: match native llama-bench (lines 2341-2360) ──
  ggml_backend_load_all();
  auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (cpu_dev) {
    auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *)
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new");
    if (ggml_threadpool_new_fn) {
      struct ggml_threadpool_params tpp =
          ggml_threadpool_params_default(n_threads_);
      tpp.poll = 50;  // native default poll value
      threadpool_ = ggml_threadpool_new_fn(&tpp);
      if (threadpool_) {
        llama_attach_threadpool(ctx_, threadpool_, NULL);
        printf("llama.cpp: Threadpool attached (%d threads)\n", n_threads_);
      }
    }
  }
  if (!threadpool_) {
    printf("llama.cpp: Threadpool not available, using set_n_threads\n");
  }

  // Create greedy sampler chain (used only by generate(), not benchmark)
  auto sparams = llama_sampler_chain_default_params();
  smpl_ = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());

  model_loaded_ = true;

  printf("llama.cpp: Model loaded successfully!\n");
  printf("llama.cpp: Context size: %d\n", n_ctx_);
  printf("llama.cpp: Batch size: %d / ubatch: %d\n", n_batch_, ctx_params.n_ubatch);

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
  if (!model_loaded_ || !vocab_) {
    return "Error: Model not loaded";
  }

  // Tokenize prompt (new API: takes vocab instead of context)
  int n_ctx = llama_n_ctx(ctx_);
  tokens_.resize(n_ctx);
  int n_tokens = llama_tokenize(
      vocab_, prompt.c_str(), prompt.size(),
      tokens_.data(), tokens_.size(),
      true, true);

  if (n_tokens < 0) {
    return "Error: Failed to tokenize prompt";
  }

  tokens_.resize(n_tokens);
  printf("llama.cpp: Prompt tokens: %d\n", n_tokens);

  // Clear KV cache for sequence 0 (new API: llama_memory_seq_rm)
  llama_memory_t mem = llama_get_memory(ctx_);
  llama_memory_seq_rm(mem, 0, 0, -1);
  n_past_ = 0;

  std::stringstream result;

  // Process prompt in one batch
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
    // Sample next token (new API: use sampler chain)
    llama_token new_token = llama_sampler_sample(smpl_, ctx_, -1);

    // Check for EOS (new API: llama_vocab_eos takes vocab)
    if (new_token == llama_vocab_eos(vocab_)) {
      break;
    }

    // Convert token to text (new API: llama_token_to_piece takes vocab)
    char buf[32] = {0};
    int n = llama_token_to_piece(vocab_, new_token, buf, sizeof(buf), 0, false);
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

LlamaCppBackend::TokenBenchResult LlamaCppBackend::benchmark_decode(
        int n_prompt, int n_gen, int n_repeat) {
    TokenBenchResult r = {};
#ifdef BENCHMARK_LLAMACPP
    if (!model_loaded_ || !ctx_) return r;

    const int32_t n_vocab = llama_vocab_n_tokens(vocab_);
    const bool add_bos = llama_vocab_get_add_bos(vocab_);
    const int32_t bos_token = add_bos ? llama_vocab_bos(vocab_) : 0;
    auto mem = llama_get_memory(ctx_);

    // Track peak RSS during benchmark
    auto sample_mem = []() { return utils::get_memory_usage_kb() / 1024; };  // MiB
    size_t mem_before = sample_mem();
    r.peak_memory_mib = mem_before;

    // ── Prefill benchmark (match native test_prompt + timing loop) ──
    if (n_prompt > 0) {
        // Warmup (match native lines 2363-2375)
        llama_memory_clear(mem, false);
        llama_set_n_threads(ctx_, n_threads_, n_threads_);
        std::vector<llama_token> w_tokens(n_batch_);
        int n_processed = 0;
        while (n_processed < n_prompt) {
            int n_tokens = std::min(n_prompt - n_processed, n_batch_);
            w_tokens[0] = (n_processed == 0 && add_bos)
                ? bos_token : std::rand() % n_vocab;
            for (int i = 1; i < n_tokens; i++)
                w_tokens[i] = std::rand() % n_vocab;
            llama_decode(ctx_, llama_batch_get_one(w_tokens.data(), n_tokens));
            n_processed += n_tokens;
        }
        llama_synchronize(ctx_);

        double prefill_sum = 0;
        for (int rep = 0; rep < n_repeat; rep++) {
            llama_memory_clear(mem, false);
            llama_set_n_threads(ctx_, n_threads_, n_threads_);
            n_processed = 0;

            auto t0 = std::chrono::high_resolution_clock::now();

            while (n_processed < n_prompt) {
                int n_tokens = std::min(n_prompt - n_processed, n_batch_);
                w_tokens[0] = (n_processed == 0 && add_bos)
                    ? bos_token : std::rand() % n_vocab;
                for (int i = 1; i < n_tokens; i++)
                    w_tokens[i] = std::rand() % n_vocab;
                llama_decode(ctx_, llama_batch_get_one(w_tokens.data(), n_tokens));
                n_processed += n_tokens;
            }
            llama_synchronize(ctx_);

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double spd = (double)n_prompt / (ms / 1000.0);
            r.prefill_ms.push_back(ms);
            r.prefill_per_iter.push_back(spd);
            prefill_sum += spd;

            // Sample peak memory
            size_t cur = sample_mem();
            if (cur > r.peak_memory_mib) r.peak_memory_mib = cur;
        }
        r.prefill_tok_per_s = prefill_sum / n_repeat;
        r.ttft_ms = r.prefill_ms.empty() ? 0 : r.prefill_ms[0];
    }

    // ── Decode benchmark (match native test_gen + timing loop) ──
    if (n_gen > 0) {
        // Warmup (match native lines 2377-2388)
        llama_memory_clear(mem, false);
        llama_set_n_threads(ctx_, n_threads_, n_threads_);
        llama_token w_token = add_bos ? bos_token : std::rand() % n_vocab;
        for (int i = 0; i < 1; i++) {  // native: test_gen(ctx, 1)
            llama_decode(ctx_, llama_batch_get_one(&w_token, 1));
            llama_synchronize(ctx_);
            w_token = std::rand() % n_vocab;
        }

        double decode_sum = 0, decode_ms_sum = 0;
        for (int rep = 0; rep < n_repeat; rep++) {
            llama_memory_clear(mem, false);
            llama_set_n_threads(ctx_, n_threads_, n_threads_);
            w_token = add_bos ? bos_token : std::rand() % n_vocab;

            auto t0 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < n_gen; i++) {
                llama_decode(ctx_, llama_batch_get_one(&w_token, 1));
                llama_synchronize(ctx_);
                w_token = std::rand() % n_vocab;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double spd = (double)n_gen / (ms / 1000.0);
            r.decode_ms.push_back(ms);
            r.decode_per_iter.push_back(spd);
            decode_sum += spd;
            decode_ms_sum += ms;

            size_t cur = sample_mem();
            if (cur > r.peak_memory_mib) r.peak_memory_mib = cur;
        }
        r.decode_tok_per_s = decode_sum / n_repeat;
        r.tpot_ms = (n_gen > 0) ? decode_ms_sum / (n_repeat * n_gen) : 0;
    }
#else
    (void)n_prompt; (void)n_gen; (void)n_repeat;
#endif
    return r;
}

LlamaCppBackend::VLBatchResult LlamaCppBackend::generate_vl(
        const std::string& image_path,
        const std::string& prompt,
        int max_tokens,
        const std::string& mmproj_path) {

    VLBatchResult r;

    // Build model paths (assumes standard device layout)
    std::string model_dir = "models/qwen3-vl-4b";
    std::string mmproj = mmproj_path.empty()
        ? model_dir + "/Qwen3-VL-4B-Instruct-f16.mmproj"
        : mmproj_path;
    std::string gguf = model_dir + "/Qwen3-VL-4B-Instruct-q4_k_m.gguf";
    std::string mtmd = "./llama-mtmd-cli";

    // Build command with stderr merged into stdout (2>&1)
    // Escape single quotes in prompt to prevent shell injection
    std::string escaped_prompt = prompt;
    size_t pos = 0;
    while ((pos = escaped_prompt.find("'", pos)) != std::string::npos) {
        escaped_prompt.replace(pos, 1, "'\\''");
        pos += 4;
    }
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "cd /data/local/tmp/benchmark && "
        "LD_LIBRARY_PATH=. %s "
        "-m %s "
        "--mmproj %s "
        "--image %s "
        "-p '%s' "
        "-n %d "
        "-t 4 "
        "--no-warmup "
        "--perf 2>&1",
        mtmd.c_str(), gguf.c_str(), mmproj.c_str(),
        image_path.c_str(), escaped_prompt.c_str(), max_tokens);

    auto [output, exit_code] = exec_cmd(cmd);
    if (exit_code != 0) {
        printf("WARNING: llama-mtmd-cli exited with code %d (output may be partial)\n", exit_code);
    }

    // Parse timing from perf output
    // Format: llama_perf_context_print: prompt eval time = X ms / Y tokens
    // Format: llama_perf_context_print: eval time = X ms / Y runs
    // Format: image slice encoded in X ms
    // Format: image decoded in X ms

    auto parse_ms = [](const std::string& text, const std::string& key) -> double {
        auto pos = text.find(key);
        if (pos == std::string::npos) return 0;
        pos += key.size();
        while (pos < text.size() && text[pos] == ' ') pos++;
        // Read number until space or 'm'
        std::string num;
        while (pos < text.size() && (isdigit(text[pos]) || text[pos] == '.'))
            num += text[pos++];
        return num.empty() ? 0 : std::stod(num);
    };

    // vision_time = encode + decode (both in seconds)
    r.vision_time_s = parse_ms(output, "image slice encoded in") / 1000.0
                    + parse_ms(output, "image decoded in") / 1000.0;
    r.prefill_time_s = parse_ms(output, "prompt eval time =") / 1000.0;
    r.decode_time_s = parse_ms(output, "eval time =") / 1000.0;

    // Extract generated text (between prompt and "llama_perf_context_print")
    auto perf_pos = output.find("llama_perf_context_print");
    if (perf_pos != std::string::npos) {
        // Find the last newline before perf output
        auto text_start = output.rfind('\n', perf_pos);
        if (text_start != std::string::npos)
            text_start = output.rfind('\n', text_start - 1);
        if (text_start == std::string::npos) text_start = 0;
        r.text = output.substr(text_start, perf_pos - text_start);
        // Trim whitespace
        while (!r.text.empty() && (r.text.back() == '\n' || r.text.back() == ' '))
            r.text.pop_back();
    }

    // Parse total tokens from " / Y tokens" in prompt eval line
    {
        auto pos = output.find("prompt eval time =");
        if (pos != std::string::npos) {
            auto slash = output.find('/', pos);
            if (slash != std::string::npos) {
                auto tok_pos = slash + 1;
                while (tok_pos < output.size() && output[tok_pos] == ' ') tok_pos++;
                std::string num;
                while (tok_pos < output.size() && isdigit(output[tok_pos]))
                    num += output[tok_pos++];
                r.total_tokens = num.empty() ? 0 : std::stoi(num);
            }
        }
    }

    return r;
}

void LlamaCppBackend::deinit() {
#ifdef BENCHMARK_LLAMACPP
  if (threadpool_) {
    // Match native: lookup free function dynamically
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev) {
      auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
      auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *)
          ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free");
      if (ggml_threadpool_free_fn) {
        ggml_threadpool_free_fn(threadpool_);
      }
    }
    threadpool_ = nullptr;
  }
  if (smpl_) {
    llama_sampler_free(smpl_);
    smpl_ = nullptr;
  }
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
  vocab_ = nullptr;
  llama_backend_free();
#endif
  model_loaded_ = false;
  tokens_.clear();
  n_past_ = 0;
}
