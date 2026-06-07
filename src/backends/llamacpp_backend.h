#ifndef BENCHMARK_BACKENDS_LLAMACPP_BACKEND_H_
#define BENCHMARK_BACKENDS_LLAMACPP_BACKEND_H_

#include "../common/benchmark.h"
#include <vector>
#include <string>

#ifdef BENCHMARK_LLAMACPP
#include "llama.h"
#endif

/*!
 * \brief llama.cpp Backend - LLM Inference Engine
 *
 * High-performance LLM inference with GGUF format:
 *   - Model format: .gguf (GGML Unified Format)
 *   - Quantization: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32
 *   - Optimized: ARM NEON SIMD, KV Cache, Flash Attention
 *   - Supported models: LLaMA, Qwen, Phi, Mistral, Gemma, etc.
 */
class LlamaCppBackend : public BenchmarkBackend {
 public:
  bool init(const BenchmarkConfig& config) override;
  bool infer(const std::vector<float>& input) override;
  void deinit() override;
  std::string name() const override { return "llama.cpp"; }

  // LLM-specific API
  bool load_model(const std::string& model_path, int n_ctx = 2048, int n_batch = 512);
  std::string generate(const std::string& prompt, int max_tokens = 128, float temperature = 0.7f);
  int get_context_length() const { return n_ctx_; }
  int get_batch_size() const { return n_batch_; }

 private:
#ifdef BENCHMARK_LLAMACPP
  llama_context* ctx_ = nullptr;
  llama_model* model_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  llama_sampler* smpl_ = nullptr;
  llama_batch batch_;
#endif
  int n_ctx_ = 2048;
  int n_batch_ = 512;
  int n_past_ = 0;
  bool model_loaded_ = false;
  std::vector<int> tokens_;
};

#endif  // BENCHMARK_BACKENDS_LLAMACPP_BACKEND_H_
