#ifndef BENCHMARK_BACKENDS_MNN_LLM_BACKEND_H_
#define BENCHMARK_BACKENDS_MNN_LLM_BACKEND_H_

#include "../common/benchmark.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace MNN { namespace Transformer { class Llm; } }

/*!
 * \brief MNN LLM Backend — LLM inference via MNN Transformer engine
 *
 * Uses MNN_BUILD_LLM=ON (same libMNN.so as CNN backend, plus LLM engine).
 * Model format: llmexport.py output (config.json + llm.mnn + llm.mnn.weight).
 */
class MnnLlmBackend : public BenchmarkBackend {
 public:
  MnnLlmBackend() = default;
  ~MnnLlmBackend() override;

  bool init(const BenchmarkConfig& config) override;
  bool infer(const std::vector<float>& input) override { return true; }  // unused
  void deinit() override;
  std::string name() const override { return "MNN_LLM"; }

  // LLM-specific API
  bool load_model(const std::string& config_path);
  std::string generate(const std::string& prompt, int max_tokens = 128);
  void reset();

  // Benchmark: prefill (prompt processing) + decode (token generation)
  struct LlmBenchResult {
    double prefill_tok_per_s;   // prompt processing speed
    double decode_tok_per_s;    // token generation speed
    double load_time_s;         // model loading time
    int n_prompt;               // prompt token count
    int n_generate;             // generated token count
  };
  LlmBenchResult benchmark(int n_prompt, int n_generate, int n_repeat = 5);

 private:
  MNN::Transformer::Llm* llm_ = nullptr;
  bool model_loaded_ = false;
};

#endif  // BENCHMARK_BACKENDS_MNN_LLM_BACKEND_H_
