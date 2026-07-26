#ifndef BENCHMARK_BACKENDS_MNN_LLM_BACKEND_H_
#define BENCHMARK_BACKENDS_MNN_LLM_BACKEND_H_

#include "../common/benchmark.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace MNN { namespace Transformer { class Llm; } }

// ── VL (Vision-Language) inference input ──
struct MultimodalInput {
    std::vector<uint8_t> image_data;  // RAW RGB pixels, row-major
    int width = 0;
    int height = 0;
    std::string prompt;
    int max_tokens = 128;
};

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
    double prefill_tok_per_s;       // mean prefill speed
    double decode_tok_per_s;        // mean decode speed
    double ttft_ms;                 // time to first token
    double tpot_ms;                 // time per output token
    double load_time_s;             // model loading time
    size_t peak_memory_mib;         // peak RSS during benchmark
    int n_prompt;                   // prompt token count
    int n_generate;                 // generated token count
    std::vector<double> prefill_per_iter;  // tok/s per iteration
    std::vector<double> decode_per_iter;   // tok/s per iteration
    std::vector<double> prefill_ms;  // prefill latency per iteration (ms)
    std::vector<double> decode_ms;   // decode latency per iteration (ms)
  };
  LlmBenchResult benchmark(int n_prompt, int n_generate, int n_repeat = 5);

  // VL (Vision-Language) API
  std::string generate_vl(const MultimodalInput& input);
  LlmBenchResult benchmark_vl(int n_prompt, int n_gen, int n_repeat = 5);

  // @TODO: implement logits extraction when MNN LLM internal API exposes logits
  // Currently returns empty vector — needed for accuracy verification
  std::vector<float> get_last_logits() const;

 private:
  MNN::Transformer::Llm* llm_ = nullptr;
  bool model_loaded_ = false;
  std::vector<float> last_logits_;
};

#endif  // BENCHMARK_BACKENDS_MNN_LLM_BACKEND_H_
