#ifndef BENCHMARK_BACKENDS_MOBILELLM_BACKEND_H_
#define BENCHMARK_BACKENDS_MOBILELLM_BACKEND_H_

#include "../common/benchmark.h"
#include <vector>
#include <string>
#include <cstdint>

/*
 * MobileLLM 后端适配器 — 通过纯 C ABI 封装 third_party/MobileLLM 的推理引擎。
 *
 * 本文件位于 benchmark 项目自身，仅调用 MobileLLM 公开头文件 (mblm/mblm.h) 的
 * C 接口，不修改第三方源码。模型格式: GGUF（与 llama.cpp 生态共用）。
 *
 * 测量方法与 MobileLLM 自带工具 mblm_benchmark 对齐:
 *   - prefill: 一次性喂入 n_prompt 个固定 token
 *   - decode:  循环 n_generate 次单 token 推理
 *   - 输出 tok/s + 每次重复的原始数据 (mean 聚合)
 */
class MobileLlmBackend : public BenchmarkBackend {
 public:
  MobileLlmBackend() = default;
  ~MobileLlmBackend() override;

  bool init(const BenchmarkConfig& config) override;
  bool infer(const std::vector<float>& input) override { return true; }  // unused
  void deinit() override;
  std::string name() const override { return "MobileLLM"; }

  // LLM-specific API（内部持有 mblm_model_t* / mblm_context_t*，用 void* 隔离 C 类型）
  bool load_model(const std::string& model_path, int n_ctx = 320, int n_batch = 512);
  std::string generate(const std::string& prompt, int max_tokens = 128);

  // 模型量化精度（GGUF general.file_type，mblm.h 未暴露 ftype，用共享解析器）
  precision::Info get_precision() const override {
      return {precision_level_, precision_label_};
  }

  // Benchmark：与 mblm_benchmark.c measure_once 语义一致
  struct LlmBenchResult {
    double prefill_tok_per_s;       // prefill mean speed
    double decode_tok_per_s;        // decode mean speed
    double ttft_ms;                 // time to first token (= prefill latency)
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

 private:
  void* model_ = nullptr;  // mblm_model_t*
  void* ctx_ = nullptr;    // mblm_context_t*
  int n_threads_ = 4;
  bool model_loaded_ = false;
  std::string precision_level_;   // 规范级别 (f32/f16/q8/q4/...)
  std::string precision_label_;   // 人类可读 (Q4_K_M / Q8_0 / F16 / ...)
};

#endif  // BENCHMARK_BACKENDS_MOBILELLM_BACKEND_H_
