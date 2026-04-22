#ifndef BENCHMARK_BACKENDS_ORT_BACKEND_H_
#define BENCHMARK_BACKENDS_ORT_BACKEND_H_

#include "../common/benchmark.h"
#include "onnxruntime_cxx_api.h"

class ONNXRTBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    void deinit() override;
    std::string name() const override { return "ONNXRT"; }

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::vector<std::string> input_names_store_;
    std::vector<std::string> output_names_store_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
};

#endif // BENCHMARK_BACKENDS_ORT_BACKEND_H_
