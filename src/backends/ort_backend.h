#ifndef BENCHMARK_BACKENDS_ORT_BACKEND_H_
#define BENCHMARK_BACKENDS_ORT_BACKEND_H_

#include "../common/benchmark.h"
#include "onnxruntime_cxx_api.h"
#include <vector>

class ONNXRTBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    bool infer_with_output(const std::vector<float>& input, std::vector<float>& output) override;
    void deinit() override;
    std::string name() const override { return "ONNXRT"; }

private:
    void create_input_tensors(const std::vector<float>& input,
                              std::vector<Ort::Value>& input_tensors,
                              std::vector<std::vector<int64_t>>& scratch);

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING};
    Ort::Session session_{nullptr};
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::vector<int64_t> input_shape_;
    size_t num_inputs_ = 0;
    std::vector<std::string> input_names_store_;
    std::vector<std::string> output_names_store_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<ONNXTensorElementDataType> input_types_;
    bool profiling_enabled_ = false;
};

#endif // BENCHMARK_BACKENDS_ORT_BACKEND_H_