#ifndef BENCHMARK_BACKENDS_MNN_BACKEND_H_
#define BENCHMARK_BACKENDS_MNN_BACKEND_H_

#include "../common/benchmark.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

class MNNBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    bool infer_with_output(const std::vector<float>& input, std::vector<float>& output) override;
    void deinit() override;
    std::string name() const override { return "MNN"; }

private:
    std::unique_ptr<MNN::Interpreter> net_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;
};

#endif // BENCHMARK_BACKENDS_MNN_BACKEND_H_
