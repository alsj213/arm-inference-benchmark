#ifndef BENCHMARK_BACKENDS_MNN_BACKEND_H_
#define BENCHMARK_BACKENDS_MNN_BACKEND_H_

#include "../common/benchmark.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

class MNNBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool prepare(const std::vector<float>& input) override;
    bool run() override;
    bool infer(const std::vector<float>& input) override;
    bool infer_with_output(const std::vector<float>& input, std::vector<float>& output) override;
    void deinit() override;
    std::string name() const override { return use_gpu_ ? "MNN_GPU" : "MNN"; }

private:
    std::unique_ptr<MNN::Interpreter> net_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;
    MNN::Tensor* output_tensor_ = nullptr;  // cached output, avoids per-iter getSessionOutput
    std::unique_ptr<MNN::Tensor> host_input_tensor_;   // Host alias for input (benchmark.out createHostTensorFromDevice)
    std::unique_ptr<MNN::Tensor> host_output_tensor_;  // Host alias for output (benchmark.out createHostTensorFromDevice)
    bool profiling_enabled_ = false;
    std::string profile_file_;
    bool use_gpu_ = false;
};

#endif // BENCHMARK_BACKENDS_MNN_BACKEND_H_
