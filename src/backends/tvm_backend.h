#ifndef BENCHMARK_BACKENDS_TVM_BACKEND_H_
#define BENCHMARK_BACKENDS_TVM_BACKEND_H_

#include "../common/benchmark.h"
#include <string>
#include <vector>

// TVM v0.15.0 Relay GraphExecutor 后端
// 使用 tvm::runtime::Module / PackedFunc / NDArray (非 tvm-ffi)
class TVMBackend : public BenchmarkBackend {
public:
    TVMBackend() = default;
    ~TVMBackend() override;
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    bool infer_with_output(const std::vector<float>& input, std::vector<float>& output) override;
    void deinit() override;
    std::string name() const override { return "TVM"; }

private:
    void* mod_ = nullptr;            // tvm::runtime::Module*
    void* set_input_ = nullptr;      // tvm::runtime::PackedFunc*
    void* run_ = nullptr;            // tvm::runtime::PackedFunc*
    void* get_output_ = nullptr;     // tvm::runtime::PackedFunc*
    void* input_ndarray_ = nullptr;  // tvm::runtime::NDArray* (预分配，单输入模型)
    void* h_rt_ = nullptr;           // dlopen handle for libtvm_runtime.so
    std::string input_name_{"input"};
    std::vector<int64_t> input_shape_;
    size_t input_size_ = 0;
    int num_model_inputs_ = 1;
    bool initialized_ = false;
};

#endif
