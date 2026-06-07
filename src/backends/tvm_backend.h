#ifndef BENCHMARK_BACKENDS_TVM_BACKEND_H_
#define BENCHMARK_BACKENDS_TVM_BACKEND_H_

#include "../common/benchmark.h"
#include <string>
#include <vector>

// 前向声明 — 实际 TVM 类型只在 .cpp 中使用
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
    void* mod_ = nullptr;          // ffi::Module*
    void* vm_ = nullptr;           // ffi::Module*
    void* set_input_ = nullptr;    // ffi::Function*
    void* invoke_ = nullptr;       // ffi::Function*
    void* get_outputs_ = nullptr;  // ffi::Function*
    std::string func_name_{"main"};
    std::vector<int64_t> input_shape_;
    size_t input_size_ = 0;
    bool initialized_ = false;
};

#endif
