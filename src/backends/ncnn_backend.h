#ifndef BENCHMARK_BACKENDS_NCNN_BACKEND_H_
#define BENCHMARK_BACKENDS_NCNN_BACKEND_H_

#include "../common/benchmark.h"
#include "net.h"
#include "allocator.h"

class NCNNBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    void deinit() override;
    std::string name() const override { return "ncnn"; }

private:
    ncnn::Net net_;
    ncnn::Mat input_;
    ncnn::UnlockedPoolAllocator blob_allocator_;
    ncnn::UnlockedPoolAllocator workspace_allocator_;
    std::string input_name_;
    std::string output_name_;
};

#endif // BENCHMARK_BACKENDS_NCNN_BACKEND_H_
