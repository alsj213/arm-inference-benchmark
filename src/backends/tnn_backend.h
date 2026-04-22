#ifndef BENCHMARK_BACKENDS_TNN_BACKEND_H_
#define BENCHMARK_BACKENDS_TNN_BACKEND_H_

#include "../common/benchmark.h"
#include "tnn/core/tnn.h"
#include "tnn/core/common.h"
#include "tnn/core/blob.h"

class TNNBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    void deinit() override;
    std::string name() const override { return "TNN"; }

private:
    std::shared_ptr<TNN_NS::TNN> tnn_;
    std::shared_ptr<TNN_NS::Instance> instance_;
    TNN_NS::InputShapesMap input_shapes_;
    std::string input_name_;
};

#endif // BENCHMARK_BACKENDS_TNN_BACKEND_H_
