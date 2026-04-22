#ifndef BENCHMARK_BACKENDS_QNN_BACKEND_H_
#define BENCHMARK_BACKENDS_QNN_BACKEND_H_

#include "../common/benchmark.h"
#include "QnnInterface.h"
#include "QnnSystemInterface.h"

class QNNBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    void deinit() override;
    std::string name() const override { return "QNN"; }

private:
    // QNN handles - filled in when SDK available
    void* backend_handle_ = nullptr;
    void* graph_handle_ = nullptr;
    // TODO: add more QNN types
};

#endif // BENCHMARK_BACKENDS_QNN_BACKEND_H_
