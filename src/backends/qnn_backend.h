#ifndef BENCHMARK_BACKENDS_QNN_BACKEND_H_
#define BENCHMARK_BACKENDS_QNN_BACKEND_H_

#include "../common/benchmark.h"

// Forward declaration of internal state
struct QnnBackendState;

class QNNBackend : public BenchmarkBackend {
public:
    QNNBackend();
    ~QNNBackend() override;

    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    void deinit() override;
    std::string name() const override { return "QNN"; }

private:
    QnnBackendState* state_;
};

#endif // BENCHMARK_BACKENDS_QNN_BACKEND_H_
