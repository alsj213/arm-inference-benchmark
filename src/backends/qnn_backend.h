#ifndef BENCHMARK_BACKENDS_QNN_BACKEND_H_
#define BENCHMARK_BACKENDS_QNN_BACKEND_H_

#include "../common/benchmark.h"

// QNN implementation is in cpp file only
class QNNBackendImpl;

class QNNBackend : public BenchmarkBackend {
public:
    QNNBackend();
    ~QNNBackend() override;

    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    void deinit() override;
    std::string name() const override { return "QNN"; }

private:
    QNNBackendImpl* impl_ = nullptr;
};

#endif // BENCHMARK_BACKENDS_QNN_BACKEND_H_
