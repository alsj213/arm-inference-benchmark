#ifndef BENCHMARK_BACKENDS_MINDSPORE_LITE_BACKEND_H_
#define BENCHMARK_BACKENDS_MINDSPORE_LITE_BACKEND_H_

#include "common/benchmark.h"
#include "common/config.h"

#ifdef BENCHMARK_MINDSPORE_LITE
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/status.h"
#include "include/api/types.h"
#endif

class MindSporeLiteBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    bool infer_with_output(const std::vector<float>& input, std::vector<float>& output) override;
    void deinit() override;
    std::string name() const override { return "mindspore_lite"; }

private:
#ifdef BENCHMARK_MINDSPORE_LITE
    std::shared_ptr<mindspore::Model> model_;
    std::shared_ptr<mindspore::Context> context_;
    std::vector<mindspore::MSTensor> inputs_;
    std::vector<mindspore::MSTensor> outputs_;
#endif
    std::string input_name_;
};

#endif // BENCHMARK_BACKENDS_MINDSPORE_LITE_BACKEND_H_
