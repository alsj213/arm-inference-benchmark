#ifndef BENCHMARK_COMMON_BENCHMARK_H_
#define BENCHMARK_COMMON_BENCHMARK_H_

#include "config.h"
#include "utils.h"

#include <vector>
#include <string>
#include <memory>

struct BenchmarkResult {
    std::string backend_name;
    std::string model_name;
    Precision precision;
    int num_threads;
    bool use_gpu;

    double init_time_ms;
    size_t peak_memory_kb;
    utils::Stats latency_stats;
    double throughput_fps;
};

class BenchmarkBackend {
public:
    virtual ~BenchmarkBackend() = default;

    // Initialize the backend and load model
    virtual bool init(const BenchmarkConfig& config) = 0;

    // Run inference once
    virtual bool infer(const std::vector<float>& input) = 0;

    // Cleanup
    virtual void deinit() = 0;

    // Get backend name
    virtual std::string name() const = 0;
};

// Run full benchmark
BenchmarkResult run_benchmark(
    std::unique_ptr<BenchmarkBackend> backend,
    const BenchmarkConfig& config,
    size_t input_size
);

// Create backend by type
std::unique_ptr<BenchmarkBackend> create_backend(BackendType type);

#endif // BENCHMARK_COMMON_BENCHMARK_H_
