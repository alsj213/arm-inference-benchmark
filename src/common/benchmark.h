#ifndef BENCHMARK_COMMON_BENCHMARK_H_
#define BENCHMARK_COMMON_BENCHMARK_H_

#include "config.h"
#include "utils.h"

#include <vector>
#include <string>
#include <memory>

struct AccuracyResult {
    bool passed;
    double cosine_similarity;      // 余弦相似度 (接近 1.0 为最佳)
    double mean_absolute_error;    // 平均绝对误差
    double max_absolute_error;     // 最大绝对误差
    double mean_relative_error;    // 平均相对误差
    double output_min;
    double output_max;
    double output_mean;
};

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

    // Accuracy comparison results
    AccuracyResult accuracy;

    std::string to_json() const;
};

class BenchmarkBackend {
public:
    virtual ~BenchmarkBackend() = default;

    // Initialize the backend and load model
    virtual bool init(const BenchmarkConfig& config) = 0;

    // Run inference once
    virtual bool infer(const std::vector<float>& input) = 0;

    // Run inference and get output (default implementation returns false - not supported)
    virtual bool infer_with_output(const std::vector<float>& input, std::vector<float>& output) {
        return false; // Default: not implemented
    }

    // Cleanup
    virtual void deinit() = 0;

    // Get backend name
    virtual std::string name() const = 0;
};

// Run full benchmark
BenchmarkResult run_benchmark(
    std::unique_ptr<BenchmarkBackend> backend,
    const BenchmarkConfig& config,
    size_t input_size,
    const std::vector<float>& reference_output = {}  // Optional: reference output from ORT
);

// Create backend by type
std::unique_ptr<BenchmarkBackend> create_backend(BackendType type);

#endif // BENCHMARK_COMMON_BENCHMARK_H_
