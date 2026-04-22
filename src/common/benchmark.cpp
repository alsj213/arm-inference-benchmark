#include "benchmark.h"
#include "utils.h"

#include <algorithm>
#include <memory>

#ifdef BENCHMARK_NCNN
#include "backends/ncnn_backend.h"
#endif
#ifdef BENCHMARK_MNN
#include "backends/mnn_backend.h"
#endif
#ifdef BENCHMARK_TNN
#include "backends/tnn_backend.h"
#endif
#ifdef BENCHMARK_TFLITE
#include "backends/tflite_backend.h"
#endif
#ifdef BENCHMARK_QNN
#include "backends/qnn_backend.h"
#endif
#ifdef BENCHMARK_ORT
#include "backends/ort_backend.h"
#endif
#ifdef BENCHMARK_TVM
#include "backends/tvm_backend.h"
#endif

BenchmarkResult run_benchmark(
    std::unique_ptr<BenchmarkBackend> backend,
    const BenchmarkConfig& config,
    size_t input_size
) {
    BenchmarkResult result;
    result.backend_name = backend->name();
    result.model_name = config.model_name;
    result.precision = config.precision;
    result.num_threads = config.num_threads;
    result.use_gpu = config.use_gpu;

    // Initialize and measure init time
    utils::Timer timer;
    bool success = backend->init(config);
    result.init_time_ms = timer.elapsed_ms();

    if (!success) {
        printf("Failed to initialize backend %s\n", backend->name().c_str());
        return result;
    }

    // Generate random input
    std::vector<float> input(input_size);
    utils::fill_random_float(input.data(), input.size());

    // Warmup
    for (int i = 0; i < config.warmup_runs; ++i) {
        backend->infer(input);
    }

    // Measure memory before test
    size_t mem_before = utils::get_memory_usage_kb();

    // Run benchmark
    std::vector<double> times;
    times.reserve(config.test_runs);

    for (int i = 0; i < config.test_runs; ++i) {
        utils::Timer t;
        backend->infer(input);
        times.push_back(t.elapsed_ms());
    }

    // Measure peak memory
    result.peak_memory_kb = utils::get_memory_usage_kb() - mem_before;

    // Calculate statistics
    result.latency_stats = utils::calculate_stats(times);

    // Calculate throughput
    double total_time = 0.0;
    for (double t : times) {
        total_time += t;
    }
    result.throughput_fps = 1000.0 * config.test_runs / total_time;

    backend->deinit();
    return result;
}

std::unique_ptr<BenchmarkBackend> create_backend(BackendType type) {
    switch (type) {
#ifdef BENCHMARK_NCNN
        case BackendType::NCNN:
            return std::make_unique<NCNNBackend>();
#endif
#ifdef BENCHMARK_MNN
        case BackendType::MNN:
            return std::make_unique<MNNBackend>();
#endif
#ifdef BENCHMARK_TNN
        case BackendType::TNN:
            return std::make_unique<TNNBackend>();
#endif
#ifdef BENCHMARK_TFLITE
        case BackendType::TFLITE:
            return std::make_unique<TFLiteBackend>();
#endif
#ifdef BENCHMARK_QNN
        case BackendType::QNN:
            return std::make_unique<QNNBackend>();
#endif
#ifdef BENCHMARK_ORT
        case BackendType::ONNXRT:
            return std::make_unique<ONNXRTBackend>();
#endif
#ifdef BENCHMARK_TVM
        case BackendType::TVM:
            return std::make_unique<TVMBackend>();
#endif
        default:
            return nullptr;
    }
}
