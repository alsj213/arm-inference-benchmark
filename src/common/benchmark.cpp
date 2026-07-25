#include "benchmark.h"
#include "utils.h"

#include "json.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <random>
using json = nlohmann::json;

#include <algorithm>
#include <memory>
#include <cmath>
#include <numeric>
#include <sstream>

#ifdef BENCHMARK_MNN
#include "backends/mnn_backend.h"
#endif
#ifdef BENCHMARK_ORT
#include "backends/ort_backend.h"
#endif
#ifdef BENCHMARK_TVM
#include "backends/tvm_backend.h"
#endif
#ifdef BENCHMARK_LLAMACPP
#include "backends/llamacpp_backend.h"
#endif

// ── Utility functions: run ID and timestamp ──
std::string generate_run_id() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::gmtime(&t), "%Y%m%d-");
    // 4-digit random hex
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 0xFFFF);
    ss << std::hex << std::setw(4) << std::setfill('0') << dis(gen);
    return ss.str();
}

std::string now_iso8601() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

// Calculate cosine similarity between two vectors
double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }

    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }

    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0;
    }

    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// Compare output with reference and calculate accuracy metrics
AccuracyResult compare_with_reference(
    const std::vector<float>& output,
    const std::vector<float>& reference_output,
    const std::string& backend_name
) {
    AccuracyResult result;
    result.passed = false;
    result.cosine_similarity = 0.0;
    result.mean_absolute_error = 0.0;
    result.max_absolute_error = 0.0;
    result.mean_relative_error = 0.0;

    if (output.empty() || reference_output.empty()) {
        printf("  ⚠️  [Accuracy] %s: Empty output\n", backend_name.c_str());
        return result;
    }

    if (output.size() != reference_output.size()) {
        printf("  ⚠️  [Accuracy] %s: Output size mismatch (%zu vs %zu)\n",
               backend_name.c_str(), output.size(), reference_output.size());
        return result;
    }

    // Check for NaN
    for (float val : output) {
        if (std::isnan(val)) {
            printf("  ⚠️  [Accuracy] %s: Output contains NaN\n", backend_name.c_str());
            return result;
        }
    }

    // Calculate cosine similarity
    result.cosine_similarity = cosine_similarity(output, reference_output);

    // Calculate absolute errors
    double sum_abs_error = 0.0;
    double max_abs_error = 0.0;
    double sum_rel_error = 0.0;
    int rel_error_count = 0;

    for (size_t i = 0; i < output.size(); ++i) {
        double abs_error = std::abs(static_cast<double>(output[i]) - reference_output[i]);
        sum_abs_error += abs_error;
        max_abs_error = std::max(max_abs_error, abs_error);

        // Relative error (avoid division by zero)
        if (std::abs(reference_output[i]) > 1e-6) {
            sum_rel_error += abs_error / std::abs(reference_output[i]);
            rel_error_count++;
        }
    }

    result.mean_absolute_error = sum_abs_error / output.size();
    result.max_absolute_error = max_abs_error;
    result.mean_relative_error = (rel_error_count > 0) ? sum_rel_error / rel_error_count : 0.0;

    // Output statistics
    result.output_min = *std::min_element(output.begin(), output.end());
    result.output_max = *std::max_element(output.begin(), output.end());
    result.output_mean = std::accumulate(output.begin(), output.end(), 0.0) / output.size();

    // Pass criteria: cosine similarity > 0.99 (for FP32)
    result.passed = (result.cosine_similarity > 0.99);

    // Print results
    if (result.passed) {
        printf("  ✅ [Accuracy] %s: PASSED\n", backend_name.c_str());
    } else {
        printf("  ⚠️  [Accuracy] %s: WARNING (cosine similarity low)\n", backend_name.c_str());
    }

    printf("      Cosine Similarity:  %.6f %s\n",
           result.cosine_similarity,
           result.cosine_similarity > 0.999 ? "(Excellent)" :
           result.cosine_similarity > 0.99  ? "(Good)" :
           result.cosine_similarity > 0.95  ? "(Acceptable)" : "(Poor)");
    printf("      Mean Absolute Error: %.6f\n", result.mean_absolute_error);
    printf("      Max Absolute Error:  %.6f\n", result.max_absolute_error);
    printf("      Mean Relative Error: %.4f%%\n", result.mean_relative_error * 100.0);
    printf("      Output range: [%.4f, %.4f], Mean: %.4f\n",
           result.output_min, result.output_max, result.output_mean);

    return result;
}

BenchmarkResult run_benchmark(
    std::unique_ptr<BenchmarkBackend> backend,
    const BenchmarkConfig& config,
    size_t input_size,
    const std::vector<float>& reference_output
) {
    BenchmarkResult result;
    result.run_id = generate_run_id();
    result.timestamp = now_iso8601();
    result.backend_name = backend->name();
    result.model_name = config.model_name;
    result.precision = config.precision;
    result.precision_str = (config.precision == Precision::FP32 ? "fp32"
                           : config.precision == Precision::FP16 ? "fp16" : "int8");
    result.num_threads = config.num_threads;
    result.use_gpu = config.use_gpu;
    result.warmup_runs = config.warmup_runs;
    result.test_runs = config.test_runs;

    // Initialize and measure init time
    utils::Timer timer;
    bool success = backend->init(config);
    result.init_time_ms = timer.elapsed_ms();

    if (!success) {
        printf("Failed to initialize backend %s\n", backend->name().c_str());
        return result;
    }

    // Use fixed seed for reproducible input
    std::vector<float> input(input_size);
    utils::fill_random_float(input.data(), input.size(), (unsigned int)42);  // Fixed seed = 42

    // Accuracy verification: compare with reference output
    if (!reference_output.empty()) {
        printf("\n--- Accuracy Comparison ---\n");
        std::vector<float> output;
        if (backend->infer_with_output(input, output)) {
            result.accuracy = compare_with_reference(output, reference_output, backend->name());
        } else {
            printf("  ⚠️  [Accuracy] %s: Failed to get output\n", backend->name().c_str());
            result.accuracy.passed = false;
        }
    }

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
#ifdef BENCHMARK_MNN
        case BackendType::MNN:
        case BackendType::MNN_GPU:
            return std::make_unique<MNNBackend>();
#endif
#ifdef BENCHMARK_ORT
        case BackendType::ONNXRT:
            return std::make_unique<ONNXRTBackend>();
#endif
#ifdef BENCHMARK_TVM
        case BackendType::TVM:
            return std::make_unique<TVMBackend>();
#endif
#ifdef BENCHMARK_LLAMACPP
        case BackendType::LLAMACPP:
            return std::make_unique<LlamaCppBackend>();
#endif
        default:
            return nullptr;
    }
}

std::string BenchmarkResult::to_json() const {
    json j;
    j["run_id"] = run_id;
    j["timestamp"] = timestamp;
    j["git_commit"] = GIT_COMMIT_HASH;
    j["track"] = "cnn";
    j["model"] = model_name;
    j["framework"] = backend_name;
    j["precision"] = precision_str;
    j["threads"] = num_threads;
    j["warmup_runs"] = warmup_runs;
    j["test_runs"] = test_runs;
    j["init_time_ms"] = init_time_ms;
    j["peak_memory_kb"] = peak_memory_kb;
    j["metrics"] = {
        {"p50_ms", latency_stats.p50_ms},
        {"p90_ms", latency_stats.p90_ms},
        {"p99_ms", latency_stats.p99_ms},
        {"mean_ms", latency_stats.mean_ms},
        {"min_ms", latency_stats.min_ms},
        {"max_ms", latency_stats.max_ms},
        {"std_dev", latency_stats.std_dev},
        {"throughput_fps", throughput_fps}
    };
    j["accuracy"] = {
        {"passed", accuracy.passed},
        {"cosine_similarity", accuracy.cosine_similarity},
        {"mean_absolute_error", accuracy.mean_absolute_error},
        {"max_absolute_error", accuracy.max_absolute_error},
        {"mean_relative_error", accuracy.mean_relative_error}
    };
    return j.dump();
}
