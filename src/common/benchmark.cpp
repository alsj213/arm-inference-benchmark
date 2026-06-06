#include "benchmark.h"
#include "utils.h"

#include <algorithm>
#include <memory>
#include <cmath>
#include <numeric>
#include <sstream>

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
#ifdef BENCHMARK_MINDSPORE_LITE
#include "backends/mindspore_lite_backend.h"
#endif
#ifdef BENCHMARK_LLAMACPP
#include "backends/llamacpp_backend.h"
#endif

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
#ifdef BENCHMARK_MINDSPORE_LITE
        case BackendType::MINDSPORE_LITE:
            return std::make_unique<MindSporeLiteBackend>();
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
    std::ostringstream ss;
    ss << "{";
    ss << "\"backend\":\"" << backend_name << "\",";
    ss << "\"model\":\"" << model_name << "\",";
    ss << "\"precision\":\"" << (precision == Precision::FP32 ? "fp32" : precision == Precision::FP16 ? "fp16" : "int8") << "\",";
    ss << "\"threads\":" << num_threads << ",";
    ss << "\"gpu\":" << (use_gpu ? "true" : "false") << ",";
    ss << "\"init_time_ms\":" << init_time_ms << ",";
    ss << "\"peak_memory_kb\":" << peak_memory_kb << ",";
    ss << "\"latency\":{";
    ss << "\"min\":" << latency_stats.min_ms << ",";
    ss << "\"max\":" << latency_stats.max_ms << ",";
    ss << "\"mean\":" << latency_stats.mean_ms << ",";
    ss << "\"p50\":" << latency_stats.p50_ms << ",";
    ss << "\"p90\":" << latency_stats.p90_ms << ",";
    ss << "\"p95\":" << latency_stats.p95_ms << ",";
    ss << "\"p99\":" << latency_stats.p99_ms << ",";
    ss << "\"std_dev\":" << latency_stats.std_dev;
    ss << "},";
    ss << "\"throughput_fps\":" << throughput_fps << ",";
    ss << "\"accuracy\":{";
    ss << "\"passed\":" << (accuracy.passed ? "true" : "false") << ",";
    ss << "\"cosine_similarity\":" << accuracy.cosine_similarity << ",";
    ss << "\"mean_absolute_error\":" << accuracy.mean_absolute_error << ",";
    ss << "\"max_absolute_error\":" << accuracy.max_absolute_error;
    ss << "}";
    ss << "}";
    return ss.str();
}
