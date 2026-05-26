#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <numeric>

#include "common/benchmark.h"
#include "common/utils.h"
#include "common/config.h"

#include "models/model_info.h"
#include "model_registry.h"

void print_usage() {
    printf("Usage: ./benchmark_inference [options]\n");
    printf("Options:\n");
    printf("  --backend <mnn|onnxrt|ort>  Backend to test (default: all)\n");
    printf("  --model <mobilenetv2|resnet50|yolov8n|bert>  Model to test (default: all)\n");
    printf("  --precision <fp32|fp16|int8>                  Precision (default: fp32)\n");
    printf("  --threads <num>                                Number of threads (default: 1)\n");
    printf("  --warmup <num>                                 Number of warmup runs (default: 10)\n");
    printf("  --runs <num>                                   Number of test runs (default: 100)\n");
    printf("  --gpu                                          Use GPU if available\n");
    printf("  --profiling <file>                             Enable operator profiling (output to file)\n");
    printf("  --json                                          Output results as JSON lines\n");
    printf("  --help                                         Show this help\n");
}

struct CommandLineArgs {
    std::string backend = "all";
    std::string model = "all";
    std::string precision = "fp32";
    int threads = 1;
    int warmup = 10;
    int runs = 100;
    bool use_gpu = false;
    bool help = false;
    bool json_output = false;
    bool show_version = false;
    std::string profiling_file = "";  // Profiling output file
};

CommandLineArgs parse_args(int argc, char** argv) {
    CommandLineArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            args.backend = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--precision" && i + 1 < argc) {
            args.precision = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            args.threads = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            args.runs = std::atoi(argv[++i]);
        } else if (arg == "--gpu") {
            args.use_gpu = true;
        } else if (arg == "--profiling" && i + 1 < argc) {
            args.profiling_file = argv[++i];
        } else if (arg == "--json") {
            args.json_output = true;
        } else if (arg == "--help") {
            args.help = true;
        } else if (arg == "--version") {
            args.show_version = true;
        }
    }
    return args;
}

Precision parse_precision(const std::string& p) {
    if (p == "fp16") return Precision::FP16;
    if (p == "int8") return Precision::INT8;
    return Precision::FP32;
}

BackendType parse_backend(const std::string& b) {
    // 已停用的后端映射保留在此，重新启用后取消注释即可
    if (b == "ncnn" || b == "NCNN") return BackendType::NCNN;
    if (b == "mnn" || b == "MNN") return BackendType::MNN;
    // if (b == "tnn" || b == "TNN") return BackendType::TNN;       // TNN — 已停用
    // if (b == "tflite" || b == "TFLite" || b == "TFLITE") return BackendType::TFLITE;  // TFLite — 已停用
    // if (b == "qnn" || b == "QNN") return BackendType::QNN;
    if (b == "onnxrt" || b == "ort" || b == "ONNXRT" || b == "ORT") return BackendType::ONNXRT;
    // if (b == "tvm" || b == "TVM") return BackendType::TVM;
    if (b == "mindspore_lite" || b == "mslite") return BackendType::MINDSPORE_LITE;
    return (BackendType)-1;
}

int main(int argc, char** argv) {
    CommandLineArgs args = parse_args(argc, argv);
    if (args.help) {
        print_usage();
        return 0;
    }
    if (args.show_version) {
        printf("benchmark_inference %s\n", GIT_COMMIT_HASH);
        return 0;
    }

    printf("=== ARM Inference Benchmark ===\n");
    printf("Backend:  %s\n", args.backend.c_str());
    printf("Model:    %s\n", args.model.c_str());
    printf("Precision:%s\n", args.precision.c_str());
    printf("Threads:  %d\n", args.threads);
    printf("Warmup:   %d\n", args.warmup);
    printf("Test runs:%d\n", args.runs);
    printf("Use GPU:  %s\n", args.use_gpu ? "yes" : "no");
    printf("\n");

    std::vector<std::string> backends_to_test;
    if (args.backend == "all") {
        backends_to_test = {"mnn", "onnxrt", "ncnn", "mindspore_lite"};
    } else {
        backends_to_test = {args.backend};
    }

    std::vector<std::string> models_to_test;
    if (args.model == "all") {
        models_to_test = get_all_model_names();
    } else {
        models_to_test = {args.model};
    }

    for (const auto& model_name : models_to_test) {
        ModelInfo model_info = get_model_info(model_name);
        if (model_info.input_shape.empty()) {
            printf("Unknown model: %s\n", model_name.c_str());
            continue;
        }

        printf("\n========================================================\n");
        printf("Model: %s\n", model_name.c_str());
        printf("Input shape: ");
        for (int dim : model_info.input_shape) {
            printf("%d ", dim);
        }
        printf("\n========================================================\n");

        // Calculate input size
        size_t input_size = 1;
        for (int dim : model_info.input_shape) {
            input_size *= dim;
        }

        // Generate fixed input for all backends (same seed for reproducibility)
        std::vector<float> input(input_size);
        utils::fill_random_float(input.data(), input.size(), (unsigned int)42);  // Fixed seed

        // Step 1: Get reference output from ONNX Runtime
        std::vector<float> reference_output;
        bool has_reference = false;

        // Always attempt ORT reference; if ORT backend unavailable, skip silently
        {
            (void)backends_to_test;  // suppress unused-variable warning

            printf("\n--- [Step 1] Getting reference output from ONNX Runtime ---\n");

            BenchmarkConfig ort_config;
            ort_config.model_name = model_name;
            ort_config.model_path = model_info.get_model_path("onnxrt");
            ort_config.weights_path = model_info.get_weights_path("onnxrt");
            ort_config.input_shape = model_info.input_shape;
            ort_config.backend_type = BackendType::ONNXRT;
            ort_config.precision = parse_precision(args.precision);
            ort_config.num_threads = args.threads;
            ort_config.use_gpu = args.use_gpu;
            ort_config.warmup_runs = 1;
            ort_config.test_runs = 1;

            auto ort_backend = create_backend(BackendType::ONNXRT);
            if (ort_backend) {
                ort_backend->init(ort_config);
                if (ort_backend->infer_with_output(input, reference_output)) {
                    has_reference = true;
                    printf("  ✅ Reference output obtained (%zu elements)\n", reference_output.size());
                    printf("      Reference stats - Min: %.4f, Max: %.4f, Mean: %.4f\n",
                           *std::min_element(reference_output.begin(), reference_output.end()),
                           *std::max_element(reference_output.begin(), reference_output.end()),
                           std::accumulate(reference_output.begin(), reference_output.end(), 0.0) / reference_output.size());
                } else {
                    printf("  ⚠️  Failed to get reference output from ORT\n");
                }
                ort_backend->deinit();
            }
        }

        // Step 2: Test all backends
        printf("\n--- [Step 2] Running benchmarks ---\n");

        for (const auto& backend_name : backends_to_test) {
            printf("\n>> Testing %s on %s...\n", backend_name.c_str(), model_name.c_str());

            BenchmarkConfig config;
            config.model_name = model_name;
            config.model_path = model_info.get_model_path(backend_name);
            config.weights_path = model_info.get_weights_path(backend_name);
            config.input_shape = model_info.input_shape;
            config.backend_type = parse_backend(backend_name);
            config.precision = parse_precision(args.precision);
            config.num_threads = args.threads;
            config.use_gpu = args.use_gpu;
            config.warmup_runs = args.warmup;
            config.test_runs = args.runs;

            // Enable profiling if configured
            if (!args.profiling_file.empty()) {
                config.enable_profiling = true;
                // ONNX Runtime expects a file prefix, not a full filename
                // Strip .json extension if present (e.g., "profiling/ort_profile.json" -> "profiling/ort_profile")
                config.profile_file = args.profiling_file;
                if (config.profile_file.size() > 5 &&
                    config.profile_file.substr(config.profile_file.size() - 5) == ".json") {
                    config.profile_file = config.profile_file.substr(0, config.profile_file.size() - 5);
                }
            }

            auto backend = create_backend(config.backend_type);
            if (!backend) {
                printf("Failed to create backend: %s\n", backend_name.c_str());
                continue;
            }

            // Pass reference output for comparison (if available and not ORT itself)
            std::vector<float> ref_output;
            if (has_reference && backend_name != "onnxrt") {
                ref_output = reference_output;
            }

            BenchmarkResult result = run_benchmark(std::move(backend), config, input_size, ref_output);
            if (result.init_time_ms <= 0) {
                continue;
            }

            // Print performance results
            if (args.json_output) {
                printf("JSON_RESULT: %s\n", result.to_json().c_str());
            } else {
                printf("\n--- Performance Results ---\n");
                printf("  Init time:  %.2f ms\n", result.init_time_ms);
                utils::print_stats(result.latency_stats);
                printf("  Throughput: %.2f FPS\n", result.throughput_fps);
                printf("  Peak mem:   %zu KB\n", result.peak_memory_kb);
            }
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}
