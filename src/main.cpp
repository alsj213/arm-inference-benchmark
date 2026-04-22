#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>

#include "common/benchmark.h"
#include "common/utils.h"
#include "common/config.h"

#include "models/model_info.h"

void print_usage() {
    printf("Usage: ./benchmark_inference [options]\n");
    printf("Options:\n");
    printf("  --backend <ncnn|mnn|tnn|tflite|qnn|onnxrt|tvm>  Backend to test (default: all)\n");
    printf("  --model <mobilenetv2|resnet50|yolov8n|bert>  Model to test (default: all)\n");
    printf("  --precision <fp32|fp16|int8>                  Precision (default: fp32)\n");
    printf("  --threads <num>                                Number of threads (default: 1)\n");
    printf("  --warmup <num>                                 Number of warmup runs (default: 10)\n");
    printf("  --runs <num>                                   Number of test runs (default: 100)\n");
    printf("  --gpu                                          Use GPU if available\n");
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
        } else if (arg == "--help") {
            args.help = true;
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
    if (b == "ncnn" || b == "NCNN") return BackendType::NCNN;
    if (b == "mnn" || b == "MNN") return BackendType::MNN;
    if (b == "tnn" || b == "TNN") return BackendType::TNN;
    if (b == "tflite" || b == "TFLite" || b == "TFLITE") return BackendType::TFLITE;
    if (b == "qnn" || b == "QNN") return BackendType::QNN;
    if (b == "onnxrt" || b == "ort" || b == "ONNXRT" || b == "ORT") return BackendType::ONNXRT;
    if (b == "tvm" || b == "TVM") return BackendType::TVM;
    return (BackendType)-1;
}

ModelInfo get_model_info(const std::string& name) {
    if (name == "mobilenetv2") {
        return get_mobilenetv2_info();
    } else if (name == "resnet50") {
        return get_resnet50_info();
    } else if (name == "yolov8n") {
        return get_yolov8n_info();
    } else if (name == "bert") {
        return get_bert_info();
    }
    return {};
}

int main(int argc, char** argv) {
    CommandLineArgs args = parse_args(argc, argv);
    if (args.help) {
        print_usage();
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
        backends_to_test = {"ncnn", "mnn", "tnn", "tflite", "qnn", "onnxrt", "tvm"};
    } else {
        backends_to_test = {args.backend};
    }

    std::vector<std::string> models_to_test;
    if (args.model == "all") {
        models_to_test = {"mobilenetv2", "resnet50", "yolov8n", "bert"};
    } else {
        models_to_test = {args.model};
    }

    for (const auto& model_name : models_to_test) {
        ModelInfo model_info = get_model_info(model_name);
        if (model_info.input_shape.empty()) {
            printf("Unknown model: %s\n", model_name.c_str());
            continue;
        }

        printf("\n----------------------------------------\n");
        printf("Model: %s\n", model_name.c_str());
        printf("Input shape: ");
        for (int dim : model_info.input_shape) {
            printf("%d ", dim);
        }
        printf("\n");

        for (const auto& backend_name : backends_to_test) {
            printf("\n>> Running %s on %s...\n", backend_name.c_str(), model_name.c_str());

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

            size_t input_size = 1;
            for (int dim : config.input_shape) {
                input_size *= dim;
            }

            auto backend = create_backend(config.backend_type);
            if (!backend) {
                printf("Failed to create backend: %s\n", backend_name.c_str());
                continue;
            }

            BenchmarkResult result = run_benchmark(std::move(backend), config, input_size);
            if (result.init_time_ms <= 0) {
                continue;
            }

            printf("  Init time:  %.2f ms\n", result.init_time_ms);
            utils::print_stats(result.latency_stats);
            printf("  Throughput: %.2f FPS\n", result.throughput_fps);
            printf("  Peak mem:   %zu KB\n", result.peak_memory_kb);
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}
