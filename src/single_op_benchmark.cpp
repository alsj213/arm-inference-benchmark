#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "backends/ncnn_backend.h"
#include "backends/mnn_backend.h"
#include "backends/ort_backend.h"
#include "common/benchmark.h"
#include "common/utils.h"

void print_usage() {
    printf("Single Operator Benchmark Tool\n");
    printf("Usage: ./single_op_benchmark [options]\n");
    printf("Options:\n");
    printf("  --backend <ncnn|mnn|onnxrt>       Backend to test (required)\n");
    printf("  --model <path>                    Model path (required)\n");
    printf("  --input_shape <N,C,H,W|N,D>      Input shape (default: 1,3,224,224)\n");
    printf("  --warmup <num>                    Warmup runs (default: 10)\n");
    printf("  --runs <num>                      Test runs (default: 100)\n");
    printf("  --threads <num>                   Number of threads (default: 1)\n");
    printf("  --precision <fp32|fp16>           Precision (default: fp32)\n");
    printf("  --help                            Show this help\n");
}

std::vector<int> parse_shape(const std::string& s) {
    std::vector<int> shape;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        shape.push_back(std::atoi(item.c_str()));
    }
    return shape;
}

int main(int argc, char** argv) {
    std::string backend = "";
    std::string model_path = "";
    std::vector<int> input_shape = {1, 3, 224, 224};
    int warmup = 10;
    int runs = 100;
    int threads = 1;
    std::string precision = "fp32";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--input_shape" && i + 1 < argc) {
            input_shape = parse_shape(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--precision" && i + 1 < argc) {
            precision = argv[++i];
        } else if (arg == "--help") {
            print_usage();
            return 0;
        }
    }

    if (backend.empty() || model_path.empty()) {
        printf("Error: --backend and --model are required\n\n");
        print_usage();
        return 1;
    }

    printf("=== Single Operator Benchmark ===\n");
    printf("Backend:    %s\n", backend.c_str());
    printf("Model:      %s\n", model_path.c_str());
    printf("Input:      ");
    for (int d : input_shape) printf("%d ", d);
    printf("\n");
    printf("Warmup:     %d\n", warmup);
    printf("Runs:       %d\n", runs);
    printf("Threads:    %d\n", threads);
    printf("Precision:  %s\n", precision.c_str());
    printf("\n");

    BenchmarkConfig config;
    config.model_name = "single_op";
    config.input_shape = input_shape;
    config.num_threads = threads;
    config.use_gpu = false;
    config.precision = (precision == "fp16") ? Precision::FP16 : Precision::FP32;

    if (backend == "ncnn") {
        config.model_path = model_path + ".param";
        config.weights_path = model_path + ".bin";
    } else {
        config.model_path = model_path;
    }

    std::unique_ptr<BenchmarkBackend> infer;
    if (backend == "ncnn") {
        infer = std::make_unique<NCNNBackend>();
    } else if (backend == "mnn") {
        infer = std::make_unique<MNNBackend>();
    } else if (backend == "onnxrt" || backend == "ort") {
        infer = std::make_unique<ONNXRTBackend>();
    } else {
        printf("Error: Unknown backend %s\n", backend.c_str());
        return 1;
    }

    if (!infer->init(config)) {
        printf("Error: Failed to initialize %s backend\n", backend.c_str());
        return 1;
    }

    // Prepare input
    size_t input_size = 1;
    for (int d : input_shape) input_size *= d;
    std::vector<float> input(input_size, 0.5f);

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < warmup; ++i) {
        infer->infer(input);
    }

    // Benchmark
    printf("Benchmarking...\n");
    std::vector<double> times;
    times.reserve(runs);

    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        infer->infer(input);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    utils::Stats stats = utils::calculate_stats(times);
    printf("\n");
    printf("mean=%.3fms, min=%.3fms, max=%.3fms, std=%.3fms\n",
           stats.mean_ms, stats.min_ms, stats.max_ms, stats.std_dev);

    infer->deinit();

    return 0;
}
