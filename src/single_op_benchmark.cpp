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
#include <unistd.h>
#include <map>

#include "backends/mnn_backend.h"
#include "backends/ort_backend.h"
#include "backends/tvm_backend.h"
#include "common/benchmark.h"
#include "common/utils.h"

// ── 算子分类 → 模型路径映射表 ──
// 路径指向 models/single_ops/ 下已有的 ONNX 模型
// 由 scripts/generate_single_ops.py (PyTorch 导出) 或
//   scripts/generate_single_operator_models.py (ONNX raw API) 生成
static const std::map<std::string, std::vector<std::string>> CATEGORY_MODELS = {
    // ═══════ Conv1x1: 网络提取 + 形状分桶（15） ═══════
    {"conv1x1", {
        "models/single_ops/Conv1x1_K16_C64_M784.onnx",
        "models/single_ops/Conv1x1_K1024_C256_M784.onnx",
        "models/single_ops/Conv1x1_M49_C32_K64.onnx",
        "models/single_ops/Conv1x1_M49_C256_K512.onnx",
        "models/single_ops/Conv1x1_M784_C32_K64.onnx",
        "models/single_ops/Conv1x1_M3136_C64_K128.onnx",
    }},

    // ═══════ Conv1x1 特化路径：C 对齐退化 ═══════
    {"conv1x1_misaligned", {
        "models/single_ops/Conv1x1_Misaligned_C31_K64.onnx",
        "models/single_ops/Conv1x1_Misaligned_C33_K64.onnx",
    }},

    // ═══════ DWConv（8） ═══════
    {"dwconv", {
        "models/single_ops/DWConv_C16_3x3.onnx",
        "models/single_ops/DWConv_C960_3x3.onnx",
    }},

    // ═══════ MatMul（15） ═══════
    {"matmul", {
        "models/single_ops/MatMul_512x512x512.onnx",
        "models/single_ops/MatMul_768x768x768.onnx",
        "models/single_ops/MatMul_3072x768.onnx",
        "models/single_ops/MatMul_768x3072.onnx",
    }},

    // ═══════ LayerNorm ═══════
    {"layernorm", {
        "models/single_ops/LayerNorm.onnx",
    }},

    // ═══════ Softmax ═══════
    {"softmax", {
        "models/single_ops/Softmax.onnx",
    }},

    // ═══════ GELU ═══════
    {"gelu", {
        "models/single_ops/GELU.onnx",
    }},
};

void print_usage() {
    printf("Single Operator Benchmark Tool\n");
    printf("Usage: ./single_op_benchmark [options]\n");
    printf("Options:\n");
    printf("  --backend <mnn|onnxrt|ort|tvm>  Backend to test (required)\n");
    printf("  --model <path>                  Model path (mutually exclusive with --category)\n");
    printf("  --category <name>               Operator category (");
    bool first = true;
    for (const auto& kv : CATEGORY_MODELS) {
        if (!first) printf("|");
        printf("%s", kv.first.c_str());
        first = false;
    }
    printf(")\n");
    printf("  --input_shape <N,C,H,W|N,D>    Input shape (default: 1,3,224,224)\n");
    printf("  --warmup <num>                  Warmup runs (default: 10)\n");
    printf("  --runs <num>                    Test runs (default: 100)\n");
    printf("  --threads <num>                 Number of threads (default: 1)\n");
    printf("  --precision <fp32|fp16>         Precision (default: fp32)\n");
    printf("  --help                          Show this help\n");
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

BackendType parse_backend(const std::string& b) {
    if (b == "mnn" || b == "MNN") return BackendType::MNN;
    if (b == "onnxrt" || b == "ort" || b == "ORT") return BackendType::ONNXRT;
    if (b == "tvm" || b == "TVM") return BackendType::TVM;
    if (b == "llamacpp" || b == "llama") return BackendType::LLAMACPP;
    return BackendType::MNN;  // fallback
}

// 从模型文件名中提取正确的 input_shape
static std::vector<int> parse_shape_from_name(const std::string& fname) {
    // MatMul (Linear → 2D input)
    if (fname.find("MatMul_512x512x512") != std::string::npos) return {1, 512};
    if (fname.find("MatMul_768x768x768") != std::string::npos) return {1, 768};
    if (fname.find("MatMul_768x3072") != std::string::npos)     return {1, 768};
    if (fname.find("MatMul_3072x768") != std::string::npos)     return {1, 3072};

    // DWConv: 通道数决定空间尺寸
    if (fname.find("DWConv_C16_3x3") != std::string::npos)  return {1, 16, 112, 112};
    if (fname.find("DWConv_C960_3x3") != std::string::npos) return {1, 960, 7, 7};

    // Misaligned: C=31/33, spatial=56
    if (fname.find("Misaligned_C31") != std::string::npos) return {1, 31, 56, 56};
    if (fname.find("Misaligned_C33") != std::string::npos) return {1, 33, 56, 56};

    // Conv1x1: 从 _M 和 _C 字段解析
    int C = 64, M = 784;
    size_t cp = fname.rfind("_C");
    if (cp != std::string::npos) {
        size_t ce = fname.find_first_of("._", cp + 2);
        C = std::stoi(fname.substr(cp + 2, ce - cp - 2));
    }
    size_t mp = fname.rfind("_M");
    if (mp != std::string::npos) {
        size_t me = fname.find_first_of("._", mp + 2);
        M = std::stoi(fname.substr(mp + 2, me - mp - 2));
    }
    int HW = (int)std::sqrt(M);
    return {1, C, HW, HW};
}

static bool run_single_op(const std::string& backend_name, const std::string& model_path,
                          const std::vector<int>& input_shape,
                          int warmup, int runs, int threads, const std::string& precision) {
    printf("  [%s] %s\n", backend_name.c_str(), model_path.c_str());
    fflush(stdout);

    if (access(model_path.c_str(), F_OK) != 0) {
        printf("  ⚠️  Model not found, skipping\n");
        fflush(stdout);
        return false;
    }

    BenchmarkConfig config;
    config.model_name = "single_op";
    config.model_path = model_path;
    config.input_shape = input_shape;
    config.num_threads = threads;
    config.use_gpu = false;
    config.precision = (precision == "fp16") ? Precision::FP16 : Precision::FP32;

    // MNN 后端需要 .mnn 文件，自动替换扩展名
    std::string actual_model_path = model_path;
    BackendType bt = parse_backend(backend_name);
    if (bt == BackendType::MNN) {
        size_t pos = actual_model_path.rfind(".onnx");
        if (pos != std::string::npos) {
            actual_model_path.replace(pos, 5, ".mnn");
        }
    }
    config.model_path = actual_model_path;

    std::unique_ptr<BenchmarkBackend> infer;
    switch (bt) {
        case BackendType::MNN:
            infer = std::make_unique<MNNBackend>();
            break;
        case BackendType::ONNXRT:
            infer = std::make_unique<ONNXRTBackend>();
            break;
        case BackendType::TVM:
            infer = std::make_unique<TVMBackend>();
            break;
        default:
            printf("  ❌ Unknown backend %s\n", backend_name.c_str());
            return false;
    }

    if (!infer->init(config)) {
        printf("  ❌ init failed\n");
        return false;
    }

    // Prepare input
    size_t input_size = 1;
    for (int d : input_shape) input_size *= d;
    std::vector<float> input(input_size, 0.5f);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        infer->infer(input);
    }

    // Benchmark
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        infer->infer(input);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    infer->deinit();

    utils::Stats stats = utils::calculate_stats(times);
    printf("  mean=%.3fms min=%.3fms max=%.3fms std=%.3fms\n",
           stats.mean_ms, stats.min_ms, stats.max_ms, stats.std_dev);
    fflush(stdout);
    return true;
}

int main(int argc, char** argv) {
    std::string backend = "";
    std::string model_path = "";
    std::string category = "";
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
        } else if (arg == "--category" && i + 1 < argc) {
            category = argv[++i];
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

    if (backend.empty()) {
        printf("Error: --backend is required\n\n");
        print_usage();
        return 1;
    }

    printf("=== Single Operator Benchmark ===\n");
    printf("Backend:    %s\n", backend.c_str());
    printf("Threads:    %d\n", threads);
    printf("Runs:       %d\n\n", runs);

    int total_ok = 0, total_fail = 0;

    // ── 分类模式：批量执行某个类别的所有算子 ──
    if (!category.empty()) {
        auto it = CATEGORY_MODELS.find(category);
        if (it == CATEGORY_MODELS.end()) {
            printf("Error: Unknown category '%s'\n", category.c_str());
            return 1;
        }
        printf("Category: %s (%zu models)\n\n", category.c_str(), it->second.size());
        for (const auto& mp : it->second) {
            auto shape = parse_shape_from_name(mp);
            printf("  [auto shape: ");
            for (size_t i = 0; i < shape.size(); i++) {
                if (i) printf(",");
                printf("%d", shape[i]);
            }
            printf("] ");
            if (run_single_op(backend, mp, shape, warmup, runs, threads, precision))
                total_ok++;
            else
                total_fail++;
        }
        printf("\nCategory done: %d OK, %d skipped/failed\n", total_ok, total_fail);
        return total_fail > 0 ? 1 : 0;
    }

    // ── 单模型模式 ──
    if (model_path.empty()) {
        printf("Error: --model or --category is required\n\n");
        print_usage();
        return 1;
    }
    run_single_op(backend, model_path, input_shape, warmup, runs, threads, precision);
    return 0;
}
