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
        "models/single_ops/basic/Conv1x1_K16_C64_M784.onnx",
        "models/single_ops/basic/Conv1x1_K1024_C256_M784.onnx",
        "models/single_ops/basic/Conv1x1_M49_C32_K64.onnx",
        "models/single_ops/basic/Conv1x1_M49_C256_K512.onnx",
        "models/single_ops/basic/Conv1x1_M784_C32_K64.onnx",
        "models/single_ops/basic/Conv1x1_M3136_C64_K128.onnx",
    }},

    // ═══════ Conv1x1 特化路径：C 对齐退化 ═══════
    {"conv1x1_misaligned", {
        "models/single_ops/basic/Conv1x1_Misaligned_C31_K64.onnx",
        "models/single_ops/basic/Conv1x1_Misaligned_C33_K64.onnx",
    }},

    // ═══════ DWConv（8） ═══════
    {"dwconv", {
        "models/single_ops/basic/DWConv_C16_3x3.onnx",
        "models/single_ops/basic/DWConv_C960_3x3.onnx",
    }},

    // ═══════ MatMul（15） ═══════
    {"matmul", {
        "models/single_ops/basic/MatMul_512x512x512.onnx",
        "models/single_ops/basic/MatMul_768x768x768.onnx",
        "models/single_ops/basic/MatMul_3072x768.onnx",
        "models/single_ops/basic/MatMul_768x3072.onnx",
    }},

    // ═══════ LayerNorm (BERT: [1,128,768] normalized_shape=768) ═══════
    {"layernorm", {
        "models/single_ops/basic/LayerNorm_BERT.onnx",
    }},

    // ═══════ Softmax (BERT attention: [1,128,128] dim=-1) ═══════
    {"softmax", {
        "models/single_ops/basic/Softmax_BERT.onnx",
    }},

    // ═══════ GELU (BERT FFN: [1,128,3072]) ═══════
    {"gelu", {
        "models/single_ops/basic/GELU_BERT.onnx",
    }},

    // ═══════ GEMM Pack/Unpack 分析（24 算子）═══════
    {"gemm", {
        "models/single_ops_gemm/GEMM_Square_16x16.onnx",
        "models/single_ops_gemm/GEMM_Square_32x32.onnx",
        "models/single_ops_gemm/GEMM_Square_64x64.onnx",
        "models/single_ops_gemm/GEMM_Square_128x128.onnx",
        "models/single_ops_gemm/GEMM_Square_256x256.onnx",
        "models/single_ops_gemm/GEMM_Square_512x512.onnx",
        "models/single_ops_gemm/GEMM_Square_1024x1024.onnx",
        "models/single_ops_gemm/GEMM_KScan_K16_M256_N256.onnx",
        "models/single_ops_gemm/GEMM_KScan_K32_M256_N256.onnx",
        "models/single_ops_gemm/GEMM_KScan_K64_M256_N256.onnx",
        "models/single_ops_gemm/GEMM_KScan_K128_M256_N256.onnx",
        "models/single_ops_gemm/GEMM_KScan_K512_M256_N256.onnx",
        "models/single_ops_gemm/GEMM_KScan_K1024_M256_N256.onnx",
        "models/single_ops_gemm/GEMM_MNScan_M16_K256_N16.onnx",
        "models/single_ops_gemm/GEMM_MNScan_M32_K256_N32.onnx",
        "models/single_ops_gemm/GEMM_MNScan_M64_K256_N64.onnx",
        "models/single_ops_gemm/GEMM_MNScan_M128_K256_N128.onnx",
        "models/single_ops_gemm/GEMM_MNScan_M512_K256_N512.onnx",
        "models/single_ops_gemm/GEMM_MNScan_M1024_K256_N1024.onnx",
        "models/single_ops_gemm/GEMM_BERT_FFN1_S128_H768_I3072.onnx",
        "models/single_ops_gemm/GEMM_BERT_Proj_S128_H3072_O768.onnx",
        "models/single_ops_gemm/GEMM_Attention_128x768x128.onnx",
        "models/single_ops_gemm/GEMM_LLM_FFN_1x4096x14336.onnx",
        "models/single_ops_gemm/GEMM_LLM_AttnProj_1x4096x4096.onnx",
    }},
};

void print_usage() {
    printf("Single Operator Benchmark Tool\n");
    printf("Usage: ./single_op_benchmark [options]\n");
    printf("Options:\n");
    printf("  --backend <mnn|mnn_gpu|onnxrt|ort|tvm>  Backend to test (required)\n");
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
    if (b == "mnn_gpu" || b == "MNN_GPU") return BackendType::MNN_GPU;
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

    // NLP ops (BERT-specific shapes)
    if (fname.find("LayerNorm_BERT") != std::string::npos) return {1, 128, 768};
    if (fname.find("Softmax_BERT") != std::string::npos)    return {1, 128, 128};
    if (fname.find("GELU_BERT") != std::string::npos)       return {1, 128, 3072};

    // DWConv: 通道数决定空间尺寸
    if (fname.find("DWConv_C16_3x3") != std::string::npos)  return {1, 16, 112, 112};
    if (fname.find("DWConv_C960_3x3") != std::string::npos) return {1, 960, 7, 7};

    // Misaligned: C=31/33, spatial=56
    if (fname.find("Misaligned_C31") != std::string::npos) return {1, 31, 56, 56};
    if (fname.find("Misaligned_C33") != std::string::npos) return {1, 33, 56, 56};

// ── GEMM 算子: 通用模式解析 ──
// 所有 GEMM_ 前缀的模型文件名包含 M/K/N 维度信息
// 注意: 使用 rfind (反向查找) 避免 _KScan 比 _K16 先匹配
{
    // 尝试通用解析: _M{num}_K{num}_N{num} 模式
    // 只匹配 _M 后面紧跟数字的情况 (MScan → 跳过)
    size_t mp = std::string::npos, kp = std::string::npos;
    for (size_t i = 0; i + 2 < fname.size(); i++) {
        if (fname[i] == '_' && fname[i+1] == 'M' && i+2 < fname.size() && isdigit(fname[i+2])) {
            mp = i;
        }
        if (fname[i] == '_' && fname[i+1] == 'K' && i+2 < fname.size() && isdigit(fname[i+2])) {
            kp = i;
        }
    }
    if (mp != std::string::npos && kp != std::string::npos) {
        int Mval = 0, Kval = 0;
        try {
            size_t me = fname.find_first_of("_x.", mp + 2);
            Mval = std::stoi(fname.substr(mp + 2, me - mp - 2));
            size_t ke = fname.find_first_of("_x.", kp + 2);
            Kval = std::stoi(fname.substr(kp + 2, ke - kp - 2));
        } catch (...) {}
        if (Mval > 0 && Kval > 0) return {Mval, Kval};
    }
}
{
    // Square 模式: GEMM_Square_{D}x{D}
    size_t sp = fname.find("_Square_");
    if (sp != std::string::npos) {
        std::string tail = fname.substr(sp + 8);
        size_t xp = tail.find('x');
        if (xp != std::string::npos) {
            try {
                int D = std::stoi(tail.substr(0, xp));
                return {D, D};
            } catch (...) {}
        }
    }
}
{
    // BERT/LLM 模式: 尝试从命名提取 M 和 K
    // GEMM_Attention_{M}x{K}x{N}
    size_t ap = fname.find("_Attention_");
    if (ap != std::string::npos) {
        std::string tail = fname.substr(ap + 11);
        size_t x1 = tail.find('x');
        size_t x2 = tail.find('x', x1 + 1);
        if (x1 != std::string::npos && x2 != std::string::npos) {
            try {
                int Mval = std::stoi(tail.substr(0, x1));
                int Kval = std::stoi(tail.substr(x1 + 1, x2 - x1 - 1));
                return {Mval, Kval};
            } catch (...) {}
        }
    }
    // GEMM_BERT_FFN1_S{seq}_H{hidden}_I{inter}
    size_t bp = fname.find("_BERT_FFN");
    if (bp != std::string::npos) {
        size_t sp = fname.find("_S", bp);
        size_t hp = fname.find("_H", bp);
        if (sp != std::string::npos && hp != std::string::npos) {
            try {
                size_t se = fname.find("_", sp + 2);
                int Mval = std::stoi(fname.substr(sp + 2, se - sp - 2));
                size_t he = fname.find("_", hp + 2);
                int Kval = std::stoi(fname.substr(hp + 2, he - hp - 2));
                return {Mval, Kval};
            } catch (...) {}
        }
    }
    // GEMM_BERT_Proj 类似
    if (fname.find("_BERT_Proj") != std::string::npos) {
        size_t sp = fname.find("_S");
        size_t hp = fname.find("_H");
        if (sp != std::string::npos && hp != std::string::npos) {
            try {
                size_t se = fname.find("_", sp + 2);
                int Mval = std::stoi(fname.substr(sp + 2, se - sp - 2));
                size_t he = fname.find("_", hp + 2);
                int Kval = std::stoi(fname.substr(hp + 2, he - hp - 2));
                return {Mval, Kval};
            } catch (...) {}
        }
    }
    // GEMM_LLM_FFN_1x{K}x{N} / GEMM_LLM_AttnProj_1x{K}x{N}
    size_t lp = fname.find("_LLM_");
    if (lp != std::string::npos) {
        std::string tail = fname.substr(lp + 5);  // FFN_1x4096x14336 or AttnProj_1x4096x4096
        size_t us = tail.find('_');
        if (us != std::string::npos) {
            std::string dims = tail.substr(us + 1);  // 1x4096x14336
            size_t x1 = dims.find('x');
            size_t x2 = dims.find('x', x1 + 1);
            if (x1 != std::string::npos && x2 != std::string::npos) {
                try {
                    int Mval = std::stoi(dims.substr(0, x1));
                    int Kval = std::stoi(dims.substr(x1 + 1, x2 - x1 - 1));
                    return {Mval, Kval};
                } catch (...) {}
            }
        }
    }
}

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

    // 后端的路径解析（先替换扩展名，再检查文件）
    std::string actual_model_path = model_path;
    BackendType bt = parse_backend(backend_name);

    if (bt == BackendType::MNN || bt == BackendType::MNN_GPU) {
        size_t pos = actual_model_path.rfind(".onnx");
        if (pos != std::string::npos) {
            actual_model_path.replace(pos, 5, ".mnn");
        }
    }
    // TVM 后端需要 tvm_models/ 下的 .so 文件
    if (bt == BackendType::TVM) {
        size_t last_slash = actual_model_path.rfind('/');
        std::string fname = (last_slash != std::string::npos)
            ? actual_model_path.substr(last_slash + 1)
            : actual_model_path;
        size_t dot = fname.rfind(".onnx");
        if (dot != std::string::npos) {
            fname = fname.substr(0, dot);
        }
        actual_model_path = "tvm_models/" + fname + "_tvm.so";
    }

    if (access(actual_model_path.c_str(), F_OK) != 0) {
        printf("  ⚠️  Model not found: %s\n", actual_model_path.c_str());
        fflush(stdout);
        return false;
    }

    BenchmarkConfig config;
    config.model_name = "single_op";
    config.model_path = actual_model_path;
    config.input_shape = input_shape;
    config.num_threads = threads;
    config.use_gpu = false;
    config.precision = (precision == "fp16") ? Precision::FP16 : Precision::FP32;

    if (bt == BackendType::MNN_GPU) {
        config.use_gpu = true;
        config.backend_type = BackendType::MNN_GPU;
    }

    std::unique_ptr<BenchmarkBackend> infer;
    switch (bt) {
        case BackendType::MNN:
        case BackendType::MNN_GPU:
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
