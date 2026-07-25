#ifndef BENCHMARK_COMMON_CONFIG_H_
#define BENCHMARK_COMMON_CONFIG_H_

#include <string>
#include <vector>

enum class Precision {
    FP32,
    FP16,
    INT8,
};

enum class BackendType {
    MNN,
    ONNXRT,
    TVM,
    MNN_GPU,        // MNN OpenCL GPU 后端
    LLAMACPP,       // llama.cpp LLM 推理引擎 (GGUF)
};

struct BenchmarkConfig {
    // Model info
    std::string model_name;
    std::string model_path;
    std::string weights_path;
    std::vector<int> input_shape;
    std::string input_name;  // ONNX 模型第一个数据输入名

    // Backend config
    BackendType backend_type;
    Precision precision;
    int num_threads;
    bool use_gpu;

    // Test config
    int warmup_runs;
    int test_runs;

    // Framework-specific configuration
    bool enable_fp16;           // 是否启用 FP16 推理
    bool enable_int8;           // 是否启用 INT8 量化
    int intra_op_threads;       // ONNX Runtime 内部操作线程数
    int inter_op_threads;       // ONNX Runtime 间操作线程数

    // Memory management configuration
    size_t memory_limit_mb;     // 内存限制（MB）
    bool enable_memory_pool;    // 是否启用内存池

    // Statistical configuration
    int outlier_removal_count;  // 异常值移除数量（首尾）
    double confidence_level;    // 置信区间置信度（0.95, 0.99）

    // Profiling configuration
    bool enable_profiling;      // 是否启用逐算子 profiling
    std::string profile_file;   // profiling 输出文件路径

    BenchmarkConfig()
        : precision(Precision::FP32)
        , num_threads(1)
        , use_gpu(false)
        , warmup_runs(10)
        , test_runs(100)
        , enable_fp16(false)
        , enable_int8(false)
        , intra_op_threads(1)
        , inter_op_threads(1)
        , memory_limit_mb(0)
        , enable_memory_pool(false)
        , outlier_removal_count(0)
        , confidence_level(0.95)
        , enable_profiling(false)
        , profile_file("") {}
};

#endif // BENCHMARK_COMMON_CONFIG_H_
