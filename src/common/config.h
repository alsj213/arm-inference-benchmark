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
    NCNN,
    MNN,
    TNN,
    TFLITE,
    QNN,
    ONNXRT,
    TVM,
};

struct BenchmarkConfig {
    // Model info
    std::string model_name;
    std::string model_path;
    std::string weights_path;
    std::vector<int> input_shape;

    // Backend config
    BackendType backend_type;
    Precision precision;
    int num_threads;
    bool use_gpu;

    // Test config
    int warmup_runs;
    int test_runs;

    BenchmarkConfig()
        : precision(Precision::FP32)
        , num_threads(1)
        , use_gpu(false)
        , warmup_runs(10)
        , test_runs(100) {}
};

#endif // BENCHMARK_COMMON_CONFIG_H_
