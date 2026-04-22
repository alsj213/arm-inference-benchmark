#include "ncnn_backend.h"

#include "cstring"
#include "vector"

bool NCNNBackend::init(const BenchmarkConfig& config) {
    net_.clear();

    int use_fp16 = (config.precision == Precision::FP16) ? 1 : 0;
    net_.opt.use_fp16_packed = use_fp16;
    net_.opt.use_fp16_storage = use_fp16;
    net_.opt.use_fp16_arithmetic = use_fp16;
    net_.opt.use_int8_inference = (config.precision == Precision::INT8);
    net_.opt.use_winograd_convolution = true;
    net_.opt.use_sgemm_convolution = true;
    net_.opt.use_int8_storage = false;
    net_.opt.use_int8_arithmetic = false;
    net_.opt.use_packing_layout = true;
    net_.opt.lightmode = true;

    // Set up memory allocators
    net_.opt.blob_allocator = &blob_allocator_;
    net_.opt.workspace_allocator = &workspace_allocator_;

    // ncnn handles thread setting via option
    net_.opt.num_threads = config.num_threads;

    // Load model
    int ret = net_.load_param(config.model_path.c_str());
    if (ret != 0) {
        printf("Failed to load param: %s\n", config.model_path.c_str());
        return false;
    }
    ret = net_.load_model(config.weights_path.c_str());
    if (ret != 0) {
        printf("Failed to load model: %s\n", config.weights_path.c_str());
        return false;
    }

    // Get input and output names
    const std::vector<const char*>& input_names = net_.input_names();
    const std::vector<const char*>& output_names = net_.output_names();
    if (!input_names.empty()) {
        input_name_ = input_names[0];
    }
    if (!output_names.empty()) {
        output_name_ = output_names[0];
    }

    // Create input mat - ncnn uses (w, h, c) for 3D
    if (config.input_shape.size() == 4) {
        // config.input_shape is [1, 3, 224, 224] = [N, C, H, W]
        // ncnn Mat needs (w=224, h=224, c=3), elemsize=4 for float
        input_.create(config.input_shape[3], config.input_shape[2], config.input_shape[1], (size_t)4, (ncnn::Allocator*)nullptr);
    } else if (config.input_shape.size() == 2) {
        // [N, seq_len]
        input_.create(config.input_shape[1], (size_t)4, 1, (ncnn::Allocator*)nullptr);
    }

    return true;
}

bool NCNNBackend::infer(const std::vector<float>& input) {
    // Copy data to input mat
    memcpy(input_.data, input.data(), input.size() * sizeof(float));

    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);

    // Use input name
    int ret = ex.input(input_name_.c_str(), input_);
    if (ret != 0) {
        printf("ncnn: ex.input failed\n");
        return false;
    }

    ncnn::Mat output;
    // Use output name
    ret = ex.extract(output_name_.c_str(), output);
    if (ret != 0) {
        printf("ncnn: ex.extract failed\n");
        return false;
    }

    return ret == 0;
}

void NCNNBackend::deinit() {
    net_.clear();
}