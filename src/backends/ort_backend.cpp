#include "ort_backend.h"

#include "onnxruntime_cxx_api.h"
#include <vector>
#include <string>
#include <cstdint>

bool ONNXRTBackend::init(const BenchmarkConfig& config) {
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(config.num_threads);
    session_options_->SetInterOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable profiling if configured
    if (config.enable_profiling && !config.profile_file.empty()) {
        session_options_->EnableProfiling(config.profile_file.c_str());
        profiling_enabled_ = true;
        printf("ONNXRT: Profiling enabled, output prefix: %s\n", config.profile_file.c_str());
        printf("ONNXRT: Profiling will generate: %s_<timestamp>.json\n", config.profile_file.c_str());
    }

    printf("ONNXRT: loading model: %s\n", config.model_path.c_str());
    session_ = Ort::Session(env_, config.model_path.c_str(), *session_options_);
    if (!session_) {
        printf("ONNXRT: Failed to create session\n");
        return false;
    }

    // Store input shape
    input_shape_.resize(config.input_shape.size());
    for (size_t i = 0; i < config.input_shape.size(); ++i) {
        input_shape_[i] = config.input_shape[i];
    }

    // Get input info
    Ort::AllocatorWithDefaultOptions allocator;
    num_inputs_ = session_.GetInputCount();
    input_names_store_.resize(num_inputs_);
    input_names_.resize(num_inputs_);
    input_types_.resize(num_inputs_);
    for (size_t i = 0; i < num_inputs_; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator);
        input_names_store_[i] = name.get();
        input_names_[i] = input_names_store_[i].c_str();
        auto type_info = session_.GetInputTypeInfo(i);
        input_types_[i] = type_info.GetTensorTypeAndShapeInfo().GetElementType();
        printf("ONNXRT: input %zu = %s (type %d)\n", i, input_names_[i], (int)input_types_[i]);
    }

    size_t num_outputs = session_.GetOutputCount();
    output_names_store_.resize(num_outputs);
    output_names_.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator);
        output_names_store_[i] = name.get();
        output_names_[i] = output_names_store_[i].c_str();
        printf("ONNXRT: output %zu = %s\n", i, output_names_[i]);
    }

    return true;
}

void ONNXRTBackend::create_input_tensors(const std::vector<float>& input,
                                         std::vector<Ort::Value>& input_tensors,
                                         std::vector<std::vector<int64_t>>& scratch) {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    scratch.resize(num_inputs_);

    for (size_t i = 0; i < num_inputs_; ++i) {
        if (input_types_[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            // Generate int64 tensor from the float input
            scratch[i].resize(input.size());
            for (size_t j = 0; j < input.size(); ++j) {
                int64_t val = static_cast<int64_t>(input[j] * 1000.0f) % 30521;
                if (val < 0) val = -val;
                if (val == 0) val = 101;  // [CLS] token
                scratch[i][j] = val;
            }

            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                mem_info, scratch[i].data(), scratch[i].size(),
                input_shape_.data(), input_shape_.size()));
        } else {
            // Float tensor (or other, default to float)
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                mem_info, const_cast<float*>(input.data()), input.size(),
                input_shape_.data(), input_shape_.size()));
        }
    }
}

bool ONNXRTBackend::infer(const std::vector<float>& input) {
    std::vector<Ort::Value> input_tensors;
    std::vector<std::vector<int64_t>> scratch;
    create_input_tensors(input, input_tensors, scratch);

    std::vector<Ort::Value> output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names_.data(),
        output_names_.size());

    return output_tensors.size() > 0;
}

bool ONNXRTBackend::infer_with_output(const std::vector<float>& input, std::vector<float>& output) {
    std::vector<Ort::Value> input_tensors;
    std::vector<std::vector<int64_t>> scratch;
    create_input_tensors(input, input_tensors, scratch);

    std::vector<Ort::Value> output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names_.data(),
        output_names_.size());

    if (output_tensors.empty()) {
        return false;
    }

    // Get output tensor data
    auto& output_tensor = output_tensors[0];
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    output.resize(output_size);
    const float* output_data = output_tensor.GetTensorData<float>();
    std::copy(output_data, output_data + output_size, output.data());

    return true;
}

void ONNXRTBackend::deinit() {
    if (profiling_enabled_) {
        printf("ONNXRT: Profiling enabled, session will write profiling data on destruction\n");
    }

    session_options_.reset();
    input_names_store_.clear();
    output_names_store_.clear();
    input_names_.clear();
    output_names_.clear();
    input_types_.clear();
}
