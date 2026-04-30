#include "ort_backend.h"

#include "onnxruntime_cxx_api.h"
#include <vector>
#include <string>

bool ONNXRTBackend::init(const BenchmarkConfig& config) {
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(config.num_threads);
    session_options_->SetInterOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

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

    // Get input/output names - ORT v1.21.0 API (GetInputNameAllocated/GetOutputNameAllocated)
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_.GetInputCount();
    input_names_store_.resize(num_inputs);
    input_names_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator);
        input_names_store_[i] = name.get();
        input_names_[i] = input_names_store_[i].c_str();
        printf("ONNXRT: input %zu = %s\n", i, input_names_[i]);
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

bool ONNXRTBackend::infer(const std::vector<float>& input) {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(input.data()), input.size(),
        input_shape_.data(), input_shape_.size()));

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
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(input.data()), input.size(),
        input_shape_.data(), input_shape_.size()));

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
    if (session_) {
        session_.release();
    }
    session_options_.reset();
    input_names_store_.clear();
    output_names_store_.clear();
    input_names_.clear();
    output_names_.clear();
}