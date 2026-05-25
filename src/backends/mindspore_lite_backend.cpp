#include "mindspore_lite_backend.h"

#ifdef BENCHMARK_MINDSPORE_LITE
#include <cstring>

bool MindSporeLiteBackend::init(const BenchmarkConfig& config) {
    context_ = std::make_shared<mindspore::Context>();
    auto& device_list = context_->MutableDeviceInfo();

    // CPU device
    auto cpu_info = std::make_shared<mindspore::CPUDeviceInfo>();
    cpu_info->SetEnableFP16(config.precision == Precision::FP16);
    device_list.push_back(cpu_info);

    // Set threads
    context_->SetThreadNum(config.num_threads);
    context_->SetThreadAffinity(0);  // 0 = no affinity

    // Load model
    model_ = std::make_shared<mindspore::Model>();
    auto status = model_->Build(config.model_path, mindspore::kMindIR_Lite, context_);
    if (status != mindspore::kSuccess) {
        printf("MindSpore Lite: Failed to build model: %d\n", static_cast<int>(status));
        return false;
    }

    // Get input info
    inputs_ = model_->GetInputs();
    if (inputs_.empty()) {
        printf("MindSpore Lite: No inputs found\n");
        return false;
    }
    input_name_ = inputs_[0].Name();

    return true;
}

bool MindSporeLiteBackend::infer(const std::vector<float>& input) {
    // Set input tensor data
    auto& in_tensor = inputs_[0];
    in_tensor.SetData(const_cast<float*>(input.data()));
    in_tensor.SetElementNum(input.size());

    auto status = model_->Predict(inputs_, &outputs_);
    return status == mindspore::kSuccess;
}

bool MindSporeLiteBackend::infer_with_output(const std::vector<float>& input, std::vector<float>& output) {
    auto& in_tensor = inputs_[0];
    in_tensor.SetData(const_cast<float*>(input.data()));
    in_tensor.SetElementNum(input.size());

    auto status = model_->Predict(inputs_, &outputs_);
    if (status != mindspore::kSuccess) return false;

    if (!outputs_.empty()) {
        auto* data = static_cast<float*>(outputs_[0].MutableData());
        output.assign(data, data + outputs_[0].ElementNum());
    }
    return true;
}

void MindSporeLiteBackend::deinit() {
    model_.reset();
    context_.reset();
    inputs_.clear();
    outputs_.clear();
}

#else
// Stub implementations when SDK not available
bool MindSporeLiteBackend::init(const BenchmarkConfig&) { return false; }
bool MindSporeLiteBackend::infer(const std::vector<float>&) { return false; }
bool MindSporeLiteBackend::infer_with_output(const std::vector<float>&, std::vector<float>&) { return false; }
void MindSporeLiteBackend::deinit() {}
#endif
