#include "mindspore_lite_backend.h"

#ifdef BENCHMARK_MINDSPORE_LITE
#include <fstream>
#include <cstring>
#include <memory>

bool MindSporeLiteBackend::init(const BenchmarkConfig& config) {
    // Store input shape
    input_shape_.clear();
    for (int dim : config.input_shape) {
        input_shape_.push_back(static_cast<int64_t>(dim));
    }

    // Create context
    context_ = std::make_shared<mindspore::Context>();
    auto& device_list = context_->MutableDeviceInfo();

    auto cpu_info = std::make_shared<mindspore::CPUDeviceInfo>();
    cpu_info->SetEnableFP16(config.precision == Precision::FP16);
    device_list.push_back(cpu_info);

    context_->SetThreadNum(config.num_threads);
    context_->SetThreadAffinity(0);

    // Build model from file path
    model_ = std::make_shared<mindspore::Model>();
    auto status = model_->Build(config.model_path, mindspore::kMindIR_Lite, context_);
    if (status != mindspore::kSuccess) {
        printf("MindSpore Lite: Build failed (path=%s, err=%d)\n",
               config.model_path.c_str(), static_cast<int>(status.StatusCode()));
        return false;
    }

    // Get actual input name from model
    auto model_inputs = model_->GetInputs();
    if (model_inputs.empty()) {
        printf("MindSpore Lite: No inputs found in model\n");
        return false;
    }
    input_name_ = model_inputs[0].Name();
    printf("MindSpore Lite: input='%s' shape=", input_name_.c_str());
    for (auto s : model_inputs[0].Shape()) printf("%lld ", (long long)s);
    printf("\n");

    // Resize model inputs to the concrete batch size from config
    auto resize_ret = model_->Resize(model_inputs, {input_shape_});
    if (resize_ret != mindspore::kSuccess) {
        printf("MindSpore Lite: Resize failed code=%d\n", static_cast<int>(resize_ret.StatusCode()));
        return false;
    }
    printf("MindSpore Lite: resized to [");
    auto resized = model_->GetInputs();
    for (size_t i = 0; i < resized[0].Shape().size(); ++i) {
        if (i > 0) printf(", ");
        printf("%lld", (long long)resized[0].Shape()[i]);
    }
    printf("]\n");

    return true;
}

bool MindSporeLiteBackend::infer(const std::vector<float>& input) {
    // Get input tensors from the model (already resized to concrete shape)
    auto inputs = model_->GetInputs();
    if (inputs.empty()) {
        printf("MindSpore Lite: No input tensors available\n");
        return false;
    }
    if (inputs[0].ElementNum() != static_cast<int64_t>(input.size())) {
        printf("MindSpore Lite: Input size mismatch (model=%zu, data=%zu)\n",
               inputs[0].ElementNum(), input.size());
        return false;
    }

    // Copy data into tensor-owned memory
    memcpy(inputs[0].MutableData(), input.data(), input.size() * sizeof(float));

    auto status = model_->Predict(inputs, &outputs_);
    if (status != mindspore::kSuccess) {
        printf("MindSpore Lite: Predict failed code=%d\n", static_cast<int>(status.StatusCode()));
    }
    return status == mindspore::kSuccess;
}

bool MindSporeLiteBackend::infer_with_output(const std::vector<float>& input, std::vector<float>& output) {
    auto inputs = model_->GetInputs();
    if (inputs.empty()) {
        printf("MindSpore Lite: No input tensors available\n");
        return false;
    }
    if (inputs[0].ElementNum() != static_cast<int64_t>(input.size())) {
        printf("MindSpore Lite: Input size mismatch (model=%zu, data=%zu)\n",
               inputs[0].ElementNum(), input.size());
        return false;
    }

    memcpy(inputs[0].MutableData(), input.data(), input.size() * sizeof(float));

    auto status = model_->Predict(inputs, &outputs_);

    printf("MindSpore Lite: Predict code=%d, outputs=%zu\n",
           static_cast<int>(status.StatusCode()), outputs_.size());
    if (status != mindspore::kSuccess) return false;

    if (!outputs_.empty()) {
        auto* data = static_cast<float*>(outputs_[0].MutableData());
        printf("MindSpore Lite: output[0] name=%s elements=%zu\n",
               outputs_[0].Name().c_str(), outputs_[0].ElementNum());
        output.assign(data, data + outputs_[0].ElementNum());
    }
    return true;
}

void MindSporeLiteBackend::deinit() {
    model_.reset();
    context_.reset();
    model_buf_.reset();
    outputs_.clear();
    input_shape_.clear();
}

#else
// Stub implementations when SDK not available
bool MindSporeLiteBackend::init(const BenchmarkConfig&) { return false; }
bool MindSporeLiteBackend::infer(const std::vector<float>&) { return false; }
bool MindSporeLiteBackend::infer_with_output(const std::vector<float>&, std::vector<float>&) { return false; }
void MindSporeLiteBackend::deinit() {}
#endif
