#include "mnn_backend.h"

#include <cstring>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

bool MNNBackend::init(const BenchmarkConfig& config) {
    MNN::ScheduleConfig schedule_config;
    schedule_config.numThread = config.num_threads;

    net_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(config.model_path.c_str()));
    if (!net_) {
        printf("Failed to load MNN model: %s\n", config.model_path.c_str());
        return false;
    }

    session_ = net_->createSession(schedule_config);
    if (!session_) {
        printf("Failed to create MNN session\n");
        return false;
    }

    input_tensor_ = net_->getSessionInput(session_, nullptr);
    if (!input_tensor_) {
        printf("Failed to get input tensor\n");
        return false;
    }

    // Resize input
    std::vector<int> shapes = config.input_shape;
    if (shapes.size() == 4) {
        net_->resizeTensor(input_tensor_, shapes);
        net_->resizeSession(session_);
    }

    return true;
}

bool MNNBackend::infer(const std::vector<float>& input) {
    // Copy input data
    memcpy(input_tensor_->host<float>(), input.data(), input.size() * sizeof(float));

    net_->runSession(session_);
    MNN::Tensor* output = net_->getSessionOutput(session_, nullptr);

    return output != nullptr;
}

void MNNBackend::deinit() {
    if (session_) {
        net_->releaseSession(session_);
    }
    if (net_) {
        MNN::Interpreter::destroy(net_.release());
    }
    session_ = nullptr;
    input_tensor_ = nullptr;
}
