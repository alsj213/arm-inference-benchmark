#include "mnn_backend.h"

#include <cstring>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

bool MNNBackend::init(const BenchmarkConfig& config) {
    MNN::ScheduleConfig schedule_config;
    schedule_config.numThread = config.num_threads;

    // Enable profiling if configured
    if (config.enable_profiling && !config.profile_file.empty()) {
        profiling_enabled_ = true;
        profile_file_ = config.profile_file;
        // MNN profiling is enabled via environment variable MNN_PROFILING
        printf("MNN: Profiling enabled, output: %s\n", config.profile_file.c_str());
    }

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

    // Run inference with profiling if enabled
    if (profiling_enabled_) {
        // MNN profiling is controlled via environment variable
        // Set MNN_PROFILING=1 and MNN_PROFILING_FILE=<file> before running
        net_->runSession(session_);
    } else {
        net_->runSession(session_);
    }

    MNN::Tensor* output = net_->getSessionOutput(session_, nullptr);

    return output != nullptr;
}

bool MNNBackend::infer_with_output(const std::vector<float>& input, std::vector<float>& output) {
    // Copy input data
    memcpy(input_tensor_->host<float>(), input.data(), input.size() * sizeof(float));

    net_->runSession(session_);
    MNN::Tensor* output_tensor = net_->getSessionOutput(session_, nullptr);

    if (!output_tensor) {
        return false;
    }

    // Get output size
    auto output_shape = output_tensor->shape();
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    // Copy output data
    output.resize(output_size);
    memcpy(output.data(), output_tensor->host<float>(), output_size * sizeof(float));

    return true;
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
