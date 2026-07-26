#include "mnn_backend.h"

#include <cstring>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>

bool MNNBackend::init(const BenchmarkConfig& config) {
    MNN::ScheduleConfig schedule_config;
    schedule_config.numThread = config.num_threads;

    // BackendConfig for precision control
    MNN::BackendConfig backend_config;

    // GPU mode: use OpenCL backend with CPU fallback
    if (config.backend_type == BackendType::MNN_GPU) {
        schedule_config.type = MNN_FORWARD_OPENCL;
        schedule_config.backupType = MNN_FORWARD_CPU;
        use_gpu_ = true;
        // Use Precision_High to force FP32 compute (default Precision_Normal uses FP16 on Adreno)
        backend_config.precision = MNN::BackendConfig::Precision_High;
        backend_config.memory = MNN::BackendConfig::Memory_High;
        schedule_config.backendConfig = &backend_config;
        printf("MNN OpenCL GPU backend enabled (CPU fallback, Precision=High/FP32)\n");
    } else {
        schedule_config.type = MNN_FORWARD_CPU;
        // Match benchmark.out: Precision_Low (speed priority) + Power_High
        backend_config.precision = MNN::BackendConfig::Precision_Low;
        backend_config.power = MNN::BackendConfig::Power_High;
        schedule_config.backendConfig = &backend_config;
    }

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

    // Cache input/output tensors once (benchmark.out pattern — avoids per-iteration alloc)
    input_tensor_ = net_->getSessionInput(session_, nullptr);
    output_tensor_ = net_->getSessionOutput(session_, nullptr);
    if (!input_tensor_ || !output_tensor_) {
        printf("Failed to get input/output tensor\n");
        return false;
    }

    // Resize input to match config shape (works for both CPU and GPU)
    std::vector<int> shapes = config.input_shape;
    if (shapes.size() >= 2) {
        net_->resizeTensor(input_tensor_, shapes);
        net_->resizeSession(session_);
    }

    // For GPU mode, create a host staging tensor for input data transfer
    // Use CAFFE (NCHW) dimension type to match ONNX model format
    if (use_gpu_) {
        host_input_tensor_.reset(MNN::Tensor::create<float>(shapes, nullptr, MNN::Tensor::CAFFE));
        if (!host_input_tensor_) {
            printf("Failed to create host input tensor for GPU mode\n");
            return false;
        }
    }

    return true;
}

bool MNNBackend::prepare(const std::vector<float>& input) {
    // ── Untimed: copy input data to tensor (matches benchmark.out's map/unmap + pre-load) ──
    if (use_gpu_) {
        memcpy(host_input_tensor_->host<float>(), input.data(), input.size() * sizeof(float));
        input_tensor_->copyFromHostTensor(host_input_tensor_.get());
    } else {
        memcpy(input_tensor_->host<float>(), input.data(), input.size() * sizeof(float));
    }
    return true;
}

bool MNNBackend::run() {
    // ── Timed: matches benchmark.out timing scope ──
    net_->runSession(session_);
    // map/unmap output to trigger sync (near-zero on CPU, needed for GPU)
    if (use_gpu_) {
        auto host_out = MNN::Tensor::createHostTensorFromDevice(output_tensor_, true);
    } else {
        void* out_host = output_tensor_->map(MNN::Tensor::MAP_TENSOR_READ,
                                             output_tensor_->getDimensionType());
        output_tensor_->unmap(MNN::Tensor::MAP_TENSOR_READ,
                              output_tensor_->getDimensionType(), out_host);
    }
    return true;
}

bool MNNBackend::infer(const std::vector<float>& input) {
    // Legacy single-shot: prepare + run (for backward compat)
    prepare(input);
    return run();
}

bool MNNBackend::infer_with_output(const std::vector<float>& input, std::vector<float>& output) {
    // Copy input data — GPU uses copyFromHostTensor, CPU uses direct memcpy
    if (use_gpu_) {
        memcpy(host_input_tensor_->host<float>(), input.data(), input.size() * sizeof(float));
        input_tensor_->copyFromHostTensor(host_input_tensor_.get());
    } else {
        memcpy(input_tensor_->host<float>(), input.data(), input.size() * sizeof(float));
    }

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

    output.resize(output_size);

    // Copy output data — GPU uses createHostTensorFromDevice, CPU uses direct memcpy
    if (use_gpu_) {
        std::unique_ptr<MNN::Tensor> host_output(
            MNN::Tensor::createHostTensorFromDevice(output_tensor, true));
        if (host_output) {
            memcpy(output.data(), host_output->host<float>(), output_size * sizeof(float));
        } else {
            printf("[MNN GPU] createHostTensorFromDevice failed, trying copyToHostTensor\n");
            std::unique_ptr<MNN::Tensor> host_out(
                MNN::Tensor::create<float>(output_shape, nullptr, MNN::Tensor::CAFFE));
            if (host_out && output_tensor->copyToHostTensor(host_out.get())) {
                memcpy(output.data(), host_out->host<float>(), output_size * sizeof(float));
            } else {
                printf("[MNN GPU] copyToHostTensor also failed!\n");
                return false;
            }
        }
    } else {
        memcpy(output.data(), output_tensor->host<float>(), output_size * sizeof(float));
    }

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
