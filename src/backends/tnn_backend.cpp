#include "tnn_backend.h"

#include "tnn/core/tnn.h"
#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/core/instance.h"
#include "tnn/core/blob.h"
#include "tnn/core/mat.h"
#include "tnn/utils/blob_converter.h"

#include <fstream>
#include <sstream>
#include <string>

std::string read_file_content(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool TNNBackend::init(const BenchmarkConfig& config) {
    std::string proto_content = read_file_content(config.model_path);
    std::string model_content = read_file_content(config.weights_path);

    if (proto_content.empty() || model_content.empty()) {
        printf("Failed to read TNN model files: %s or %s\n", config.model_path.c_str(), config.weights_path.c_str());
        return false;
    }

    TNN_NS::ModelConfig model_config;
    model_config.params = {proto_content, model_content};

    tnn_ = std::make_shared<TNN_NS::TNN>();
    TNN_NS::Status status;
    status = tnn_->Init(model_config);
    if (status != TNN_NS::TNN_OK) {
        printf("Failed to init TNN: %s\n", status.description().c_str());
        return false;
    }

    // Get input shapes from model
    TNN_NS::InputShapesMap model_shapes;
    status = tnn_->GetModelInputShapesMap(model_shapes);
    if (status != TNN_NS::TNN_OK || model_shapes.empty()) {
        printf("Failed to get TNN input shapes: %s\n", status.description().c_str());
        return false;
    }

    // Use first input name from model
    std::string input_name = model_shapes.begin()->first;
    input_shapes_[input_name] = config.input_shape;

    TNN_NS::NetworkConfig net_config;
#ifdef __ANDROID__
    net_config.device_type = TNN_NS::DEVICE_ARM;
#else
    net_config.device_type = TNN_NS::DEVICE_NAIVE;
#endif
    net_config.precision = TNN_NS::PRECISION_NORMAL;

    instance_ = tnn_->CreateInst(net_config, status, input_shapes_);
    if (status != TNN_NS::TNN_OK || !instance_) {
        printf("Failed to create TNN instance: %s\n", status.description().c_str());
        return false;
    }

    // Store the actual input name for inference
    input_name_ = input_name;

    return true;
}

bool TNNBackend::infer(const std::vector<float>& input) {
    TNN_NS::BlobMap input_blobs;
    TNN_NS::Status status = instance_->GetAllInputBlobs(input_blobs);
    if (status != TNN_NS::TNN_OK || input_blobs.empty()) {
        printf("Failed to get input blobs\n");
        return false;
    }

    // Get first input blob - BlobMap is map<string, Blob*>
    TNN_NS::Blob* input_blob = input_blobs.begin()->second;

    // Create Mat on CPU with our input data - NCHW_FLOAT is float32 NCHW
    TNN_NS::DimsVector shape = input_shapes_[input_name_];

#ifdef __ANDROID__
    TNN_NS::Mat input_mat(TNN_NS::DEVICE_ARM, TNN_NS::NCHW_FLOAT, shape,
                          const_cast<float*>(input.data()));
#else
    TNN_NS::Mat input_mat(TNN_NS::DEVICE_X86, TNN_NS::NCHW_FLOAT, shape,
                          const_cast<float*>(input.data()));
#endif

    // Copy data from Mat to Blob
    TNN_NS::BlobConverter converter(input_blob);
    TNN_NS::MatConvertParam param;
    status = converter.ConvertFromMat(input_mat, param, nullptr);
    if (status != TNN_NS::TNN_OK) {
        printf("TNN ConvertFromMat failed: %s\n", status.description().c_str());
        return false;
    }

    status = instance_->Forward();
    if (status != TNN_NS::TNN_OK) {
        printf("TNN inference failed: %s\n", status.description().c_str());
        return false;
    }

    return true;
}

void TNNBackend::deinit() {
    instance_.reset();
    tnn_.reset();
}
