#ifndef BENCHMARK_MODELS_MODEL_INFO_H_
#define BENCHMARK_MODELS_MODEL_INFO_H_

#include <vector>
#include <string>

struct ModelInfo {
    std::string name;
    std::vector<int> input_shape;
    std::string base_path;
    std::string input_name;  // ONNX 模型的第一个数据输入名

    std::string get_model_path(const std::string& backend) const {
        if (backend == "mnn" || backend == "MNN" || backend == "mnn_gpu" || backend == "MNN_GPU") {
            return "models/exported/mnn/" + name + ".mnn";
        }
        if (backend == "ncnn" || backend == "NCNN") {
            return base_path + "/" + name + "_ncnn.param";
        }
        if (backend == "onnxrt" || backend == "ort") {
            return base_path + "/" + name + ".onnx";
        }
        if (backend == "tnn" || backend == "TNN") {
            return base_path + "/" + name + "_TNN.tnnproto";
        }
        if (backend == "tvm" || backend == "TVM") {
            // TVM 编译产物统一放在设备 tvm_models/ 目录
            return "tvm_models/" + name + "_tvm.so";
        }
        if (backend == "tflite" || backend == "TFLite") {
            return base_path + "/" + name + ".tflite";
        }
        if (backend == "mindspore_lite" || backend == "mslite") {
            return base_path + "/" + name + ".ms";
        }
        if (backend == "llamacpp" || backend == "llama") {
            return base_path + "/" + name + ".gguf";
        }
        return base_path + "/" + name + "_" + backend + ".model";
    }

    std::string get_weights_path(const std::string& backend) const {
        if (backend == "mnn" || backend == "MNN" || backend == "onnxrt" || backend == "ort") {
            return ""; // MNN and ONNX Runtime store everything in one file
        }
        if (backend == "ncnn" || backend == "NCNN") {
            return base_path + "/" + name + "_ncnn.bin";
        }
        if (backend == "tnn" || backend == "TNN") {
            return base_path + "/" + name + "_TNN.tnnmodel";
        }
        return base_path + "/" + name + "_" + backend + ".bin";
    }
};

// Declaration of model info getters
ModelInfo get_mobilenetv2_info();
ModelInfo get_resnet50_info();
ModelInfo get_shufflenet_v2_info();
ModelInfo get_yolov8n_info();
ModelInfo get_bert_info();
ModelInfo get_mobilevit_s_info();

#endif // BENCHMARK_MODELS_MODEL_INFO_H_
