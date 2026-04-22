#include "model_info.h"

ModelInfo get_mobilenetv2_info() {
    ModelInfo info;
    info.name = "mobilenetv2";
    info.input_shape = {1, 3, 224, 224};  // NCHW
    info.base_path = "./models/classification/mobilenetv2";
    return info;
}
