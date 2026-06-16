#include "model_info.h"

ModelInfo get_resnet50_info() {
    ModelInfo info;
    info.name = "resnet50";
    info.input_shape = {1, 3, 224, 224};  // NCHW
    info.base_path = "./models/source/classification/resnet50";
    return info;
}
