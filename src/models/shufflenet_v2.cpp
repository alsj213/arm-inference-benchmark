#include "model_info.h"

ModelInfo get_shufflenet_v2_info() {
    ModelInfo info;
    info.name = "shufflenet_v2_x0_5";
    info.input_shape = {1, 3, 224, 224};  // NCHW
    info.base_path = "./models/classification/shufflenet_v2";
    return info;
}
