#include "model_info.h"

ModelInfo get_yolov8n_info() {
    ModelInfo info;
    info.name = "yolov8n";
    info.input_shape = {1, 3, 640, 640};  // NCHW
    info.base_path = "./models/detection/yolov8n";
    return info;
}
