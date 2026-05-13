#include "model_info.h"

ModelInfo get_bert_info() {
    ModelInfo info;
    info.name = "bert";
    info.input_shape = {1, 128};  // batch, sequence_length
    info.base_path = "./models/nlp/bert";
    return info;
}
