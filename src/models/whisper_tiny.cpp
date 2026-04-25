#include "model_info.h"

ModelInfo get_whisper_tiny_encoder_info() {
    ModelInfo info;
    info.name = "whisper_tiny_encoder";
    info.input_shape = {1, 80, 3000};  // [batch, n_mels, n_frames] - 30 seconds
    info.base_path = "./models/speech/whisper_tiny";
    return info;
}

ModelInfo get_whisper_tiny_decoder_info() {
    ModelInfo info;
    info.name = "whisper_tiny_decoder";
    info.input_shape = {1, 1};  // Decoder input tokens
    info.base_path = "./models/speech/whisper_tiny";
    return info;
}
