#include "tflite_backend.h"

#ifdef BENCHMARK_TFLITE
#include "tensorflow/lite/c/c_api.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

bool TFLiteBackend::init(const BenchmarkConfig& config) {
  printf("=== TensorFlow Lite Backend ===\n");

  // TFLite uses NHWC format: convert NCHW -> NHWC
  // NCHW: [1, C, H, W] -> NHWC: [1, H, W, C]
  std::vector<int> tflite_input_shape;
  if (config.input_shape.size() == 4) {
    // NCHW to NHWC
    int n = config.input_shape[0];
    int c = config.input_shape[1];
    int h = config.input_shape[2];
    int w = config.input_shape[3];
    tflite_input_shape = {n, h, w, c};
  } else {
    tflite_input_shape = config.input_shape;
  }

  // Calculate input size (use original for external interface)
  input_size_ = 1;
  for (int dim : config.input_shape) {
    input_size_ *= dim;
  }
  input_buffer_.resize(input_size_);

  printf("TFLite: Input shape (NCHW): [");
  for (size_t i = 0; i < config.input_shape.size(); i++) {
    if (i > 0) printf(", ");
    printf("%d", config.input_shape[i]);
  }
  printf("]\n");
  printf("TFLite: Input shape (NHWC): [");
  for (size_t i = 0; i < tflite_input_shape.size(); i++) {
    if (i > 0) printf(", ");
    printf("%d", tflite_input_shape[i]);
  }
  printf("]\n");
  printf("TFLite: Model path: %s\n", config.model_path.c_str());

#ifdef BENCHMARK_TFLITE
  // Check if model file exists
  std::ifstream model_file(config.model_path, std::ios::binary | std::ios::ate);
  if (!model_file.is_open()) {
    printf("TFLite: Model not found, using validation mode\n");
    printf("TFLite: Expected: models/source/classification/mobilenetv2/mobilenetv2.tflite\n");
    use_real_inference_ = false;
    return true;
  }

  // Read model file
  std::streamsize model_size = model_file.tellg();
  model_file.seekg(0, std::ios::beg);
  std::vector<char> model_data(model_size);
  if (!model_file.read(model_data.data(), model_size)) {
    printf("TFLite: Failed to read model file\n");
    return false;
  }
  model_file.close();
  printf("TFLite: Model loaded, size: %zd bytes\n", model_size);

  // Create TFLite model
  model_ = TfLiteModelCreate(model_data.data(), model_size);
  if (!model_) {
    printf("TFLite: Failed to create model\n");
    return false;
  }

  // Create interpreter options
  options_ = TfLiteInterpreterOptionsCreate();
  if (!options_) {
    printf("TFLite: Failed to create options\n");
    return false;
  }

  // Set number of threads
  TfLiteInterpreterOptionsSetNumThreads(options_, config.num_threads);
  printf("TFLite: Using %d thread(s)\n", config.num_threads);

  // Create interpreter
  interpreter_ = TfLiteInterpreterCreate(model_, options_);
  if (!interpreter_) {
    printf("TFLite: Failed to create interpreter\n");
    return false;
  }

  // Allocate tensors
  TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter_);
  if (status != kTfLiteOk) {
    printf("TFLite: Failed to allocate tensors\n");
    return false;
  }

  // Get input tensor count
  int input_count = TfLiteInterpreterGetInputTensorCount(interpreter_);
  printf("TFLite: Input tensor count: %d\n", input_count);

  // Get first input tensor
  input_tensor_index_ = 0;
  const TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter_, input_tensor_index_);
  if (!input_tensor) {
    printf("TFLite: Failed to get input tensor\n");
    return false;
  }

  // Print input tensor details
  int input_dims = TfLiteTensorNumDims(input_tensor);
  printf("TFLite: Input tensor dimensions: %d\n", input_dims);
  printf("TFLite: Input tensor shape: [");
  for (int i = 0; i < input_dims; i++) {
    if (i > 0) printf(", ");
    printf("%d", TfLiteTensorDim(input_tensor, i));
  }
  printf("]\n");
  printf("TFLite: Input tensor bytes: %d\n", TfLiteTensorByteSize(input_tensor));

  // Get output tensor count
  int output_count = TfLiteInterpreterGetOutputTensorCount(interpreter_);
  printf("TFLite: Output tensor count: %d\n", output_count);

  // Print output tensor details
  for (int i = 0; i < output_count; i++) {
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter_, i);
    if (output_tensor) {
      int output_dims = TfLiteTensorNumDims(output_tensor);
      printf("TFLite: Output %d shape: [", i);
      for (int j = 0; j < output_dims; j++) {
        if (j > 0) printf(", ");
        printf("%d", TfLiteTensorDim(output_tensor, j));
      }
      printf("] bytes: %d\n", TfLiteTensorByteSize(output_tensor));
    }
  }

  use_real_inference_ = true;
  printf("TFLite: Real inference enabled!\n");
  return true;

#else
  // Simulation mode
  printf("TFLite: Running in simulation mode (BENCHMARK_TFLITE not enabled)\n");
  use_real_inference_ = false;
  return true;
#endif
}

bool TFLiteBackend::infer(const std::vector<float>& input) {
  if (input.size() != static_cast<size_t>(input_size_)) {
    printf("TFLite: Input size mismatch\n");
    return false;
  }

#ifdef BENCHMARK_TFLITE
  if (use_real_inference_ && interpreter_) {
    // Copy input data to tensor
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter_, input_tensor_index_);
    if (!input_tensor) {
      printf("TFLite: Failed to get input tensor for inference\n");
      return false;
    }

    // Get input tensor dimensions
    int num_dims = TfLiteTensorNumDims(input_tensor);
    if (num_dims == 4) {
      int n = TfLiteTensorDim(input_tensor, 0);
      int h = TfLiteTensorDim(input_tensor, 1);
      int w = TfLiteTensorDim(input_tensor, 2);
      int c = TfLiteTensorDim(input_tensor, 3);

      // Convert NCHW to NHWC
      float* tflite_input = static_cast<float*>(TfLiteTensorData(input_tensor));
      const float* nchw_input = input.data();

      for (int b = 0; b < n; b++) {
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            for (int ch = 0; ch < c; ch++) {
              // NHWC index: b*h*w*c + y*w*c + x*c + ch
              // NCHW index: b*c*h*w + ch*h*w + y*w + x
              int nhwc_idx = b * h * w * c + y * w * c + x * c + ch;
              int nchw_idx = b * c * h * w + ch * h * w + y * w + x;
              tflite_input[nhwc_idx] = nchw_input[nchw_idx];
            }
          }
        }
      }
    } else {
      // For non-4D tensors, copy directly
      memcpy(TfLiteTensorData(input_tensor), input.data(), input.size() * sizeof(float));
    }

    // Run inference
    TfLiteStatus status = TfLiteInterpreterInvoke(interpreter_);
    if (status != kTfLiteOk) {
      printf("TFLite: Inference failed\n");
      return false;
    }

    return true;
  }
#endif

  // Fallback: simulate computation if real inference isn't available
  volatile float sum = 0.0f;
  for (int iter = 0; iter < 35; iter++) {
    for (size_t i = 0; i < input.size(); i += 4) {
      float v0 = input[i + 0] * 0.5f;
      float v1 = input[i + 1] * 0.5f;
      float v2 = input[i + 2] * 0.5f;
      float v3 = input[i + 3] * 0.5f;
      sum += v0 + v1 + v2 + v3;
    }
  }
  (void)sum;
  return true;
}

void TFLiteBackend::deinit() {
#ifdef BENCHMARK_TFLITE
  if (interpreter_) {
    TfLiteInterpreterDelete(interpreter_);
    interpreter_ = nullptr;
  }
  if (options_) {
    TfLiteInterpreterOptionsDelete(options_);
    options_ = nullptr;
  }
  if (model_) {
    TfLiteModelDelete(model_);
    model_ = nullptr;
  }
#endif
  input_size_ = 0;
  input_buffer_.clear();
}
