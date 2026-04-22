#ifndef BENCHMARK_BACKENDS_TFLITE_BACKEND_H_
#define BENCHMARK_BACKENDS_TFLITE_BACKEND_H_

#include "../common/benchmark.h"
#include <vector>

#ifdef BENCHMARK_TFLITE
// Forward declarations for TFLite C API
typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteTensor TfLiteTensor;
#endif

/*!
 * \brief TensorFlow Lite Backend - Official TFLite Integration
 *
 * Official TensorFlow Lite integration:
 *   - Model format: .tflite (FlatBuffer)
 *   - Runtime: Official TFLite library
 *   - XNNPACK delegate enabled for ARM optimization
 */
class TFLiteBackend : public BenchmarkBackend {
 public:
  bool init(const BenchmarkConfig& config) override;
  bool infer(const std::vector<float>& input) override;
  void deinit() override;
  std::string name() const override { return "TFLite"; }

 private:
#ifdef BENCHMARK_TFLITE
  TfLiteModel* model_ = nullptr;
  TfLiteInterpreter* interpreter_ = nullptr;
  TfLiteInterpreterOptions* options_ = nullptr;
  int input_tensor_index_ = 0;
#endif
  std::vector<float> input_buffer_;
  int input_size_ = 0;
  bool use_real_inference_ = false;
};

#endif  // BENCHMARK_BACKENDS_TFLITE_BACKEND_H_
