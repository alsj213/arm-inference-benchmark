#ifndef BENCHMARK_BACKENDS_TVM_BACKEND_H_
#define BENCHMARK_BACKENDS_TVM_BACKEND_H_

#include "../common/benchmark.h"
#include <vector>

/*!
 * \brief TVM Backend - Apache TVM Integration
 *
 * \note 当前实现：手工实现 MobileNetV2 的完整推理流程
 *       - 包含所有关键层: Conv2D, DepthwiseConv, PointwiseConv, ReLU, AvgPool
 *       - 完整的 Bottleneck 结构 x17
 *       - 目的: 提供真实的、可复现的性能基准
 *
 *       真实的 TVM 编译流程需要:
 *       1. 完整编译 TVM 主机版 (含 LLVM, Relay 编译器)
 *       2. Python 前端: ONNX -> TVM Relay -> 交叉编译 ARM64
 *       3. 生成 .so 动态库文件
 *
 *       当前实现性能基线: 纯手工实现，无 NEON 优化，无 AutoTVM 调度
 */
class TVMBackend : public BenchmarkBackend {
 public:
  bool init(const BenchmarkConfig& config) override;
  bool infer(const std::vector<float>& input) override;
  void deinit() override;
  std::string name() const override { return "TVM"; }

 private:
  std::vector<float> input_buffer_;
  std::vector<float> intermediate_buffer_;
  int input_size_ = 0;
  bool use_real_inference_ = false;
};

#endif  // BENCHMARK_BACKENDS_TVM_BACKEND_H_
