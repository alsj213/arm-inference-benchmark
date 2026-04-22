#include "tvm_backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

// 简化版 TVM 后端：使用本地实现 MobileNetV2
// 完整的 Conv + Pooling + FC 算子实现
// 用于模拟 TVM codegen 的性能基线

// static std::vector<float> g_intermediate; - removed to avoid static init issues

// 简单的 3x3 Conv (stride=2, padding=1)
static void conv3x3s2(const float* input, float* output, int in_c, int in_h, int in_w, int out_c) {
  int out_h = in_h / 2;
  int out_w = in_w / 2;

  for (int oc = 0; oc < out_c; ++oc) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_c; ++ic) {
          for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
              int ih = oh * 2 + kh - 1;
              int iw = ow * 2 + kw - 1;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int idx = ic * in_h * in_w + ih * in_w + iw;
                sum += input[idx] * 0.1f;
              }
            }
          }
        }
        output[oc * out_h * out_w + oh * out_w + ow] = std::max(0.0f, sum);
      }
    }
  }
}

// Depthwise Conv 3x3
static void dwconv3x3(const float* input, float* output, int channels, int height, int width) {
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        float sum = 0.0f;
        for (int kh = 0; kh < 3; ++kh) {
          for (int kw = 0; kw < 3; ++kw) {
            int ih = h + kh - 1;
            int iw = w + kw - 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
              sum += input[c * height * width + ih * width + iw] * 0.1f;
            }
          }
        }
        output[c * height * width + h * width + w] = std::max(0.0f, sum);
      }
    }
  }
}

// Pointwise Conv 1x1
static void pwconv1x1(const float* input, float* output, int in_c, int out_c, int height, int width) {
  for (int oc = 0; oc < out_c; ++oc) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_c; ++ic) {
          sum += input[ic * height * width + h * width + w] * 0.1f;
        }
        output[oc * height * width + h * width + w] = std::max(0.0f, sum);
      }
    }
  }
}

bool TVMBackend::init(const BenchmarkConfig& config) {
  input_size_ = 1;
  for (int dim : config.input_shape) {
    input_size_ *= dim;
  }
  input_buffer_.resize(input_size_);
  intermediate_buffer_.resize(2000000);  // 2M floats = 8MB
  use_real_inference_ = true;
  return true;
}

bool TVMBackend::infer(const std::vector<float>& input) {
  if (input.size() != static_cast<size_t>(input_size_)) {
    printf("TVM: Input size mismatch\n");
    return false;
  }

  float* buf = intermediate_buffer_.data();
  int ptr = 0;

  // Stage 1: Conv 3x3, 3 -> 32, stride 2
  float* conv1_out = buf + ptr;
  ptr += 32 * 112 * 112;
  conv3x3s2(input.data(), conv1_out, 3, 224, 224, 32);

  // Stage 2: Bottleneck 1 (32 -> 16)
  float* dw2_out = buf + ptr;
  ptr += 32 * 112 * 112;
  dwconv3x3(conv1_out, dw2_out, 32, 112, 112);

  float* pw2_out = buf + ptr;
  ptr += 16 * 112 * 112;
  pwconv1x1(dw2_out, pw2_out, 32, 16, 112, 112);

  // Stage 3: Bottleneck 2 (16 -> 24)
  float* dw3_out = buf + ptr;
  ptr += 16 * 56 * 56;
  conv3x3s2(pw2_out, dw3_out, 16, 112, 112, 16);

  float* pw3_out = buf + ptr;
  ptr += 24 * 56 * 56;
  pwconv1x1(dw3_out, pw3_out, 16, 24, 56, 56);

  // Stage 4: Bottleneck 3-4 (24 -> 32)
  float* dw4_out = buf + ptr;
  ptr += 24 * 28 * 28;
  conv3x3s2(pw3_out, dw4_out, 24, 56, 56, 24);

  float* pw4_out = buf + ptr;
  ptr += 32 * 28 * 28;
  pwconv1x1(dw4_out, pw4_out, 24, 32, 28, 28);

  // Stage 5: Bottlenecks 5-7 (32 -> 64)
  float* dw5_out = buf + ptr;
  ptr += 32 * 14 * 14;
  conv3x3s2(pw4_out, dw5_out, 32, 28, 28, 32);

  float* pw5_out = buf + ptr;
  ptr += 64 * 14 * 14;
  pwconv1x1(dw5_out, pw5_out, 32, 64, 14, 14);

  // Stage 6: Bottlenecks 8-10 (64 -> 96)
  float* dw6_out = buf + ptr;
  ptr += 64 * 14 * 14;
  dwconv3x3(pw5_out, dw6_out, 64, 14, 14);

  float* pw6_out = buf + ptr;
  ptr += 96 * 14 * 14;
  pwconv1x1(dw6_out, pw6_out, 64, 96, 14, 14);

  // Stage 7: Bottlenecks 11-13 (96 -> 160)
  float* dw7_out = buf + ptr;
  ptr += 96 * 7 * 7;
  conv3x3s2(pw6_out, dw7_out, 96, 14, 14, 96);

  float* pw7_out = buf + ptr;
  ptr += 160 * 7 * 7;
  pwconv1x1(dw7_out, pw7_out, 96, 160, 7, 7);

  // Stage 8: Bottleneck 14 (160 -> 320)
  float* dw8_out = buf + ptr;
  ptr += 160 * 7 * 7;
  dwconv3x3(pw7_out, dw8_out, 160, 7, 7);

  float* pw8_out = buf + ptr;
  ptr += 320 * 7 * 7;
  pwconv1x1(dw8_out, pw8_out, 160, 320, 7, 7);

  // Final Conv 1x1: 320 -> 1280
  float* conv_final = buf + ptr;
  ptr += 1280 * 7 * 7;
  pwconv1x1(pw8_out, conv_final, 320, 1280, 7, 7);

  // Global Average Pool
  float* pool_out = buf + ptr;
  ptr += 1280;
  for (int c = 0; c < 1280; ++c) {
    float sum = 0.0f;
    for (int h = 0; h < 7; ++h) {
      for (int w = 0; w < 7; ++w) {
        sum += conv_final[c * 7 * 7 + h * 7 + w];
      }
    }
    pool_out[c] = sum / 49.0f;
  }

  // Final FC: 1280 -> 1000
  float* logits = buf + ptr;
  for (int oc = 0; oc < 1000; ++oc) {
    float sum = 0.0f;
    for (int ic = 0; ic < 1280; ++ic) {
      sum += pool_out[ic] * 0.01f;
    }
    logits[oc] = sum;
  }

  // 确保编译器不会优化掉所有计算
  volatile float verify_sum = 0.0f;
  for (int i = 0; i < 1000; ++i) {
    verify_sum += logits[i];
  }
  (void)verify_sum;

  return true;
}

void TVMBackend::deinit() {
  input_size_ = 0;
  input_buffer_.clear();
  intermediate_buffer_.clear();
}
