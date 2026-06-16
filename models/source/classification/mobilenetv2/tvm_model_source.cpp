/*
 * TVM 格式的模型 .so 文件
 * 模拟 TVM 编译后的 ABI 接口
 * 真实的 TVM 编译输出包含:
 * - 各个算子的 PackedFunc
 * - __tvm_main__ 入口
 * - 参数数据
 */

#include <cstdint>
#include <cstring>
#include <algorithm>

// 简单的卷积实现
static void conv2d_3x3_s2(const float* input, float* output,
                           int in_c, int in_h, int in_w,
                           int out_c) {
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
                sum += input[ic * in_h * in_w + ih * in_w + iw] * 0.1f;
              }
            }
          }
        }
        output[oc * out_h * out_w + oh * out_w + ow] = std::max(0.0f, sum);
      }
    }
  }
}

// 深度可分离卷积
static void depthwise_conv(const float* input, float* output,
                           int channels, int height, int width) {
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

// 1x1 卷积
static void conv2d_1x1(const float* input, float* output,
                        int in_c, int out_c, int height, int width) {
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

// 全局平均池化
static void global_avg_pool(const float* input, float* output,
                             int channels, int height, int width) {
  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        sum += input[c * height * width + h * width + w];
      }
    }
    output[c] = sum / (height * width);
  }
}

// 全连接层
static void dense(const float* input, float* output, int in_c, int out_c) {
  for (int oc = 0; oc < out_c; ++oc) {
    float sum = 0.0f;
    for (int ic = 0; ic < in_c; ++ic) {
      sum += input[ic] * 0.01f;
    }
    output[oc] = sum;
  }
}

// 中间缓冲区 (约 5MB)
static float workspace[1280 * 14 * 14];

// TVM 风格的主推理入口
// TVM_ 前缀避免符号冲突
extern "C" int TVM_mobilenetv2_infer(const float* input, float* output) {
  float* buf = workspace;
  int ptr = 0;

  // Stage 1: Conv 3x3, 3 -> 32, stride 2
  float* conv1_out = buf + ptr;
  ptr += 32 * 112 * 112;
  conv2d_3x3_s2(input, conv1_out, 3, 224, 224, 32);

  // Stage 2: Depthwise + Pointwise
  float* dw_out = buf + ptr;
  ptr += 32 * 112 * 112;
  depthwise_conv(conv1_out, dw_out, 32, 112, 112);

  float* pw_out = buf + ptr;
  ptr += 16 * 112 * 112;
  conv2d_1x1(dw_out, pw_out, 32, 16, 112, 112);

  // Stage 3: Bottlenecks (简化版)
  float* bn3_dw = buf + ptr;
  ptr += 16 * 56 * 56;
  depthwise_conv(pw_out, bn3_dw, 16, 56, 56);

  float* bn3_pw = buf + ptr;
  ptr += 24 * 56 * 56;
  conv2d_1x1(bn3_dw, bn3_pw, 16, 24, 56, 56);

  float* bn4_dw = buf + ptr;
  ptr += 24 * 28 * 28;
  depthwise_conv(bn3_pw, bn4_dw, 24, 28, 28);

  float* bn4_pw = buf + ptr;
  ptr += 32 * 28 * 28;
  conv2d_1x1(bn4_dw, bn4_pw, 24, 32, 28, 28);

  float* bn5_dw = buf + ptr;
  ptr += 32 * 14 * 14;
  depthwise_conv(bn4_pw, bn5_dw, 32, 14, 14);

  float* bn5_pw = buf + ptr;
  ptr += 64 * 14 * 14;
  conv2d_1x1(bn5_dw, bn5_pw, 32, 64, 14, 14);

  float* bn6_dw = buf + ptr;
  ptr += 64 * 14 * 14;
  depthwise_conv(bn5_pw, bn6_dw, 64, 14, 14);

  float* bn6_pw = buf + ptr;
  ptr += 96 * 14 * 14;
  conv2d_1x1(bn6_dw, bn6_pw, 64, 96, 14, 14);

  float* bn7_dw = buf + ptr;
  ptr += 96 * 7 * 7;
  depthwise_conv(bn6_pw, bn7_dw, 96, 7, 7);

  float* bn7_pw = buf + ptr;
  ptr += 160 * 7 * 7;
  conv2d_1x1(bn7_dw, bn7_pw, 96, 160, 7, 7);

  float* bn8_dw = buf + ptr;
  ptr += 160 * 7 * 7;
  depthwise_conv(bn7_pw, bn8_dw, 160, 7, 7);

  float* bn8_pw = buf + ptr;
  ptr += 320 * 7 * 7;
  conv2d_1x1(bn8_dw, bn8_pw, 160, 320, 7, 7);

  // Final Conv 1x1
  float* conv_final = buf + ptr;
  ptr += 1280 * 7 * 7;
  conv2d_1x1(bn8_pw, conv_final, 320, 1280, 7, 7);

  // Global Average Pool
  float* pool_out = buf + ptr;
  ptr += 1280;
  global_avg_pool(conv_final, pool_out, 1280, 7, 7);

  // Classifier
  dense(pool_out, output, 1280, 1000);

  return 0;
}

// TVM 标准符号
extern "C" void* TVMFuncGetGlobal(const char* name) {
  if (strcmp(name, "mobilenetv2_infer") == 0) {
    return reinterpret_cast<void*>(TVM_mobilenetv2_infer);
  }
  return nullptr;
}

extern "C" const char* TVMGetVersion() {
  return "0.24.0 - Custom Compiled Model";
}
