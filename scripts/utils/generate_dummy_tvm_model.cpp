/*
 * 最小化的 TVM 兼容模型动态库
 * 提供 TVM Runtime 期望的符号接口
 */

#include <cstdint>
#include <cstring>
#include <vector>

// TVM PackedFunc 签名
typedef void (*TVMPackedFunc)(void* args, int* type_codes, int num_args,
                               void* out_ret_value, int* out_ret_tcode,
                               void* resource_handle);

// 简单的矩阵乘法模拟 - 模拟 MobileNetV2 的计算量
extern "C" void mobilenetv2_inference(void* args, int* type_codes, int num_args,
                                       void* out_ret_value, int* out_ret_tcode,
                                       void* resource_handle) {
    // 输入是一个 1x3x224x224 的 float 张量
    // 输出是一个 1x1000 的 float 张量

    // 模拟 MobileNetV2 的计算量
    const int64_t total_ops = 300 * 1000 * 1000; // ~300M 运算
    volatile float sum = 0.0f;

    // 执行足够多的计算来模拟真实推理延迟
    for (int64_t i = 0; i < total_ops / 100; ++i) {
        sum += i * 0.001f;
    }

    // 简单地把输入复制到输出
    if (num_args >= 2 && args != nullptr) {
        float* input = static_cast<float*>(static_cast<void**>(args)[0]);
        float* output = static_cast<float*>(static_cast<void**>(args)[1]);

        // 复制前 1000 个元素作为输出
        if (input && output) {
            memcpy(output, input, 1000 * sizeof(float));
        }
    }

    (void)sum;
}

// TVMModule_GetFunction 接口
extern "C" int TVMModule_GetFunction(const char* name, void** out_func) {
    if (strcmp(name, "mobilenetv2_inference") == 0) {
        *out_func = reinterpret_cast<void*>(mobilenetv2_inference);
        return 0;
    }
    return -1;
}

// TVMBackend 期望的符号
extern "C" int TVMFuncFree(void* func) {
    return 0;
}

extern "C" int TVMArrayAlloc(void* shape, int ndim, int dtype_code,
                              int dtype_bits, int dtype_lanes,
                              void* out_array) {
    return 0;
}

extern "C" int TVMArrayFree(void* arr) {
    return 0;
}

extern "C" int TVMArrayCopyFromBytes(void* arr, void* data, long nbytes) {
    return 0;
}

extern "C" int TVMArrayCopyToBytes(void* arr, void* data, long nbytes) {
    return 0;
}

extern "C" int TVMFuncCall(void* func, void* arg_values, int* arg_tcodes,
                           int num_args, void* ret_values, int* ret_tcodes,
                           int num_ret) {
    return 0;
}
