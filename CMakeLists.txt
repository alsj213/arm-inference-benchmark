cmake_minimum_required(VERSION 3.14)
project(benchmark_arm_inference C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable exceptions for ONNX Runtime and TVM
if(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
    set(ANDROID_CPP_FEATURES "exceptions")
    set(ANDROID_ENABLE_CPP_EXCEPTIONS ON)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
endif()

# Options to enable/disable backends
# =============================================================================
# 当前活动后端：MNN (BENCHMARK_MNN) 和 ONNX Runtime (BENCHMARK_ORT)
#
# 要重新启用已停用的后端，将下方对应选项设为 ON，然后重新编译即可。
# 所有后端源代码完整保留在 src/backends/ 和 third_party/ 中。
#
# 可恢复的后端：
#   BENCHMARK_NCNN    - ncnn（腾讯）
#   BENCHMARK_TFLITE  - TensorFlow Lite
#   BENCHMARK_TNN     - TNN（字节跳动）
#   BENCHMARK_QNN     - Qualcomm QNN SDK
#   BENCHMARK_TVM     - Apache TVM
#   BENCHMARK_LLAMACPP - llama.cpp（LLM 推理）
#
# 完整的多后端配置见 backup/all-backends 分支
# =============================================================================
option(BENCHMARK_NCNN "Enable ncnn benchmark" ON)
option(BENCHMARK_MNN "Enable MNN benchmark" ON)
option(BENCHMARK_TFLITE "Enable TensorFlow Lite benchmark" ON)
option(BENCHMARK_TNN "Enable TNN benchmark" ON)
option(BENCHMARK_ORT "Enable ONNX Runtime benchmark" ON)
option(BENCHMARK_QNN "Enable QNN benchmark" OFF)          # 已停用，可恢复
option(BENCHMARK_TVM "Enable Apache TVM benchmark" OFF)   # 已停用，可恢复
option(BENCHMARK_LLAMACPP "Enable llama.cpp LLM benchmark" OFF)  # 已停用，可恢复

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add ARM specific optimizations if cross-compiling for ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_compile_options(-march=armv8.2-a+fp16)
    message(STATUS "Building for ARM64 with FP16 support")
endif()

# Disable ncnn's exception disabling since ONNX Runtime needs exceptions
set(NCNN_DISABLE_EXCEPTION OFF CACHE BOOL "" FORCE)

# Add third_party directory
add_subdirectory(third_party)

# Add source directory
add_subdirectory(src)
