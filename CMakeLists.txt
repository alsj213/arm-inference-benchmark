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
option(BENCHMARK_NCNN "Enable ncnn benchmark" ON)
option(BENCHMARK_MNN "Enable MNN benchmark" ON)
option(BENCHMARK_TNN "Enable TNN benchmark" ON)
option(BENCHMARK_TFLITE "Enable TensorFlow Lite benchmark" ON)
option(BENCHMARK_QNN "Enable Qualcomm QNN benchmark" OFF)
option(BENCHMARK_ORT "Enable ONNX Runtime benchmark" ON)
option(BENCHMARK_TVM "Enable Apache TVM benchmark" ON)

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
