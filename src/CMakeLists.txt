# src CMake configuration

# Enable exceptions for ONNX Runtime and TVM
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -UNO_EXCEPTIONS")

# Common library
add_library(benchmark_common
    common/utils.cpp
    common/benchmark.cpp
    common/config.cpp
)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add third_party include directories
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
if(BENCHMARK_NCNN)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/src)
target_include_directories(benchmark_common PUBLIC ${PROJECT_BINARY_DIR}/third_party/ncnn/src)
endif()
if(BENCHMARK_MNN)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/MNN/include)
endif()
if(BENCHMARK_TFLITE)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tflite_extracted/headers)
endif()
if(BENCHMARK_ORT)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/include)
endif()
if(BENCHMARK_TVM)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tvm/include)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tvm/3rdparty/tvm-ffi/include)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include)
endif()

# Add compile definitions for enabled backends
if(BENCHMARK_NCNN)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_NCNN)
endif()
if(BENCHMARK_MNN)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_MNN)
endif()
if(BENCHMARK_TFLITE)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_TFLITE)
endif()
if(BENCHMARK_ORT)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_ORT)
endif()
if(BENCHMARK_TVM)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_TVM)
endif()

# Backend sources
set(BACKEND_SRCS)
set(BACKEND_LIBS)

if(BENCHMARK_NCNN)
    list(APPEND BACKEND_SRCS backends/ncnn_backend.cpp)
    list(APPEND BACKEND_LIBS ncnn)
endif()

if(BENCHMARK_MNN)
    list(APPEND BACKEND_SRCS backends/mnn_backend.cpp)
    list(APPEND BACKEND_LIBS MNN)
endif()

if(BENCHMARK_TFLITE)
    list(APPEND BACKEND_SRCS backends/tflite_backend.cpp)
    list(APPEND BACKEND_LIBS tensorflow-lite)
endif()

if(BENCHMARK_ORT)
    list(APPEND BACKEND_SRCS backends/ort_backend.cpp)
    list(APPEND BACKEND_LIBS onnxruntime)
endif()

if(BENCHMARK_TVM)
    list(APPEND BACKEND_SRCS backends/tvm_backend.cpp)
endif()

# Model wrappers
set(MODEL_SRCS
    models/mobilenet.cpp
    models/resnet.cpp
    models/yolov8.cpp
    models/bert.cpp
)

# Main executable
add_executable(benchmark_inference
    main.cpp
    ${BACKEND_SRCS}
    ${MODEL_SRCS}
)
target_compile_options(benchmark_inference PRIVATE -fexceptions)
target_compile_options(benchmark_common PUBLIC -fexceptions)

# Copy compile definitions from benchmark_common to benchmark_inference
# This ensures all backend macros are visible in main.cpp
if(BENCHMARK_NCNN)
    target_compile_definitions(benchmark_inference PRIVATE BENCHMARK_NCNN)
endif()
if(BENCHMARK_MNN)
    target_compile_definitions(benchmark_inference PRIVATE BENCHMARK_MNN)
endif()
if(BENCHMARK_TFLITE)
    target_compile_definitions(benchmark_inference PRIVATE BENCHMARK_TFLITE)
endif()
if(BENCHMARK_ORT)
    target_compile_definitions(benchmark_inference PRIVATE BENCHMARK_ORT)
endif()
if(BENCHMARK_TVM)
    target_compile_definitions(benchmark_inference PRIVATE BENCHMARK_TVM)
endif()

target_link_libraries(benchmark_inference
    PRIVATE
    benchmark_common
    ${BACKEND_LIBS}
)

# Single operator benchmark executable
add_executable(single_op_benchmark
    single_op_benchmark.cpp
    ${BACKEND_SRCS}
)
target_compile_options(single_op_benchmark PRIVATE -fexceptions)

if(BENCHMARK_NCNN)
    target_compile_definitions(single_op_benchmark PRIVATE BENCHMARK_NCNN)
endif()
if(BENCHMARK_MNN)
    target_compile_definitions(single_op_benchmark PRIVATE BENCHMARK_MNN)
endif()
if(BENCHMARK_ORT)
    target_compile_definitions(single_op_benchmark PRIVATE BENCHMARK_ORT)
endif()

target_link_libraries(single_op_benchmark
    PRIVATE
    benchmark_common
    ${BACKEND_LIBS}
)
