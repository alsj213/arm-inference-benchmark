# src CMake configuration

# Enable exceptions for ONNX Runtime and TVM
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -UNO_EXCEPTIONS")

# Common library - STATIC to avoid linker issues with backend vtable symbols
add_library(benchmark_common STATIC
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
if(BENCHMARK_TNN)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/TNN/include)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/TNN/source)
endif()
if(BENCHMARK_ORT)
target_include_directories(benchmark_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/include/onnxruntime/core/session
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/include/onnxruntime
)
endif()
if(BENCHMARK_TVM)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tvm/include)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tvm/3rdparty/tvm-ffi/include)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include)
endif()
if(BENCHMARK_LLAMACPP)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/include)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/src)
endif()
if(BENCHMARK_MINDSPORE_LITE)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/mindspore_lite)
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
if(BENCHMARK_TNN)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_TNN)
endif()
if(BENCHMARK_ORT)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_ORT)
endif()
if(BENCHMARK_TVM)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_TVM)
endif()
if(BENCHMARK_LLAMACPP)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_LLAMACPP)
endif()
if(BENCHMARK_MINDSPORE_LITE)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_MINDSPORE_LITE)
endif()

# Backend sources - add to benchmark_common STATIC library so vtable symbols are resolved
set(BACKEND_SRCS)
set(BACKEND_LIBS)

if(BENCHMARK_NCNN)
    list(APPEND BACKEND_SRCS backends/ncnn_backend.cpp)
    list(APPEND BACKEND_LIBS ncnn)
endif()

if(BENCHMARK_MNN)
    list(APPEND BACKEND_SRCS backends/mnn_backend.cpp backends/mnn_llm_backend.cpp)
    list(APPEND BACKEND_LIBS MNN)
    # MNN LLM engine header (for llm_benchmark / mnn_llm_backend)
    target_include_directories(benchmark_common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/MNN/transformers/llm/engine/include)
endif()

if(BENCHMARK_TFLITE)
    list(APPEND BACKEND_SRCS backends/tflite_backend.cpp)
    list(APPEND BACKEND_LIBS tensorflow-lite)
endif()

if(BENCHMARK_TNN)
    list(APPEND BACKEND_SRCS backends/tnn_backend.cpp)
    list(APPEND BACKEND_LIBS TNN)
endif()

if(BENCHMARK_ORT)
    list(APPEND BACKEND_SRCS backends/ort_backend.cpp)
    list(APPEND BACKEND_LIBS onnxruntime)
endif()

if(BENCHMARK_TVM)
    list(APPEND BACKEND_SRCS backends/tvm_backend.cpp)
endif()

if(BENCHMARK_LLAMACPP)
    list(APPEND BACKEND_SRCS backends/llamacpp_backend.cpp)
    list(APPEND BACKEND_LIBS llama)
endif()

if(BENCHMARK_MINDSPORE_LITE)
    list(APPEND BACKEND_SRCS backends/mindspore_lite_backend.cpp)
    list(APPEND BACKEND_LIBS mindspore_lite)
endif()

# Add backend sources to benchmark_common so create_backend() can resolve symbols
target_sources(benchmark_common PRIVATE ${BACKEND_SRCS})
target_link_libraries(benchmark_common PUBLIC ${BACKEND_LIBS})

# Generate model registry from models.json
set(MODELS_JSON ${CMAKE_SOURCE_DIR}/models.json)
set(MODEL_GEN_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/model_registry.h)

add_custom_command(
    OUTPUT ${MODEL_GEN_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_model_header.py ${MODELS_JSON} ${MODEL_GEN_HEADER}
    DEPENDS ${MODELS_JSON} ${CMAKE_SOURCE_DIR}/scripts/generate_model_header.py
    COMMENT "Generating model registry from models.json"
)
add_custom_target(generate_models DEPENDS ${MODEL_GEN_HEADER})

# Main executable
add_executable(benchmark_inference
    main.cpp
)

add_dependencies(benchmark_inference generate_models)
target_include_directories(benchmark_inference PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Inject git commit hash for --version
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_COMMIT_HASH)
    set(GIT_COMMIT_HASH "unknown")
endif()
message(STATUS "Git commit: ${GIT_COMMIT_HASH}")
target_compile_definitions(benchmark_inference PRIVATE
    GIT_COMMIT_HASH="${GIT_COMMIT_HASH}"
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
if(BENCHMARK_TNN)
    target_compile_definitions(benchmark_inference PRIVATE BENCHMARK_TNN)
endif()
if(BENCHMARK_ORT)
    target_compile_definitions(benchmark_inference PRIVATE BENCHMARK_ORT)
endif()
if(BENCHMARK_TVM)
    target_compile_definitions(benchmark_inference PRIVATE BENCHMARK_TVM)
    target_link_libraries(benchmark_inference PRIVATE tvm_runtime)
endif()
if(BENCHMARK_MINDSPORE_LITE)
    target_compile_definitions(benchmark_inference PRIVATE BENCHMARK_MINDSPORE_LITE)
endif()

# Link against all required libraries (backends already linked to benchmark_common)
target_link_libraries(benchmark_inference
    PRIVATE
    benchmark_common
)

# Single operator benchmark executable
add_executable(single_op_benchmark
    single_op_benchmark.cpp
)
target_compile_options(single_op_benchmark PRIVATE -fexceptions)

# Copy compile definitions from main executable
if(BENCHMARK_MNN)
    target_compile_definitions(single_op_benchmark PRIVATE BENCHMARK_MNN)
endif()
if(BENCHMARK_ORT)
    target_compile_definitions(single_op_benchmark PRIVATE BENCHMARK_ORT)
endif()
if(BENCHMARK_TVM)
    target_compile_definitions(single_op_benchmark PRIVATE BENCHMARK_TVM)
endif()

target_link_libraries(single_op_benchmark
    PRIVATE
    benchmark_common
)

# Link backend runtime libraries for symbol resolution
if(BENCHMARK_TVM)
    target_link_libraries(single_op_benchmark PRIVATE tvm_runtime)
endif()

# LLM benchmark executable for llama.cpp + MNN LLM
if(BENCHMARK_LLAMACPP OR BENCHMARK_MNN)
    add_executable(llm_benchmark
        llm_benchmark.cpp
    )
    target_compile_options(llm_benchmark PRIVATE -fexceptions)
    if(BENCHMARK_LLAMACPP)
        target_compile_definitions(llm_benchmark PRIVATE BENCHMARK_LLAMACPP)
    endif()
    if(BENCHMARK_MNN)
        target_compile_definitions(llm_benchmark PRIVATE BENCHMARK_MNN)
    endif()
    target_link_libraries(llm_benchmark
        PRIVATE
        benchmark_common
    )
endif()
