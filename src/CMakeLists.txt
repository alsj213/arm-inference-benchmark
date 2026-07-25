# src CMake configuration
# 三条主线分发：cnn/ llm/ single_op

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
if(BENCHMARK_MNN)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/MNN/include)
endif()
if(BENCHMARK_ORT)
target_include_directories(benchmark_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/include/onnxruntime/core/session
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/include/onnxruntime
)
endif()
if(BENCHMARK_TVM)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tvm/include)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tvm/3rdparty/dlpack/include)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tvm/3rdparty/dmlc-core/include)
endif()
if(BENCHMARK_LLAMACPP)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/include)
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/src)
endif()

# Add compile definitions for enabled backends
if(BENCHMARK_MNN)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_MNN)
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

# Backend sources - add to benchmark_common STATIC library so vtable symbols are resolved
set(BACKEND_SRCS)
set(BACKEND_LIBS)

if(BENCHMARK_MNN)
    list(APPEND BACKEND_SRCS backends/mnn_backend.cpp backends/mnn_llm_backend.cpp)
    list(APPEND BACKEND_LIBS MNN)
    # MNN LLM engine header (for llm_benchmark / mnn_llm_backend)
    target_include_directories(benchmark_common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/MNN/transformers/llm/engine/include)
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

# Add backend sources to benchmark_common so create_backend() can resolve symbols
target_sources(benchmark_common PRIVATE ${BACKEND_SRCS})
target_link_libraries(benchmark_common PUBLIC ${BACKEND_LIBS})

# Generate model registry from models.json
set(MODELS_JSON ${CMAKE_SOURCE_DIR}/models.json)
set(MODEL_GEN_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/model_registry.h)

add_custom_command(
    OUTPUT ${MODEL_GEN_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/utils/generate-model-header.py ${MODELS_JSON} ${MODEL_GEN_HEADER}
    DEPENDS ${MODELS_JSON} ${CMAKE_SOURCE_DIR}/scripts/utils/generate-model-header.py
    COMMENT "Generating model registry from models.json"
)
add_custom_target(generate_models DEPENDS ${MODEL_GEN_HEADER})

target_compile_options(benchmark_common PUBLIC -fexceptions)

# Support for model source files via include directories
target_include_directories(benchmark_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models)

# ============================================================================
# 三条主线子目录分发
# ============================================================================
add_subdirectory(cnn)
add_subdirectory(llm)
add_subdirectory(single_op)
