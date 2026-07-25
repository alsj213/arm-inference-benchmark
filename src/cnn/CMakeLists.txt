# CNN inference benchmark target (ORT, MNN, TVM)
add_executable(benchmark_inference main.cpp)

add_dependencies(benchmark_inference generate_models)
target_include_directories(benchmark_inference PRIVATE ${CMAKE_BINARY_DIR}/src/generated)
target_include_directories(benchmark_inference PRIVATE ${CMAKE_SOURCE_DIR}/third_party)

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

# Copy compile definitions from benchmark_common to benchmark_inference
# This ensures all backend macros are visible in main.cpp
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

# Link against benchmark_common and all its backend dependencies
target_link_libraries(benchmark_inference PRIVATE benchmark_common)
