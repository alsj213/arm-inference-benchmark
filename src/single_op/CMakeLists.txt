# Single operator benchmark target
add_executable(single_op_benchmark single_op_benchmark.cpp)
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

target_link_libraries(single_op_benchmark PRIVATE benchmark_common)

# Link backend runtime libraries for symbol resolution
if(BENCHMARK_TVM)
    target_link_libraries(single_op_benchmark PRIVATE tvm_runtime)
endif()
