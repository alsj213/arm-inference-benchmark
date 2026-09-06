# LLM benchmark target for llama.cpp + MNN LLM + MobileLLM
if(BENCHMARK_LLAMACPP OR BENCHMARK_MNN OR BENCHMARK_MOBILELLM)
    add_executable(llm_benchmark llm_benchmark.cpp)
    target_compile_options(llm_benchmark PRIVATE -fexceptions)
    target_include_directories(llm_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
    if(BENCHMARK_LLAMACPP)
        target_compile_definitions(llm_benchmark PRIVATE BENCHMARK_LLAMACPP)
    endif()
    if(BENCHMARK_MNN)
        target_compile_definitions(llm_benchmark PRIVATE BENCHMARK_MNN)
    endif()
    if(BENCHMARK_MOBILELLM)
        target_compile_definitions(llm_benchmark PRIVATE BENCHMARK_MOBILELLM)
        # mobilellm 目标自带 PUBLIC include 传播 (third_party/MobileLLM/include)
        target_link_libraries(llm_benchmark PRIVATE mobilellm m)
    endif()
    target_link_libraries(llm_benchmark PRIVATE benchmark_common)
endif()

# MNN VL multimodal test
if(BENCHMARK_MNN)
    add_executable(mnn_vl_test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/mnn_vl_test.cpp
    )
    target_include_directories(mnn_vl_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/MNN/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/MNN/transformers/llm/engine/include
    )
    target_link_libraries(mnn_vl_test PRIVATE MNN)
endif()
