# third_party CMake configuration

# ============================================================================
# MNN — 核心推理框架 (CPU + GPU)
# ============================================================================
if(BENCHMARK_MNN)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/MNN)
        set(MNN_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
        set(MNN_SEP_BUILD OFF CACHE BOOL "" FORCE)
        set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
        set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
        set(MNN_KLEIDIAI OFF CACHE BOOL "Disable KleidiAI for older NDK compat" FORCE)
        set(MNN_BUILD_LLM ON CACHE BOOL "Enable LLM support" FORCE)
        set(MNN_LLM_BUILD_DEMO OFF CACHE BOOL "" FORCE)
        set(MNN_BUILD_FOR_ANDROID_COMMAND ON CACHE BOOL "Skip post-build copy on Android cross-compile" FORCE)
        add_subdirectory(MNN)
        set(MNN_FOUND TRUE)
        message(STATUS "Found MNN (with LLM support)")
    else()
        message(WARNING "MNN not found, please add as git submodule")
        set(BENCHMARK_MNN OFF)
    endif()
endif()

# ============================================================================
# ONNX Runtime v1.21.0 — 精度基准 + 跨框架验证
# ============================================================================
if(BENCHMARK_ORT)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/cmake/CMakeLists.txt)
        add_library(onnxruntime SHARED IMPORTED GLOBAL)
        target_include_directories(onnxruntime INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include/onnxruntime/core/session
            ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include/onnxruntime
        )

        # Select ORT library based on build type
        if(ANDROID)
            if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                set(ORT_LIB ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/build/Android/Debug/libonnxruntime.so)
            else()
                set(ORT_LIB ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/build/Android/Release/libonnxruntime.so)
            endif()
        else()
            if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                find_library(ORT_LIB NAMES onnxruntime PATHS ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/build/Linux/Debug)
            else()
                find_library(ORT_LIB NAMES onnxruntime PATHS ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/build/Linux/Release)
            endif()
        endif()

        if(EXISTS ${ORT_LIB})
            set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ORT_LIB})
            target_link_libraries(onnxruntime INTERFACE ${ORT_LIB} -pthread -ldl -lstdc++)
            set(ONNXRUNTIME_FOUND TRUE)
            message(STATUS "Found ONNX Runtime v1.21.0 (${CMAKE_BUILD_TYPE}): ${ORT_LIB}")
        else()
            message(WARNING "ONNX Runtime library not found for target architecture: ${ORT_LIB}")
            set(BENCHMARK_ORT OFF)
        endif()
    else()
        message(WARNING "ONNX Runtime submodule not found. Run: git submodule update --init third_party/onnxruntime")
        set(BENCHMARK_ORT OFF)
    endif()
endif()

# ============================================================================
# Apache TVM v0.15.0 — Relay + GraphExecutor
# ============================================================================
if(BENCHMARK_TVM)
    set(TVM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tvm)
    if(ANDROID)
        set(TVM_LIB ${TVM_ROOT}/build_android/libtvm_runtime.so)
    else()
        set(TVM_LIB ${TVM_ROOT}/build/libtvm_runtime.so)
    endif()

    if(EXISTS ${TVM_LIB})
        add_library(tvm_runtime INTERFACE)
        target_include_directories(tvm_runtime INTERFACE ${TVM_ROOT}/include)
        target_include_directories(tvm_runtime INTERFACE ${TVM_ROOT}/3rdparty/dlpack/include)
        target_include_directories(tvm_runtime INTERFACE ${TVM_ROOT}/3rdparty/dmlc-core/include)
        target_link_libraries(tvm_runtime INTERFACE ${TVM_LIB})
        set(TVM_FOUND TRUE)
        message(STATUS "Found TVM Runtime (v0.15.0): ${TVM_LIB}")
    else()
        message(WARNING "TVM Runtime not found at ${TVM_LIB}")
        set(BENCHMARK_TVM OFF)
    endif()
endif()

# ============================================================================
# llama.cpp — LLM 推理引擎
# ============================================================================
if(BENCHMARK_LLAMACPP)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)
        set(LLAMA_STATIC ON CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
        set(LLAMA_NATIVE OFF CACHE BOOL "" FORCE)
        if(ANDROID)
            set(LLAMA_ANDROID ON CACHE BOOL "" FORCE)
        endif()
        add_subdirectory(llama.cpp)
        set(LLAMACPP_FOUND TRUE)
        message(STATUS "Found llama.cpp for LLM inference")
    else()
        message(WARNING "llama.cpp not found")
        set(BENCHMARK_LLAMACPP OFF)
    endif()
endif()
