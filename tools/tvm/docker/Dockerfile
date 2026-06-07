# TVM Codegen Docker — 编译 ONNX → ARM .so
# 用法: docker build -t tvm-codegen -f tools/tvm/docker/Dockerfile .
FROM docker.m.daocloud.io/library/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TVM_HOME=/tvm

# ═══ 1. 系统依赖 (阿里云镜像 + Python 3.12) ═══
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3.12 python3.12-dev python3.12-venv \
    llvm-15 llvm-15-dev libssl-dev libomp-dev wget curl \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# 确保 pip 可用 (Python 3.12 通过 ensurepip)
RUN python3.12 -m ensurepip --upgrade 2>&1 || true

ENV LLVM_CONFIG=/usr/bin/llvm-config-15

# ═══ 2. TVM 编译 (C++ 层，不依赖 Python 包) ═══
COPY third_party/tvm $TVM_HOME
WORKDIR $TVM_HOME
RUN rm -rf build && mkdir build && cd build && \
    cmake .. \
    -DUSE_LLVM=$LLVM_CONFIG \
    -DUSE_RPC=ON \
    -DUSE_THREADS=ON \
    -DUSE_OPENMP=ON \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1 | tail -5 && \
    make -j8 2>&1 | grep -E "error:|Error|Built target|Linking" | head -30 && \
    echo "TVM build done" && \
    ls -lh libtvm.so libtvm_runtime.so 2>/dev/null

# ═══ 3. Python 依赖 (放在 TVM 编译后，不破坏缓存) ═══
RUN python3.12 -m pip install --break-system-packages --no-cache-dir \
    numpy onnx decorator attrs psutil scipy pytest typing_extensions

# ═══ 4. 环境配置 ═══
ENV PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/3rdparty/tvm-ffi/python
ENV LD_LIBRARY_PATH=$TVM_HOME/build

RUN ls -lh $TVM_HOME/build/libtvm.so && \
    ln -sf $TVM_HOME/build/lib/libtvm_ffi.so $TVM_HOME/build/libtvm_ffi.so && \
    ln -sf $TVM_HOME/build/lib/libtvm_ffi_testing.so $TVM_HOME/build/libtvm_ffi_testing.so && \
    find $TVM_HOME/3rdparty/tvm-ffi/build -name '*.abi3.so' -exec sh -c 'ln -sf {} $TVM_HOME/3rdparty/tvm-ffi/python/tvm_ffi/$(basename {})' \; && \
    echo "TVM libraries ready"

# ═══ 5. 编译入口 ═══
COPY tools/tvm/docker/compile_model.py /compile_model.py
RUN chmod +x /compile_model.py
VOLUME ["/workspace"]
WORKDIR /workspace
ENTRYPOINT ["python3", "/compile_model.py"]
