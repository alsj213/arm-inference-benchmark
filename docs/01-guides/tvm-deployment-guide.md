# TVM Relax 端侧部署方案

> Apache TVM v0.24.0-dev Relax VM 在骁龙 865 (SM8250) 上的端到端部署指南  
> 最后更新: 2026-06-07 | 分支: `feat/tvm-integration`

---

## 目录

1. [架构概览](#1-架构概览)
2. [模型编译（Python → .so）](#2-模型编译)
3. [C++ 推理后端](#3-c-推理后端)
4. [构建系统集成](#4-构建系统集成)
5. [部署流程](#5-部署流程)
6. [踩坑记录 & 故障排除](#6-踩坑记录--故障排除)
7. [性能数据](#7-性能数据)

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    开发机 (x86_64 / WSL2)                │
│                                                         │
│  PyTorch Model ──► torch.export ──► Relax IRModule      │
│       │                                    │            │
│       │              ┌─────────────────────┘            │
│       │              ▼                                  │
│       │    relax.get_pipeline() + tvm.compile()         │
│       │              │                                  │
│       │              ▼ (LLVM 生成 ARM64 .o)              │
│       │    executable.export_library()                  │
│       │       + NDK aarch64-linux-android21-clang++     │
│       │              │                                  │
│       │              ▼                                  │
│       │    ┌──────────────────┐                         │
│       │    │ model_tvm.so     │  ← 参数嵌入 (keep_params│
│       │    │ (14.4 MB)        │     _as_input=False)    │
│       │    └──────────────────┘                         │
└───────┼─────────────────────────────────────────────────┘
        │  adb push
        ▼
┌─────────────────────────────────────────────────────────┐
│              骁龙 865 设备 (aarch64 / Android)           │
│                                                         │
│  benchmark_inference                                    │
│       │                                                 │
│       ├── libtvm_ffi.so       (TVM FFI C++ runtime)     │
│       ├── libtvm_runtime.so   (TVM 运行时)              │
│       │                                                 │
│       ▼                                                 │
│  TVMBackend::init()                                     │
│    1. dlopen("libtvm_ffi.so", RTLD_GLOBAL)              │
│    2. dlopen("libtvm_runtime.so", RTLD_GLOBAL)          │
│    3. Module::LoadFromFile("model_tvm.so")              │
│    4. vm_load_executable() → VM Module                  │
│    5. ⚠️ vm_initialization(kDLCPU, 0, POOLED_ALLOCATOR) │
│    6. Save: set_input / invoke_stateful / get_output    │
│                                                         │
│  TVMBackend::infer_with_output()                        │
│    set_input("main", DLTensor)                          │
│    → invoke_stateful("main")                            │
│    → get_output("main", 0)  ← Tensor                    │
│    → memcpy → std::vector<float>                        │
└─────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 参数处理 | `keep_params_as_input=False` | 参数嵌入 .so，C++ 只需传输入张量，简化调用 |
| 推理 API | Stateful API | `set_input` → `invoke_stateful` → `get_output` 能正确处理 PyTorch tuple 输出 |
| 输入传递 | `TensorView` (DLTensor*) | 零拷贝，外部管理内存，无所有权转移 |
| 设备初始化 | `vm_initialization(CPU, 0, POOLED)` | **必须调用**，否则 segfault |
| 符号可见性 | `dlopen(RTLD_GLOBAL)` + `LD_PRELOAD` | 模型 .so 依赖 TVM 运行时符号，需全局可见 |

---

## 2. 模型编译

### 2.1 环境准备

```bash
# 必需依赖
conda create -n tvm-build python=3.10
conda activate tvm-build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e third_party/tvm/3rdparty/tvm-ffi
pip install psutil pytest

# 环境变量
export TVM_ROOT=$(pwd)/third_party/tvm
export PYTHONPATH=$TVM_ROOT/python:$TVM_ROOT/3rdparty/tvm-ffi/python
export LD_LIBRARY_PATH=$TVM_ROOT/build/lib:$TVM_ROOT/build
```

### 2.2 编译脚本

`tools/tvm/compile_model_relax.py`:

```python
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# 1. 加载 PyTorch 模型
model = torchvision.models.mobilenet_v2(weights=...).eval()

# 2. torch.export 导出
example_input = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
exported = torch.export.export(model, example_input)

# 3. 转 Relax IR（参数嵌入）
mod = from_exported_program(exported, keep_params_as_input=False)

# 4. Relax 编译流水线
target = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "aarch64-linux-android",
    "mattr": ["+neon"]
})
pipeline = relax.get_pipeline()
with target:
    built_mod = pipeline(mod)
executable = tvm.compile(built_mod, target=target)

# 5. 导出 .so（NDK 交叉链接）
NDK_CXX = "/path/to/android-ndk/.../aarch64-linux-android21-clang++"
executable.export_library("model_tvm.so", cc=NDK_CXX)
```

### 2.3 运行编译

```bash
python3 tools/tvm/compile_model_relax.py mobilenetv2
# 输出: tools/tvm/compiled_models/mobilenetv2_tvm.so (14.4 MB)
```

### 2.4 支持模型

| 模型 | 输入 shape | 编译时间 | .so 大小 |
|------|-----------|---------|---------|
| MobileNetV2 | (1, 3, 224, 224) | ~2 min | 14.4 MB |
| ResNet18 | (1, 3, 224, 224) | 待编译 | — |
| YOLOv8n | (1, 3, 640, 640) | 待编译 | — |

---

## 3. C++ 推理后端

### 3.1 头文件 (`tvm_backend.h`)

```cpp
class TVMBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    bool infer_with_output(const std::vector<float>& input,
                           std::vector<float>& output) override;
    void deinit() override;
    std::string name() const override { return "TVM"; }

private:
    void* mod_ = nullptr;          // ffi::Module*  (编译产物)
    void* vm_ = nullptr;           // ffi::Module*  (Relax VM)
    void* set_input_ = nullptr;    // ffi::Function*
    void* invoke_ = nullptr;       // ffi::Function*
    void* get_outputs_ = nullptr;  // ffi::Function*
    std::vector<int64_t> input_shape_;
    size_t input_size_ = 0;
    bool initialized_ = false;
};
```

> **Why `void*`?** TVM v0.24.0 的 `ffi::Module` 类型是 **NOTNULLABLE**（无默认构造函数），不能作为 `std::make_unique` 成员。用 `void*` + heap allocation 绕过。

### 3.2 `init()` 流程

```cpp
bool TVMBackend::init(const BenchmarkConfig& config) {
    // ── 0. 符号全局可见 ──
    dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
    dlopen("libtvm_runtime.so", RTLD_NOW | RTLD_GLOBAL);

    // ── 1. 加载模型 .so ──
    auto* pmod = new tvm::ffi::Module(
        tvm::ffi::Module::LoadFromFile(so_path));

    // ── 2. 创建 VM ──
    auto vm_load = (*pmod)->GetFunction("vm_load_executable");
    auto vm_result = vm_load.value()();
    auto vm = std::move(vm_result).as<tvm::ffi::Module>();
    // → vm_ 保存

    // ── 3. ⚠️ 初始化设备（必须！否则 segfault） ──
    auto init_fn = (*vm)->GetFunction("vm_initialization");
    init_fn.value()(1, 0, 2);  // kDLCPU, device=0, POOLED_ALLOCATOR

    // ── 4. 获取 Stateful API 函数 ──
    set_input_ = new Function((*vm)->GetFunction("set_input").value());
    invoke_    = new Function((*vm)->GetFunction("invoke_stateful").value());
    get_outputs_ = new Function((*vm)->GetFunction("get_output").value());
}
```

### 3.3 `infer_with_output()` 流程

```cpp
bool TVMBackend::infer_with_output(const std::vector<float>& input,
                                    std::vector<float>& output) {
    // ── 构建 DLTensor 视图（零拷贝） ──
    DLTensor dlt;
    dlt.data = const_cast<float*>(input.data());
    dlt.device = {kDLCPU, 0};
    dlt.ndim = input_shape_.size();
    dlt.dtype = {kDLFloat, 32, 1};
    dlt.shape = const_cast<int64_t*>(input_shape_.data());
    dlt.strides = nullptr;
    dlt.byte_offset = 0;
    tvm::ffi::TensorView tv(&dlt);

    // ── Stateful API 推理 ──
    auto& si = *(Function*)set_input_;
    auto& is = *(Function*)invoke_;
    auto& go = *(Function*)get_outputs_;

    si("main", tv);               // 设置输入
    is("main");                   // 执行推理
    auto result = go("main", 0);  // 获取输出 (tuple 索引 0)

    // ── 提取输出数据 ──
    auto tensor = std::move(result).as<tvm::ffi::Tensor>();
    size_t n = tensor.value().numel();
    output.resize(n);
    std::memcpy(output.data(), tensor.value().data_ptr(), n * sizeof(float));
}
```

### 3.4 Stateful API vs 直接调用

| 方式 | API | 优点 | 缺点 |
|------|-----|------|------|
| **Stateful** | `set_input` → `invoke_stateful` → `get_output` | 输出类型明确（自动解 tuple） | 三步调用 |
| 直接调用 | `vm["main"](input)` | 简洁 | 输出可能是 `Array<Tensor>`，需手动判断类型 |

> **本项目使用 Stateful API**：PyTorch export 的输出是 tuple，`get_output("main", 0)` 直接按索引取出第一个 Tensor，避免类型判断。

---

## 4. 构建系统集成

### 4.1 根 CMakeLists.txt

```cmake
option(BENCHMARK_TVM "Enable Apache TVM benchmark" ON)  # 默认启用
```

### 4.2 third_party/CMakeLists.txt

```cmake
if(BENCHMARK_TVM)
    set(TVM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tvm)
    if(ANDROID)
        set(TVM_LIB ${TVM_ROOT}/build-android-v2/libtvm_runtime.so)
        set(TVM_FFI_LIB ${TVM_ROOT}/build-android-v2/lib/libtvm_ffi.so)
    endif()

    add_library(tvm_runtime INTERFACE)
    target_include_directories(tvm_runtime INTERFACE
        ${TVM_ROOT}/include
        ${TVM_ROOT}/3rdparty/tvm-ffi/include
        ${TVM_ROOT}/3rdparty/dlpack/include)
    target_link_libraries(tvm_runtime INTERFACE
        ${TVM_LIB} ${TVM_FFI_LIB})
endif()
```

### 4.3 src/CMakeLists.txt

```cmake
if(BENCHMARK_TVM)
    target_compile_definitions(benchmark_common PUBLIC BENCHMARK_TVM)
    list(APPEND BACKEND_SRCS backends/tvm_backend.cpp)
endif()

if(BENCHMARK_TVM)
    target_link_libraries(benchmark_inference PRIVATE tvm_runtime)
endif()
```

### 4.4 编译 Android

```bash
export ANDROID_NDK=/path/to/android-ndk
./scripts/build_android.sh
# 产物: build_android/src/benchmark_inference
```

### 4.5 编译 TVM 运行时

TVM 运行时（`libtvm_runtime.so` + `libtvm_ffi.so`）需要单独交叉编译：

```bash
cd third_party/tvm
mkdir build-android-v2 && cd build-android-v2
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_LLVM=OFF \
    -DUSE_MICRO=OFF
make -j$(nproc) tvm_runtime tvm_ffi
```

---

## 5. 部署流程

### 5.1 一键部署

```bash
# 设备目录结构:
# /data/local/tmp/
# ├── benchmark_inference
# ├── libtvm_ffi.so
# ├── libtvm_runtime.so
# └── tools/tvm/compiled_models/
#     └── mobilenetv2_tvm.so

# 推送
adb push build_android/src/benchmark_inference /data/local/tmp/
adb push third_party/tvm/build-android-v2/libtvm_runtime.so /data/local/tmp/
adb push third_party/tvm/build-android-v2/lib/libtvm_ffi.so /data/local/tmp/
adb push tools/tvm/compiled_models/mobilenetv2_tvm.so \
    /data/local/tmp/tools/tvm/compiled_models/

# 运行 (注意 LD_PRELOAD 顺序!)
adb shell "
    cd /data/local/tmp
    chmod +x benchmark_inference
    LD_LIBRARY_PATH=. \
    LD_PRELOAD=libtvm_ffi.so:libtvm_runtime.so \
    ./benchmark_inference --backend tvm --model mobilenetv2 \
        --precision fp32 --threads 1 --warmup 5 --runs 50
"
```

### 5.2 LD_PRELOAD 说明

| 环境变量 | 作用 | 必须 |
|---------|------|------|
| `LD_LIBRARY_PATH=.` | 让 dlopen 在 `/data/local/tmp` 找到 .so | ✅ |
| `LD_PRELOAD=libtvm_ffi.so:libtvm_runtime.so` | 以 RTLD_GLOBAL 预加载 TVM 运行时符号 | ✅ |

> **为什么需要 LD_PRELOAD?** 模型 .so 内部 `dlopen` 时需要解析 `TVMFFIErrorSetRaisedFromCStrParts` 等符号。Android bionic linker 不会自动跨 .so 解析，必须先用 RTLD_GLOBAL 加载 TVM 运行时。

---

## 6. 踩坑记录 & 故障排除

### 坑 #1: `vm["main"](input)` 直接 segfault

**现象**:
```
TVM: model loaded
TVM: VM created
TVM: entry fn 'main' OK
TVM: calling main...
Segmentation fault
```

**根因**: **`vm_initialization` 未调用**。Relax VM 必须先初始化设备和内存分配器，否则内部访问空指针。

**修复**:
```cpp
auto init_fn = vm->GetFunction("vm_initialization");
init_fn.value()(1, 0, 2);  // kDLCPU=1, device_id=0, POOLED_ALLOCATOR=2
```

### 坑 #2: `keep_params_as_input=True` 参数缺失

**现象**: 模型编译为 1MB .so + 14MB .npz 参数文件，调用 `main(input)` 崩溃。

**根因**: 当 `keep_params_as_input=True` 时，参数不嵌入 .so。MobileNetV2 有 158 个参数张量，需要全部传入 `main(input, p0, p1, ..., p157)`。

**修复**: 改为 `keep_params_as_input=False`，参数嵌入 .so (14.4MB)。调用简化为单参数。

### 坑 #3: `as<Tensor>()` 返回空

**现象**: 模型推理成功但无法提取输出。

**根因**: PyTorch `torch.export.export()` 输出是 tuple，TVM 编译为 `Array<Tensor>` 而非单 `Tensor`。

**修复**: 使用 Stateful API + 索引访问:
```cpp
get_output("main", 0)  // 取 tuple 第一个元素
```

### 坑 #4: NDK 交叉链接失败

**现象**:
```
x86_64-conda-linux-gnu/bin/ld: lib0.o: Relocations in generic ELF (EM: 183)
lib0.o: error adding symbols: file in wrong format
```

**根因**: LLVM 生成 ARM64 .o 文件，但 `export_library` 默认用主机 GCC 链接。

**修复**:
```python
executable.export_library("model.so",
    cc="aarch64-linux-android21-clang++")
```

### 坑 #5: `Module` NOTNULLABLE — 不能默认构造

**现象**: `std::make_unique<TVMBackend>()` 编译失败。

**根因**: TVM v0.24.0 的 `ffi::Module` 无默认构造函数（NOTNULLABLE）。

**修复**: 用 `void*` 指针 + heap allocation：
```cpp
void* mod_ = nullptr;  // ffi::Module*
// ...
mod_ = new tvm::ffi::Module(std::move(module));
```

### 坑 #6: TVM 0.24.0 不支持 CLI Target 字符串

**现象**: `ValueError: Cannot parse target string "llvm -mtriple=..."`

**根因**: v0.24.0 移除了 CLI target 字符串格式。

**修复**: 使用 JSON dict 格式：
```python
target = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "aarch64-linux-android",
    "mattr": ["+neon"]
})
```

### 故障排查清单

| 症状 | 检查项 |
|------|--------|
| segfault at `main()` | `vm_initialization` 是否调用？LD_PRELOAD 是否正确设置？ |
| `cannot locate symbol TVMFFI*` | `LD_PRELOAD=libtvm_ffi.so:libtvm_runtime.so` 必须先于主程序 |
| `Failed to load dynamic shared library` | 模型 .so 路径是否正确？文件是否推送到设备？ |
| `get_output cannot return a tuple` | 需要传索引参数: `get_output("main", 0)` |
| 输出为空 | 检查是否正确调用 `invoke_stateful` 后再取 `get_output` |
| 编译失败 `tvm/ffi/... not found` | 检查 CMake 头文件路径包含 `tvm-ffi/include` 和 `dlpack/include` |

---

## 7. 性能数据

### 7.1 测试环境

| 项目 | 详情 |
|------|------|
| 设备 | Redmi K30S Ultra (M2007J3SC) |
| SoC | 骁龙 865 (SM8250) |
| CPU | 1×A77@2.84GHz + 3×A77@2.42GHz + 4×A55@1.8GHz |
| 测试条件 | 单线程, fp32, warmup=5, runs=50 |

### 7.2 MobileNetV2 结果

| 指标 | 值 |
|------|-----|
| **推理时间** | Min: 512.79ms, Mean: **513.63ms**, Max: 514.54ms |
| **标准差** | 0.40ms (CV=0.08%) |
| **吞吐** | 1.95 FPS |
| **初始化** | 22.84ms |
| **精度** (vs ORT) | 余弦相似度 **1.000000** |
| **MAE** | 0.000002 |
| **Max AE** | 0.000011 |

### 7.3 精度验证

```
Reference stats: Min: -6.1199, Max: 6.0787, Mean: 0.0001
TVM output:      1000 elements

✅ [Accuracy] TVM: PASSED
    Cosine Similarity:  1.000000 (Excellent)
    Mean Absolute Error: 0.000002
    Max Absolute Error:  0.000011
    Mean Relative Error: 0.0007%
```

### 7.4 待完成对比

| 后端 | MobileNetV2 | 精度 |
|------|------------|------|
| **TVM** | 513ms | ✅ 1.000000 |
| MNN | 待测 | — |
| ONNX Runtime | 待测 | Baseline |

---

## 附录: 目录结构

```
benchmark/
├── tools/tvm/
│   ├── compile_model_relax.py    # 模型编译脚本
│   ├── compiled_models/
│   │   ├── .gitignore            # 忽略 *.so *.o workspace/
│   │   └── mobilenetv2_tvm.so    # 编译产物 (14.4MB)
│   └── docker/                   # Docker 备选方案
├── src/backends/
│   ├── tvm_backend.h             # TVM 后端头文件
│   └── tvm_backend.cpp           # TVM 后端实现
├── third_party/tvm/
│   ├── build-android-v2/
│   │   ├── libtvm_runtime.so     # TVM 运行时 (ARM64)
│   │   └── lib/libtvm_ffi.so     # TVM FFI 库 (ARM64)
│   ├── include/                  # TVM C API 头文件
│   └── 3rdparty/
│       ├── tvm-ffi/include/      # TVM FFI C++ 头文件
│       └── dlpack/include/       # DLPack 头文件
└── docs/
    └── tvm-deployment-guide.md   # 本文档
```

---

> **维护者**: alsj213  
> **参考**: [Apache TVM 官方文档](https://tvm.apache.org/docs/) · [TVM FFI GitHub](https://github.com/apache/tvm-ffi)
