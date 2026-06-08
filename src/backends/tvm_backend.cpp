#include "tvm_backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>

TVMBackend::~TVMBackend() { deinit(); }

bool TVMBackend::init(const BenchmarkConfig& config) {
    const std::string& so_path = config.model_path;
    printf("TVM: loading: %s\n", so_path.c_str());

    // ── 0. 确保 TVM 运行时符号全局可见 ──
    // Android 上需要使用绝对路径 + RTLD_GLOBAL
    void* h1 = dlopen("./libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h1) {
        h1 = dlopen("/data/local/tmp/benchmark/libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!h1) printf("TVM: WARNING dlopen(libtvm_ffi.so) failed: %s\n", dlerror());
    void* h2 = dlopen("./libtvm_runtime.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h2) {
        h2 = dlopen("/data/local/tmp/benchmark/libtvm_runtime.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!h2) printf("TVM: WARNING dlopen(libtvm_runtime.so) failed: %s\n", dlerror());

    // ── 1. Load model .so ──
    try {
        auto* pmod = new tvm::ffi::Module(tvm::ffi::Module::LoadFromFile(so_path));
        if (!pmod->defined()) { delete pmod; printf("TVM: ERROR LoadFromFile\n"); return false; }
        mod_ = pmod;
    } catch (const std::exception& e) {
        printf("TVM: LoadFromFile exception: %s\n", e.what());
        return false;
    }

    // ── 2. Create VM ──
    try {
        auto* pmod = (tvm::ffi::Module*)mod_;
        auto vm_load_opt = (*pmod)->GetFunction("vm_load_executable");
        if (!vm_load_opt.defined()) { printf("TVM: vm_load_executable not found\n"); return false; }

        auto vm_result = vm_load_opt.value()();
        auto vm_opt = std::move(vm_result).as<tvm::ffi::Module>();
        if (!vm_opt.has_value()) { printf("TVM: VM result is not Module\n"); return false; }

        auto* pvm = new tvm::ffi::Module(std::move(vm_opt.value()));
        if (!pvm->defined()) { delete pvm; printf("TVM: VM invalid\n"); return false; }
        vm_ = pvm;
    } catch (const std::exception& e) {
        printf("TVM: VM creation exception: %s\n", e.what());
        return false;
    }

    // ── 3. vm_initialization — 必须最先调用，初始化设备/内存分配器 ──
    try {
        auto* pvm = (tvm::ffi::Module*)vm_;
        auto init_fn_opt = (*pvm)->GetFunction("vm_initialization");
        if (!init_fn_opt.defined()) {
            printf("TVM: ERROR vm_initialization not found\n");
            return false;
        }
        init_fn_opt.value()(1, 0, 2);
    } catch (const std::exception& e) {
        printf("TVM: vm_initialization exception: %s\n", e.what());
        return false;
    }

    // ── 4. Get VM stateful API functions ──
    try {
        auto* pvm = (tvm::ffi::Module*)vm_;

        auto si_fn = (*pvm)->GetFunction("set_input");
        if (!si_fn.defined()) { printf("TVM: set_input not found\n"); return false; }
        set_input_ = new tvm::ffi::Function(si_fn.value());

        auto is_fn = (*pvm)->GetFunction("invoke_stateful");
        if (!is_fn.defined()) { printf("TVM: invoke_stateful not found\n"); return false; }
        invoke_ = new tvm::ffi::Function(is_fn.value());

        auto go_fn = (*pvm)->GetFunction("get_output");
        if (!go_fn.defined()) go_fn = (*pvm)->GetFunction("get_outputs");
        if (!go_fn.defined()) { printf("TVM: get_output not found\n"); return false; }
        get_outputs_ = new tvm::ffi::Function(go_fn.value());
    } catch (const std::exception& e) {
        printf("TVM: GetFunction exception: %s\n", e.what());
        return false;
    }

    // ── 5. 检测模型输入数量 ──
    // BERT 有 2 个输入 (input_ids + attention_mask)
    // 其他模型默认 1 个输入
    num_model_inputs_ = 1;
    if (so_path.find("bert") != std::string::npos) {
        num_model_inputs_ = 2;
    }

    // ── 6. Setup input shape ──
    input_size_ = 1;
    input_shape_.clear();
    for (auto s : config.input_shape) {
        input_shape_.push_back(static_cast<int64_t>(s));
        input_size_ *= s;
    }
    printf("TVM: init OK, model=%s input=%zu floats x%d inputs\n",
           so_path.c_str(), input_size_, num_model_inputs_);
    initialized_ = true;
    return true;
}

bool TVMBackend::infer(const std::vector<float>& input) {
    std::vector<float> unused_output;
    return infer_with_output(input, unused_output);
}

bool TVMBackend::infer_with_output(const std::vector<float>& input, std::vector<float>& output) {
    if (!initialized_) return false;
    if (input.size() != input_size_) return false;

    // 多输入模型 (BERT): 需要 int64 类型的 input_ids + attention_mask
    // 单输入模型: 直接使用 float32 数据
    std::vector<int64_t> input_ids_buf;
    std::vector<int64_t> attention_mask_buf;

    DLTensor dlt1, dlt2;

    if (num_model_inputs_ > 1) {
        // 将 float 输入转换为 int64 token IDs
        // 转换逻辑与 ORT 后端完全一致，保证精度可比
        input_ids_buf.resize(input.size());
        attention_mask_buf.resize(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            int64_t val = static_cast<int64_t>(input[i] * 1000.0f) % 30521;
            if (val < 0) val = -val;
            if (val == 0) val = 101;  // [CLS] token
            input_ids_buf[i] = val;
            attention_mask_buf[i] = val;  // 与 ORT 一致: 两者使用相同转换
        }

        // DLTensor for input_ids
        dlt1.data = input_ids_buf.data();
        dlt1.device = {kDLCPU, 0};
        dlt1.ndim = static_cast<int>(input_shape_.size());
        dlt1.dtype = {kDLInt, 64, 1};
        dlt1.shape = const_cast<int64_t*>(input_shape_.data());
        dlt1.strides = nullptr;
        dlt1.byte_offset = 0;

        // DLTensor for attention_mask
        dlt2.data = attention_mask_buf.data();
        dlt2.device = {kDLCPU, 0};
        dlt2.ndim = static_cast<int>(input_shape_.size());
        dlt2.dtype = {kDLInt, 64, 1};
        dlt2.shape = const_cast<int64_t*>(input_shape_.data());
        dlt2.strides = nullptr;
        dlt2.byte_offset = 0;
    } else {
        // 单输入: float32
        dlt1.data = const_cast<float*>(input.data());
        dlt1.device = {kDLCPU, 0};
        dlt1.ndim = static_cast<int>(input_shape_.size());
        dlt1.dtype = {kDLFloat, 32, 1};
        dlt1.shape = const_cast<int64_t*>(input_shape_.data());
        dlt1.strides = nullptr;
        dlt1.byte_offset = 0;
    }

    try {
        auto& si = *(tvm::ffi::Function*)set_input_;
        auto& is = *(tvm::ffi::Function*)invoke_;
        auto& go = *(tvm::ffi::Function*)get_outputs_;

        // Relax VM set_input: 单输入 set_input("main", tensor)
        //                    多输入 set_input("main", tensor1, tensor2)
        if (num_model_inputs_ == 1) {
            tvm::ffi::TensorView tv1(&dlt1);
            si("main", tv1);
        } else {
            tvm::ffi::TensorView tv1(&dlt1);
            tvm::ffi::TensorView tv2(&dlt2);
            si("main", tv1, tv2);
        }
        is("main");

        // 获取输出 — 兼容两种 VM 输出模式:
        // 模式 A (torch.export): get_outputs("main", 0) → Tensor (从 Array 取)
        // 模式 B (ONNX import):  get_output("main")    → Tensor 直接返回
        tvm::ffi::Any result;
        bool got_output = false;

        // 先尝试单输出模式 (ONNX import)
        try {
            result = go("main");
            got_output = true;
        } catch (...) {
            // 单输出模式失败，尝试多输出模式
        }

        if (!got_output) {
            // 多输出模式 (torch.export): 输出是 Array/Tuple
            result = go("main", 0);
        }

        auto tensor_opt = std::move(result).as<tvm::ffi::Tensor>();
        if (!tensor_opt.has_value()) return false;

        auto& t = tensor_opt.value();
        size_t n = static_cast<size_t>(t.numel());
        output.resize(n);
        std::memcpy(output.data(), t.data_ptr(), n * sizeof(float));
        return true;
    } catch (const std::exception& e) {
        printf("TVM: infer exception: %s\n", e.what());
        return false;
    } catch (...) {
        printf("TVM: infer unknown exception\n");
        return false;
    }
}

void TVMBackend::deinit() {
    delete (tvm::ffi::Function*)get_outputs_;
    delete (tvm::ffi::Function*)invoke_;
    delete (tvm::ffi::Function*)set_input_;
    delete (tvm::ffi::Module*)vm_;
    delete (tvm::ffi::Module*)mod_;
    set_input_ = invoke_ = get_outputs_ = vm_ = mod_ = nullptr;
    initialized_ = false;
}
