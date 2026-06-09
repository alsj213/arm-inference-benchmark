#include "tvm_backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

TVMBackend::~TVMBackend() { deinit(); }

// 从文件读取字符串
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool TVMBackend::init(const BenchmarkConfig& config) {
    const std::string& so_path = config.model_path;
    printf("TVM: loading model: %s\n", so_path.c_str());

    // ── 0. 确保 TVM 运行时符号全局可见 ──
    void* h = dlopen("./libtvm_runtime.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        h = dlopen("/data/local/tmp/benchmark/libtvm_runtime.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!h) printf("TVM: WARNING dlopen(libtvm_runtime.so) failed: %s\n", dlerror());
    else   h_rt_ = h;

    // ── 1. 加载完整模型 .so (包含 graph + kernel + params) ──
    // relay.build() + export_library() 生成的 .so 是一个 GraphExecutorFactoryModule
    // 加载后调用 default(device) 获取 GraphExecutor (自动加载 params)
    tvm::runtime::Module factory_mod;
    try {
        factory_mod = tvm::runtime::Module::LoadFromFile(so_path);
        printf("TVM: module loaded: %s\n", so_path.c_str());
    } catch (const std::exception& e) {
        printf("TVM: LoadFromFile exception: %s\n", e.what());
        return false;
    }

    // ── 2. 从 factory 创建 GraphExecutor ──
    // 尝试 default 函数 — factory 模块的主要入口
    auto default_fn = factory_mod.GetFunction("default");
    if (default_fn == nullptr) {
        printf("TVM: 'default' not found, trying combined load approach...\n");

        // Fallback: 尝试加载 graph JSON + kernel .so 组合
        std::string json_path = so_path;
        size_t pos = json_path.rfind("_tvm.so");
        if (pos != std::string::npos) {
            json_path = json_path.substr(0, pos) + "_graph.json";
        }
        std::string graph_json = read_file(json_path);
        if (graph_json.empty()) {
            printf("TVM: ERROR cannot read graph JSON either: %s\n", json_path.c_str());
            return false;
        }

        std::string kernel_path = so_path;
        if (pos != std::string::npos) {
            kernel_path = kernel_path.substr(0, pos) + "_kernels.so";
        }
        auto kernel_mod = tvm::runtime::Module::LoadFromFile(kernel_path);

        auto* reg = tvm::runtime::Registry::Get("tvm.graph_executor.create");
        if (reg == nullptr) {
            printf("TVM: ERROR Registry::Get failed\n");
            return false;
        }
        auto ge_mod = (*reg)(graph_json, kernel_mod, static_cast<int>(kDLCPU), 0);
        auto* pmod = new tvm::runtime::Module(std::move(ge_mod));
        mod_ = pmod;
        printf("TVM: GraphExecutor created (fallback path)\n");
    } else {
        // Standard path: factory.default(device) → GraphExecutor
        try {
            DLDevice dev{kDLCPU, 0};
            auto ge_mod = default_fn(dev);
            auto* pmod = new tvm::runtime::Module(std::move(ge_mod));
            mod_ = pmod;
            printf("TVM: GraphExecutor created via default(device)\n");
        } catch (const std::exception& e) {
            printf("TVM: default(device) exception: %s\n", e.what());
            return false;
        }
    }

    auto* graph_mod = (tvm::runtime::Module*)mod_;

    // ── 4. 获取 GraphExecutor API ──
    try {
        auto si = graph_mod->GetFunction("set_input");
        if (si == nullptr) { printf("TVM: set_input not found\n"); return false; }
        set_input_ = new tvm::runtime::PackedFunc(si);

        auto rn = graph_mod->GetFunction("run");
        if (rn == nullptr) { printf("TVM: run not found\n"); return false; }
        run_ = new tvm::runtime::PackedFunc(rn);

        auto go = graph_mod->GetFunction("get_output");
        if (go == nullptr) { printf("TVM: get_output not found\n"); return false; }
        get_output_ = new tvm::runtime::PackedFunc(go);
    } catch (const std::exception& e) {
        printf("TVM: GetFunction exception: %s\n", e.what());
        return false;
    }

    // ── 5. 获取输入名称 ──
    // 优先使用 config 传入的 input_name (从 models.json 读取)
    if (!config.input_name.empty()) {
        input_name_ = config.input_name;
    }
    // 回退: 根据模型文件名推断
    if (input_name_.empty()) {
        if (so_path.find("bert") != std::string::npos) {
            input_name_ = "input_ids";
        } else if (so_path.find("resnet50") != std::string::npos) {
            input_name_ = "data";
        } else if (so_path.find("yolov8n") != std::string::npos) {
            input_name_ = "images";
        } else {
            input_name_ = "input";
        }
    }
    printf("TVM: input_name=%s\n", input_name_.c_str());

    // ── 6. 检测模型输入数量 ──
    num_model_inputs_ = 1;
    if (so_path.find("bert") != std::string::npos) {
        num_model_inputs_ = 2;
    }

    // ── 7. Setup input shape ──
    input_size_ = 1;
    input_shape_.clear();
    for (auto s : config.input_shape) {
        input_shape_.push_back(static_cast<int64_t>(s));
        input_size_ *= s;
    }

    if (num_model_inputs_ == 1) {
        input_ndarray_ = new tvm::runtime::NDArray(
            tvm::runtime::NDArray::Empty(
                input_shape_, {kDLFloat, 32, 1}, {kDLCPU, 0}));
    }

    printf("TVM: init OK, model=%s input=%zu floats name=%s\n",
           so_path.c_str(), input_size_, input_name_.c_str());
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

    auto& si = *(tvm::runtime::PackedFunc*)set_input_;
    auto& rn = *(tvm::runtime::PackedFunc*)run_;
    auto& go = *(tvm::runtime::PackedFunc*)get_output_;

    try {
        if (num_model_inputs_ == 1) {
            auto& nd = *(tvm::runtime::NDArray*)input_ndarray_;
            std::memcpy(nd->data, input.data(), input_size_ * sizeof(float));
            si(input_name_, nd);
            rn();
            tvm::runtime::NDArray out = go(0);

            size_t n = static_cast<size_t>(out->shape[0]);
            for (int i = 1; i < out->ndim; i++) n *= out->shape[i];
            output.resize(n);
            std::memcpy(output.data(), out->data, n * sizeof(float));
        } else {
            std::vector<int64_t> input_ids_buf(input.size());
            std::vector<int64_t> attention_mask_buf(input.size());
            for (size_t i = 0; i < input.size(); i++) {
                int64_t val = static_cast<int64_t>(input[i] * 1000.0f) % 30521;
                if (val < 0) val = -val;
                if (val == 0) val = 101;
                input_ids_buf[i] = val;
                attention_mask_buf[i] = val;
            }
            auto ids_nd = tvm::runtime::NDArray::Empty(
                input_shape_, {kDLInt, 64, 1}, {kDLCPU, 0});
            auto mask_nd = tvm::runtime::NDArray::Empty(
                input_shape_, {kDLInt, 64, 1}, {kDLCPU, 0});
            std::memcpy(ids_nd->data, input_ids_buf.data(), input.size() * sizeof(int64_t));
            std::memcpy(mask_nd->data, attention_mask_buf.data(), input.size() * sizeof(int64_t));
            si("input_ids", ids_nd);
            si("attention_mask", mask_nd);
            rn();
            tvm::runtime::NDArray out = go(0);

            size_t n = static_cast<size_t>(out->shape[0]);
            for (int i = 1; i < out->ndim; i++) n *= out->shape[i];
            output.resize(n);
            std::memcpy(output.data(), out->data, n * sizeof(float));
        }
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
    delete (tvm::runtime::NDArray*)input_ndarray_;
    delete (tvm::runtime::PackedFunc*)get_output_;
    delete (tvm::runtime::PackedFunc*)run_;
    delete (tvm::runtime::PackedFunc*)set_input_;
    delete (tvm::runtime::Module*)mod_;
    if (h_rt_) dlclose(h_rt_);
    input_ndarray_ = nullptr;
    get_output_ = run_ = set_input_ = nullptr;
    mod_ = nullptr;
    h_rt_ = nullptr;
    initialized_ = false;
}
