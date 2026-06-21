# VL 多模态推理集成 + 仓库整理 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Qwen3-VL-4B/Qwen2-VL-2B 多模态推理集成到统一的 llm_benchmark 工具中，同时将 src/ 目录整理为 cnn/llm/single_op 三条主线

**Architecture:** llm_benchmark 作为统一入口，通过 `--image` 标志进入 VL 模式，MNN 端直接调用 MNN::Llm MultimodalPrompt API，llama.cpp 端子进程调用 llama-mtmd-cli。src/ 按推理类型分三个子目录各自维护 CMakeLists。

**Tech Stack:** C++17, MNN LLM (MultimodalPrompt API), llama.cpp (mtmd-cli 子进程), Android NDK cross-compile

## Global Constraints

- 所有后端源代码完整保留，不删不改停用后端
- Android arm64-v8a, API 29
- 精度对比阈值：余弦相似度 > 0.95 通过
- 编译宏隔离：BENCHMARK_MNN / BENCHMARK_LLAMACPP

---

### Task 1: MNN LLM Backend VL API 扩展

**Files:**
- Modify: `src/backends/mnn_llm_backend.h`
- Modify: `src/backends/mnn_llm_backend.cpp`

**Interfaces:**
- Consumes: MNN::Transformer::Llm (`#include "llm/llm.hpp"`), MNN::Express::VARP, MultimodalPrompt
- Produces: `MnnLlmBackend::generate_vl(image_data, width, height, prompt, max_tokens) -> std::string`
- Produces: `MnnLlmBackend::benchmark_vl(n_prompt, n_gen, n_repeat) -> LlmBenchResult`

- [ ] **Step 1: 在 mnn_llm_backend.h 中添加 VL 接口声明**

在 `MnnLlmBackend` 类中添加：

```cpp
// VL (Vision-Language) API
struct MultimodalInput {
    std::vector<uint8_t> image_data;  // RAW RGB pixels
    int width = 0;
    int height = 0;
    std::string prompt;
    int max_tokens = 128;
};
std::string generate_vl(const MultimodalInput& input);
LlmBenchResult benchmark_vl(int n_prompt, int n_gen, int n_repeat);

// Logits access for accuracy comparison
std::vector<float> get_last_logits() const;
```

注意：`LlmBenchResult` 已经存在，但需要确认它是否包含 `vision_time_s` 字段。如果不需要增加可选字段，保持现有结构。

- [ ] **Step 2: 在 mnn_llm_backend.cpp 中实现 VL API**

```cpp
std::string MnnLlmBackend::generate_vl(const MultimodalInput& input) {
    if (!model_loaded_ || !llm_) return "";

    // Create VARP from raw RGB data
    auto varp = _Const(input.image_data.data(),
        {input.height, input.width, 3}, NCHW, halide_type_of<uint8_t>());
    varp = _Cast<float>(varp) * _Const(1.0f / 255.0f);

    // Build MultimodalPrompt (reuse same pattern as mnn_vl_test.cpp)
    MultimodalPrompt mm_prompt;
    mm_prompt.prompt_template = "<|im_start|>user\n<img>img1</img>" +
        input.prompt + "<|im_end|>\n<|im_start|>assistant\n";
    // Add vision start/end tokens if needed by model
    mm_prompt.images["img1"] = {varp, input.width, input.height};

    // Run inference
    std::ostringstream oss;
    llm_->response(mm_prompt, &oss, nullptr, input.max_tokens);
    return oss.str();
}

MnnLlmBackend::LlmBenchResult MnnLlmBackend::benchmark_vl(
        int n_prompt, int n_gen, int n_repeat) {
    LlmBenchResult result = {};
    if (!model_loaded_) return result;

    // Create a dummy image (420x420 gradient like test pattern)
    int w = 420, h = 420;
    std::vector<uint8_t> dummy_img(w * h * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            dummy_img[(y*w+x)*3+0] = (uint8_t)(255 * x / w);
            dummy_img[(y*w+x)*3+1] = (uint8_t)(255 * y / h);
            dummy_img[(y*w+x)*3+2] = 128;
        }

    MultimodalInput input;
    input.image_data = dummy_img;
    input.width = w;
    input.height = h;
    input.prompt = "describe the image";
    input.max_tokens = n_gen;

    std::vector<double> prefill_speeds, decode_speeds;

    for (int r = 0; r < n_repeat + 1; r++) {
        llm_->reset();

        auto* ctx = llm_->getContext();
        int64_t vision_before = ctx->vision_us;
        int64_t prefill_before = ctx->prefill_us;
        int64_t decode_before = ctx->decode_us;

        generate_vl(input);

        int64_t vision_delta = ctx->vision_us - vision_before;
        int64_t prefill_delta = ctx->prefill_us - prefill_before;
        int64_t decode_delta = ctx->decode_us - decode_before;

        if (r == 0) continue;  // warmup

        double vision_s = vision_delta / 1e6;
        double prefill_s = prefill_delta / 1e6;
        double decode_s = decode_delta / 1e6;

        // Estimate token counts from timing ratios
        // n_prompt = text_prompt_tokens + visual_tokens
        // For simplicity, use the ctx's internal prompt_len and gen_seq_len
        prefill_speeds.push_back((prefill_s > 0) ? (double)n_prompt / prefill_s : 0);
        decode_speeds.push_back((decode_s > 0) ? (double)n_gen / decode_s : 0);
    }

    double avg_prefill = 0, avg_decode = 0;
    for (auto v : prefill_speeds) avg_prefill += v;
    for (auto v : decode_speeds) avg_decode += v;
    result.prefill_tok_per_s = avg_prefill / prefill_speeds.size();
    result.decode_tok_per_s = avg_decode / decode_speeds.size();

    return result;
}
```

- [ ] **Step 3: 编译验证**

```bash
# 在本地先做语法检查
cd build_android && cmake --build . --target llm_benchmark 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add src/backends/mnn_llm_backend.h src/backends/mnn_llm_backend.cpp
git commit -m "feat: add VL multimodal API to MNN LLM backend"
```

---

### Task 2: llama.cpp Backend VL API 扩展（子进程方式）

**Files:**
- Modify: `src/backends/llamacpp_backend.h`
- Modify: `src/backends/llamacpp_backend.cpp`

**Interfaces:**
- Consumes: subprocess execution of `llama-mtmd-cli`, stdout parsing
- Produces: `LlamaCppBackend::generate_vl(image_path, prompt, max_tokens) -> std::string`
- Produces: `LlamaCppBackend::benchmark_vl(n_prompt, n_gen, n_repeat) -> TokenBenchResult`

- [ ] **Step 1: 在 llamacpp_backend.h 中添加 VL 接口**

```cpp
// VL (Vision-Language) API via mtmd-cli subprocess
struct VLBatchResult {
    std::string text;
    double vision_time_s = 0;    // image slice encode
    double prefill_time_s = 0;   // prompt eval
    double decode_time_s = 0;    // token generation
    int total_tokens = 0;
};

// Requires: models/qwen3-vl-4b-instruct-q4_k_m.gguf + mmproj on device
// Uses: llama-mtmd-cli as subprocess
VLBatchResult generate_vl(const std::string& image_path,
                          const std::string& prompt,
                          int max_tokens = 128,
                          const std::string& mmproj_path = "");
```

- [ ] **Step 2: 在 llamacpp_backend.cpp 中实现子进程调用**

```cpp
#include <cstdio>
#include <memory>
#include <array>

// Execute shell command and capture stdout
static std::string exec_cmd(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(
        popen(cmd.c_str(), "r"), pclose);
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
        result += buffer.data();
    return result;
}

LlamaCppBackend::VLBatchResult LlamaCppBackend::generate_vl(
        const std::string& image_path,
        const std::string& prompt,
        int max_tokens,
        const std::string& mmproj_path) {

    VLBatchResult r;

    // Build model paths (assumes standard device layout)
    std::string model_dir = "models/qwen3-vl-4b";
    std::string mmproj = mmproj_path.empty()
        ? model_dir + "/Qwen3-VL-4B-Instruct-f16.mmproj"
        : mmproj_path;
    std::string gguf = model_dir + "/Qwen3-VL-4B-Instruct-q4_k_m.gguf";
    std::string mtmd = "./llama-mtmd-cli";

    // Build command
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "cd /data/local/tmp/benchmark && "
        "LD_LIBRARY_PATH=. %s "
        "-m %s "
        "--mmproj %s "
        "--image %s "
        "-p '%s' "
        "-n %d "
        "-t 4 "
        "--no-warmup "
        "--perf 2>&1",
        mtmd.c_str(), gguf.c_str(), mmproj.c_str(),
        image_path.c_str(), prompt.c_str(), max_tokens);

    std::string output = exec_cmd(cmd);

    // Parse timing from perf output
    // Format: llama_perf_context_print: prompt eval time = X ms / Y tokens
    // Format: llama_perf_context_print: eval time = X ms / Y runs
    // Format: image slice encoded in X ms
    // Format: image decoded in X ms

    auto parse_ms = [](const std::string& text, const std::string& key) -> double {
        auto pos = text.find(key);
        if (pos == std::string::npos) return 0;
        pos += key.size();
        while (pos < text.size() && text[pos] == ' ') pos++;
        // Read number until space or 'm'
        std::string num;
        while (pos < text.size() && (isdigit(text[pos]) || text[pos] == '.'))
            num += text[pos++];
        return num.empty() ? 0 : std::stod(num);
    };

    r.vision_time_s = parse_ms(output, "image slice encoded in") / 1000.0
                    + parse_ms(output, "image decoded in") / 1000.0;
    r.prefill_time_s = parse_ms(output, "prompt eval time =") / 1000.0;
    r.decode_time_s = parse_ms(output, "eval time =") / 1000.0;

    // Extract generated text (between prompt and "llama_perf_context_print")
    auto perf_pos = output.find("llama_perf_context_print");
    if (perf_pos != std::string::npos) {
        // Find the last newline before perf output
        auto text_start = output.rfind('\n', perf_pos);
        if (text_start != std::string::npos)
            text_start = output.rfind('\n', text_start - 1);
        if (text_start == std::string::npos) text_start = 0;
        r.text = output.substr(text_start, perf_pos - text_start);
        // Trim whitespace
        while (!r.text.empty() && (r.text.back() == '\n' || r.text.back() == ' '))
            r.text.pop_back();
    }

    // Parse total tokens
    // " / Y tokens" in prompt eval line
    {
        auto pos = output.find("prompt eval time =");
        if (pos != std::string::npos) {
            auto slash = output.find('/', pos);
            if (slash != std::string::npos) {
                auto tok_pos = slash + 1;
                while (tok_pos < output.size() && output[tok_pos] == ' ') tok_pos++;
                std::string num;
                while (tok_pos < output.size() && isdigit(output[tok_pos]))
                    num += output[tok_pos++];
                r.total_tokens = num.empty() ? 0 : std::stoi(num);
            }
        }
    }

    return r;
}
```

注意：这个函数在 Android 设备上运行时通过 shell 执行 `llama-mtmd-cli` 命令。在 host 侧编译时只是一个封装，不需要链接 llama.cpp。

- [ ] **Step 3: Commit**

```bash
git add src/backends/llamacpp_backend.h src/backends/llamacpp_backend.cpp
git commit -m "feat: add VL multimodal API to llama.cpp backend (subprocess)"
```

---

### Task 3: llm_benchmark 统一入口改造

**Files:**
- Modify: `src/llm_benchmark.cpp`

**Interfaces:**
- Consumes: `MnnLlmBackend::generate_vl()`, `MnnLlmBackend::benchmark_vl()`, `LlamaCppBackend::generate_vl()`
- Produces: unified CLI with `--image` flag, VL mode dispatch

- [ ] **Step 1: 扩展 Args 结构体**

```cpp
struct Args {
    std::string backend = "llamacpp";
    std::string model;
    int max_tokens = 128;
    int n_ctx = 1024;
    int n_prompt = 128;
    int n_repeat = 5;
    bool benchmark_only = false;
    // --- 新增 ---
    std::string image_path;     // 启用 VL 模式
    int image_width = 0;        // RAW 图片宽度（可选）
    int image_height = 0;       // RAW 图片高度（可选）
    std::string accuracy_ref;   // 精度对比基准后端 ("mnn" / "llamacpp")
    int seed = 42;
    std::string prompt_text;    // 自定义 prompt（可选）
};
```

- [ ] **Step 2: 扩展 parse_args 支持新参数**

在 `parse_args` 中添加：

```cpp
} else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
    args.image_path = argv[++i];
} else if (strcmp(argv[i], "--image-size") == 0 && i + 2 < argc) {
    args.image_width = atoi(argv[++i]);
    args.image_height = atoi(argv[++i]);
} else if (strcmp(argv[i], "--accuracy") == 0 && i + 1 < argc) {
    args.accuracy_ref = argv[++i];
} else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
    args.seed = atoi(argv[++i]);
} else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
    args.prompt_text = argv[++i];
}
```

- [ ] **Step 3: 更新 print_usage**

```cpp
printf("  --image <path>                  Image file for VL inference\n");
printf("  --image-size <w> <h>           RAW image dimensions\n");
printf("  --accuracy <mnn|llamacpp>       Accuracy verification mode\n");
printf("  --seed <n>                      Random seed (default: 42)\n");
printf("  --prompt <text>                 Custom prompt text\n");
```

- [ ] **Step 4: 添加 VL 模式运行函数**

```cpp
#ifdef BENCHMARK_MNN
static bool run_mnn_llm_vl(const Args& args) {
    printf("=== MNN LLM VL Benchmark ===\n\n");

    std::string config_path = args.model.empty()
        ? "models/qwen3-vl-4b-mnn/config.json" : args.model;

    MnnLlmBackend backend;
    BenchmarkConfig config;
    config.model_path = config_path;

    printf("Loading MNN VL model...\n");
    if (!backend.load_model(config_path)) {
        printf("ERROR: load failed\n"); return false;
    }

    // Read image file
    FILE* fp = fopen(args.image_path.c_str(), "rb");
    if (!fp) { printf("ERROR: cannot open image\n"); return false; }
    fseek(fp, 0, SEEK_END);
    size_t fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> img_data(fsize);
    fread(img_data.data(), 1, fsize, fp);
    fclose(fp);

    // Determine dimensions (420 default for Qwen3-VL)
    int w = args.image_width ? args.image_width : 420;
    int h = args.image_height ? args.image_height : 420;
    std::string prompt = args.prompt_text.empty()
        ? "请用中文详细描述这张图片" : args.prompt_text;

    MnnLlmBackend::MultimodalInput input;
    input.image_data = std::move(img_data);
    input.width = w; input.height = h;
    input.prompt = prompt;
    input.max_tokens = args.max_tokens;

    if (args.benchmark_only) {
        printf("--- VL Benchmark ---\n");
        auto r = backend.benchmark_vl(args.n_prompt, args.max_tokens, args.n_repeat);
        printf("prefill: %.2f tok/s  |  decode: %.2f tok/s\n",
               r.prefill_tok_per_s, r.decode_tok_per_s);
    } else {
        printf("--- VL Generate ---\n");
        auto t0 = std::chrono::high_resolution_clock::now();
        std::string output = backend.generate_vl(input);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t1 - t0).count();
        printf("\n%s\n\n", output.c_str());
        printf("Total time: %.2f s\n", total_s);
    }

    backend.deinit();
    return true;
}
#else
static bool run_mnn_llm_vl(const Args&) { return false; }
#endif
```

同理添加 `run_llamacpp_vl()` 函数，调用 `LlamaCppBackend::generate_vl()`。

- [ ] **Step 5: 更新 main() 调度逻辑**

```cpp
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    if (!args.image_path.empty()) {
        // VL mode
        if (args.backend == "mnn_llm" || args.backend == "mnn") {
            return run_mnn_llm_vl(args) ? 0 : 1;
        } else {
            return run_llamacpp_vl(args) ? 0 : 1;
        }
    }

    // Original text-only path (unchanged)
    if (args.backend == "mnn_llm" || args.backend == "mnn") {
        return run_mnn_llm(args) ? 0 : 1;
    } else {
        return run_llamacpp(args) ? 0 : 1;
    }
}
```

- [ ] **Step 6: Commit**

```bash
git add src/llm_benchmark.cpp
git commit -m "feat: add VL mode to llm_benchmark with --image flag"
```

---

### Task 4: src/ 目录结构整理（三条主线）

**Files:**
- Create: `src/cnn/CMakeLists.txt`
- Create: `src/cnn/main.cpp` (move from `src/main.cpp`)
- Create: `src/llm/CMakeLists.txt`
- Create: `src/llm/llm_benchmark.cpp` (move from `src/llm_benchmark.cpp`)
- Create: `src/single_op/CMakeLists.txt`
- Create: `src/single_op/single_op_benchmark.cpp` (move from `src/single_op_benchmark.cpp`)
- Modify: `src/CMakeLists.txt` → delegating add_subdirectory
- Modify: root `CMakeLists.txt` → add_subdirectory for new paths
- Keep: `src/backends/`, `src/common/`, `src/models/` 不变

- [ ] **Step 1: 分析当前 src/CMakeLists.txt 结构**

```bash
cat src/CMakeLists.txt | head -80
```

确认每个 target 对应的源文件。

- [ ] **Step 2: 创建 src/cnn/ 子目录**

```bash
mkdir -p src/cnn
```

```cmake
# src/cnn/CMakeLists.txt
# CNN inference benchmark targets (ORT, MNN, TVM, NCNN)
add_executable(benchmark_inference main.cpp)
target_link_libraries(benchmark_inference PRIVATE benchmark_common)
```

- [ ] **Step 3: 创建 src/llm/ 子目录**

```bash
mkdir -p src/llm
```

```cmake
# src/llm/CMakeLists.txt
# LLM + VL benchmark target (llama.cpp, MNN LLM)
add_executable(llm_benchmark llm_benchmark.cpp)
target_link_libraries(llm_benchmark PRIVATE benchmark_common)
```

- [ ] **Step 4: 创建 src/single_op/ 子目录**

```bash
mkdir -p src/single_op
```

```cmake
# src/single_op/CMakeLists.txt
# Single operator benchmark target
add_executable(single_op_benchmark single_op_benchmark.cpp)
target_link_libraries(single_op_benchmark PRIVATE benchmark_common)
```

- [ ] **Step 5: 移动源文件**

```bash
git mv src/main.cpp src/cnn/main.cpp
git mv src/llm_benchmark.cpp src/llm/llm_benchmark.cpp
git mv src/single_op_benchmark.cpp src/single_op/single_op_benchmark.cpp
```

- [ ] **Step 6: 简化 src/CMakeLists.txt 为分发模式**

```cmake
# src/CMakeLists.txt — 三条主线分发
add_subdirectory(cnn)
add_subdirectory(llm)
add_subdirectory(single_op)
```

保留 `add_subdirectory(backends)`, `add_subdirectory(common)`, `add_subdirectory(models)` 等公共模块。

- [ ] **Step 7: 编译验证**

```bash
cd build_android && cmake .. && make -j$(nproc) 2>&1 | tail -20
```

- [ ] **Step 8: Commit**

```bash
git add src/
git commit -m "refactor: split src/ into cnn/llm/single_op three main lines"
```

---

### Task 5: 设备测试 + 精度验证

**Files:**
- None (testing only)

- [ ] **Step 1: 编译并推送**

```bash
cd build_android && cmake --build . --target llm_benchmark -j$(nproc)
ADB push build_android/src/llm/llm_benchmark /data/local/tmp/benchmark/
```

- [ ] **Step 2: text-only 回归测试**

```bash
ADB shell "cd /data/local/tmp/benchmark && \
  LD_LIBRARY_PATH=. ./llm_benchmark --backend llamacpp \
  --benchmark --n-prompt 64 --max-tokens 32 --n-repeat 1"
# 预期输出: prefill: ~5 tok/s  |  decode: ~4 tok/s （与之前一致）
```

- [ ] **Step 3: MNN VL 测试**

```bash
ADB shell "cd /data/local/tmp/benchmark && \
  LD_LIBRARY_PATH=. ./llm_benchmark --backend mnn_llm \
  --image test_uniform.raw --image-size 420 420 \
  --prompt '描述这张图片' --max-tokens 32"
```

- [ ] **Step 4: llama.cpp VL 测试**

```bash
ADB shell "cd /data/local/tmp/benchmark && \
  LD_LIBRARY_PATH=. ./llm_benchmark --backend llamacpp \
  --image test_uniform.bmp \
  --prompt '描述这张图片' --max-tokens 32"
```

---

### Task 6: mobile-bench 插件更新（单独提交）

**Files:**
- Update: `/home/liu/project/mobile-bench/` 多个文件（与上次同步一致）
- Update: 本仓库 CLAUDE.md 中的框架状态表

- [ ] **Step 1: 同步本仓库 CLAUDE.md 的框架状态表**

将当前仓库的 `CLAUDE.md` 中框架状态表更新为最新状态（MNN/ORT/NCNN/TVM/MindSpore Lite 活跃，llama.cpp 开发中）

- [ ] **Step 2: 更新 mobile-bench 插件**

```bash
cd /home/liu/project/mobile-bench
git add -A && git commit -m "chore: sync with benchmark repo VL integration"
```

- [ ] **Step 3: 提交本仓库**

```bash
cd /home/liu/project/newwork/benchmark
git add -A
git commit -m "feat: VL model inference integrated into llm_benchmark

- MNN LLM backend: generate_vl() with MultimodalPrompt API
- llama.cpp backend: generate_vl() via mtmd-cli subprocess
- llm_benchmark: unified --image flag for VL mode
- src/: reorganized into cnn/llm/single_op three main lines"
```
