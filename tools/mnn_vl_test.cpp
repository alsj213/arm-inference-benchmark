// MNN VL 多模态测试程序（RAW 图片版本，跳过 imread）
#include "llm/llm.hpp"
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

using namespace MNN::Transformer;
using namespace MNN::Express;

// 读取 RAW RGB 文件（宽度高度在文件名后两个参数传入）
int main(int argc, const char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " config.json image.raw width height prompt_text" << std::endl;
        std::cout << "  image.raw = 原始 RGB 像素数据 (width*height*3 bytes)" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string image_path = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    std::string user_prompt = argv[5];

    // 创建 LLM
    auto llm = std::unique_ptr<Llm>(Llm::createLLM(config_path));
    if (!llm) { std::cerr << "Failed to create LLM" << std::endl; return 1; }
    llm->set_config(R"({"tmp_path":"tmp"})");

    // 加载模型
    std::cout << "Loading model..." << std::endl;
    if (!llm->load()) { std::cerr << "Model load failed" << std::endl; return 1; }
    std::cout << "Model loaded!" << std::endl;

    // 读取 RAW RGB 文件
    std::cout << "Reading RAW image: " << image_path << " (" << width << "x" << height << ")" << std::endl;
    std::ifstream file(image_path, std::ios::binary);
    if (!file) { std::cerr << "Failed to open image file" << std::endl; return 1; }

    size_t img_size = width * height * 3;
    std::vector<uint8_t> pixels(img_size);
    file.read(reinterpret_cast<char*>(pixels.data()), img_size);
    file.close();
    std::cout << "Read " << pixels.size() << " bytes" << std::endl;

    // 创建 VARP（NHWC 格式, uint8）
    auto varp = _Const(pixels.data(), {height, width, 3}, NCHW, halide_type_of<uint8_t>());
    // 转换到 float [0,1] 范围
    varp = _Cast<float>(varp) * _Const(1.0f / 255.0f);

    // 构建 MultimodalPrompt
    MultimodalPrompt mm_prompt;
    mm_prompt.prompt_template = "<|im_start|>system\nYou are a helpful assistant. Answer directly without thinking.\n<|im_end|>\n<|im_start|>user\n<img>img1</img>" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n";
    mm_prompt.images["img1"] = {varp, width, height};

    // 推理
    std::cout << "Running multimodal inference..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    llm->response(mm_prompt, &std::cout, nullptr, 64);
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t1 - t0).count();

    // 打印时序
    auto ctx = llm->getContext();
    std::cout << "\n\n=== Timing ===" << std::endl;
    std::cout << "vision time: " << ctx->vision_us / 1e6 << " s" << std::endl;
    std::cout << "prefill time: " << ctx->prefill_us / 1e6 << " s" << std::endl;
    std::cout << "decode time: " << ctx->decode_us / 1e6 << " s" << std::endl;
    std::cout << "prefill speed: " << (ctx->prefill_us > 0 ? ctx->prompt_len / (ctx->prefill_us / 1e6) : 0) << " tok/s" << std::endl;
    std::cout << "decode speed: " << (ctx->decode_us > 0 ? ctx->gen_seq_len / (ctx->decode_us / 1e6) : 0) << " tok/s" << std::endl;
    std::cout << "total time: " << total_s << " s" << std::endl;

    return 0;
}
