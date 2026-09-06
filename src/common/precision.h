#ifndef BENCHMARK_COMMON_PRECISION_H_
#define BENCHMARK_COMMON_PRECISION_H_

#include <string>

/*!
 * \brief 模型量化精度规范化工具 — 跨框架精度对齐。
 *
 * 目的：跨框架对比时必须使用同一量化级别（如 MNN int8 vs llama.cpp Q4_K_M
 * 属于不公平对比）。本模块把各框架的精度描述（GGUF file_type / MNN quant_bit）
 * 归一化为规范级别（f32 / f16 / q8 / q4 / q3 / q2 ...），供 llm_benchmark 的
 * --require-precision 做一致性校验。
 */
namespace precision {

struct Info {
    std::string level;   // 规范级别: f32 | f16 | q8 | q4 | q3 | q2 | q5 | q6 | iq | other
    std::string label;   // 人类可读: "F16" | "Q4_K_M" | "int8" | ...
};

/*!
 * \brief GGUF general.file_type 整数 → 规范 Info。
 * GGUF file_type 与 llama.cpp LLAMA_FTYPE_* 枚举值一致（llama.h），
 * 由 llama.cpp / MobileLLM 共用。
 */
Info from_gguf_ftype(int ftype);

/*!
 * \brief MNN llmexport config.json 的 quant_bit → 规范 Info。
 * quant_bit=0 表示未量化（MNN LLM 导出默认为 fp16 权重）。
 */
Info from_mnn_quant_bit(int quant_bit);

/*!
 * \brief 最小 GGUF 头解析器：读取 general.file_type（uint32）。
 * 自包含、不依赖 llama.cpp/MobileLLM API，供所有 GGUF 后端做精度检测。
 * \return true 且 *ftype 有效时成功。
 */
bool read_gguf_file_type(const std::string& path, int* ftype);

}  // namespace precision

#endif  // BENCHMARK_COMMON_PRECISION_H_
