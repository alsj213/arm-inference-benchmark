#include "precision.h"

#include <cstdio>
#include <cstring>
#include <cstdint>

namespace precision {

// ── GGUF 头解析（自包含，不依赖 llama.cpp API） ──
static bool read_u32(FILE* f, uint32_t* v) {
    return fread(v, sizeof(uint32_t), 1, f) == 1;
}
static bool read_u64(FILE* f, uint64_t* v) {
    return fread(v, sizeof(uint64_t), 1, f) == 1;
}
static bool skip_bytes(FILE* f, uint64_t n) {
    return fseek(f, static_cast<long>(n), SEEK_CUR) == 0;
}
static bool skip_string(FILE* f) {
    uint64_t len = 0;
    if (!read_u64(f, &len)) return false;
    return skip_bytes(f, len);
}

static bool skip_value(FILE* f, uint32_t type);
static bool skip_array(FILE* f, uint32_t elem_type, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i)
        if (!skip_value(f, elem_type)) return false;
    return true;
}

// GGUFValueType: 0..12（与 gguf.h 一致）
static bool skip_value(FILE* f, uint32_t type) {
    switch (type) {
        case 0:  // uint8
        case 1:  // int8
            return skip_bytes(f, 1);
        case 2:  // uint16
        case 3:  // int16
            return skip_bytes(f, 2);
        case 4:  // uint32
        case 5:  // int32
        case 6:  // float32
            return skip_bytes(f, 4);
        case 7:  // bool
            return skip_bytes(f, 1);
        case 8:  // string
            return skip_string(f);
        case 9: {  // array
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!read_u32(f, &elem_type) || !read_u64(f, &count)) return false;
            return skip_array(f, elem_type, count);
        }
        case 10:  // uint64
        case 11:  // int64
        case 12:  // float64
            return skip_bytes(f, 8);
        default:
            return false;
    }
}

bool read_gguf_file_type(const std::string& path, int* ftype) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) {
        fclose(f);
        return false;
    }

    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_u32(f, &version) ||
        !read_u64(f, &tensor_count) ||
        !read_u64(f, &kv_count)) {
        fclose(f);
        return false;
    }

    bool found = false;
    for (uint64_t i = 0; i < kv_count; ++i) {
        uint64_t key_len = 0;
        if (!read_u64(f, &key_len)) break;
        std::string key(key_len, '\0');
        if (key_len > 0 && fread(&key[0], 1, key_len, f) != key_len) break;

        uint32_t vtype = 0;
        if (!read_u32(f, &vtype)) break;

        if (key == "general.file_type" && vtype == 4) {  // uint32
            uint32_t ft = 0;
            if (read_u32(f, &ft)) {
                *ftype = static_cast<int>(ft);
                found = true;
            }
            break;
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f);
    return found;
}

// ── GGUF file_type → 规范 Info（与 llama.cpp LLAMA_FTYPE_* 一致） ──
Info from_gguf_ftype(int ftype) {
    switch (ftype) {
        case 0:  return {"f32", "F32"};
        case 1:  return {"f16", "F16"};
        case 2:  return {"q4", "Q4_0"};
        case 3:  return {"q4", "Q4_1"};
        case 7:  return {"q8", "Q8_0"};
        case 8:  return {"q5", "Q5_0"};
        case 9:  return {"q5", "Q5_1"};
        case 10: return {"q2", "Q2_K"};
        case 11: return {"q3", "Q3_K_S"};
        case 12: return {"q3", "Q3_K_M"};
        case 13: return {"q3", "Q3_K_L"};
        case 14: return {"q4", "Q4_K_S"};
        case 15: return {"q4", "Q4_K_M"};
        case 16: return {"q5", "Q5_K_S"};
        case 17: return {"q5", "Q5_K_M"};
        case 18: return {"q6", "Q6_K"};
        case 19: return {"iq", "IQ2_XXS"};
        case 20: return {"iq", "IQ2_XS"};
        case 21: return {"q2", "Q2_K_S"};
        case 22: return {"iq", "IQ3_XS"};
        case 23: return {"iq", "IQ3_XXS"};
        case 24: return {"iq", "IQ1_S"};
        case 25: return {"iq", "IQ4_NL"};
        case 26: return {"iq", "IQ3_S"};
        case 27: return {"iq", "IQ3_M"};
        case 28: return {"iq", "IQ2_S"};
        case 29: return {"iq", "IQ2_M"};
        case 30: return {"iq", "IQ4_XS"};
        case 31: return {"iq", "IQ1_M"};
        case 32: return {"f16", "BF16"};
        case 36: return {"tq", "TQ1_0"};
        case 37: return {"tq", "TQ2_0"};
        case 38: return {"q4", "MXFP4_MOE"};
        case 39: return {"q4", "NVFP4"};
        case 40: return {"q1", "Q1_0"};
        case 41: return {"q2", "Q2_0"};
        default: {
            char buf[32];
            snprintf(buf, sizeof(buf), "UNKNOWN(%d)", ftype);
            return {"other", buf};
        }
    }
}

// ── MNN llmexport quant_bit → 规范 Info ──
// quant_bit=0 表示未量化；MNN LLM 导出默认权重为 fp16。
Info from_mnn_quant_bit(int quant_bit) {
    switch (quant_bit) {
        case 0:  return {"f16", "FP16"};
        case 2:  return {"q2", "int2"};
        case 3:  return {"q3", "int3"};
        case 4:  return {"q4", "int4"};
        case 8:  return {"q8", "int8"};
        default: {
            char buf[32];
            snprintf(buf, sizeof(buf), "int%d", quant_bit);
            return {"other", buf};
        }
    }
}

}  // namespace precision
