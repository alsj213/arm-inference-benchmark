# 📱 手机端大模型部署全景调研报告

> 调研时间：2026 年 6 月
> 覆盖范围：国内外厂商、大语言模型（LLM）、多模态模型（VLM）、视觉模型
> 来源：25+ 一手/二手源，多源交叉验证

---

## 一、核心结论

当前手机端大模型部署呈 **三足鼎立** 格局：

| 方向 | 代表进展 | 成熟度 |
|------|----------|:------:|
| **模型自身小型化** | Google Gemini Nano v3 → 940 tok/s | ✅ 已量产 |
| **芯片-模型联合优化** | 天玑 + Qwen3 / 骁龙 + Llama 3.2 | ✅ 已落地 |
| **压缩技术突破** | AWQ 量化 / W4A4 / 知识蒸馏 | ⚠️ 学术成熟，工程待完善 |

**关键判断：**

- **7B 参数 + 4-bit 量化 = 当前旗舰手机实际能力上限**（MobileAIBench 在 iPhone 14 实测确认）
- **13B+ 模型** 量化后需 ~10GB 峰值内存，超出手机能力
- **软件生态是当前瓶颈** — 推理框架尚未有效对接 NPU（Hexagon NPU 在主流框架中完全未被利用）
- **QLoRA + 联发科 NPU 的联合优化趋势明显**，芯片-模型绑定成为主流路径

---

## 二、海外厂商布局

### 2.1 Google — Gemini Nano v3（最成熟的端侧方案）

**状态：** ✅ 已随 Pixel 10 系列首发（2025 年 8 月）

| 指标 | 数据 |
|------|:----:|
| 模型规模 | Nano（未公开精确参数，估计 <3B） |
| 文本 Prefix Speed（Pixel 10 Pro） | **940 tok/s**（v2 为 510 tok/s，↑84%） |
| 多模态 | ✅ text-to-text + image-to-text |
| 上下文窗口 | 128K |

**开放路径：**

1. **ML Kit GenAI APIs** — 预构建用例（summarization, proofreading, rewriting, image description）
2. **Prompt API（Alpha）** — 允许开发者自定义提示词
3. 通过 Google AI Edge 面向开发者开放

> 来源：Android Developers Blog、Google Developers Blog（2025.08 / 2025.10）

### 2.2 Meta — Llama 系列端侧部署

| 模型 | 参数量 | 部署平台 | 状态 |
|------|:------:|---------|:----:|
| **Llama 3.2 1B** | 1B | Snapdragon（高通合作） | ✅ 已上线 Qualcomm AI Hub |
| **Llama 3.2 3B** | 3B | Snapdragon | ✅ 已上线 Qualcomm AI Hub |

- 与高通深度合作，通过 **Qualcomm AI Hub** 分发模型
- 结合 **ExecuTorch**（Meta 自研端侧推理引擎）部署
- ExecuTorch 已用于 Instagram / WhatsApp 等生产环境

> 来源：Qualcomm 官方新闻稿（2024.09）、Meta 官方文档

### 2.3 Apple — Apple Intelligence

Apple 的端侧 AI 策略偏向封闭生态，公开信息有限：

- **Apple Intelligence**（iOS 18+）内置端侧 AI 功能
- **OpenELM**（270M~1.1B）开源系列，暗示 Apple 在探索轻量模型
- 自研 **ANE（Apple Neural Engine）** + **Core ML** 推理栈
- 报告缺乏 Apple 端侧 LLM 的一手性能数据

> ⚠️ Apple 是目前透明度最低的大厂，端侧模型规格和性能数据缺失

---

## 三、芯片厂商 × 模型厂商联合优化

### 3.1 MediaTek × 阿里通义千问

**状态：** ✅ 2025 年 4 月宣布，已部署

| 项目 | 详情 |
|------|------|
| 模型 | Qwen3-0.6B / 1.7B / 4B |
| 芯片 | Dimensity 9400（NPU 890） |
| 工具链 | Dimensity GenAI Toolkit 2.0 |

NPU 890 针对端侧推理的硬件特性：
- 混合专家模型（MoE）加速
- 多头潜在注意力（MLA）加速
- 多 Token 预测（MTP）加速

**实测性能（Vivo X200 Pro / 天玑 9400）：**
- Qwen3-1.7B q4f32 → **37 tok/s**

### 3.2 MediaTek × Microsoft

**状态：** ✅ 2025 年 5 月 28 日官宣

| 项目 | 详情 |
|------|------|
| 模型 | Phi-4-mini（3.8B）+ Phi-4-mini-reasoning |
| 芯片 | Dimensity 9400/9400+ NPU |
| 工具链 | Dimensity GenAI Toolkit 2.0 |

### 3.3 Qualcomm × Meta

**状态：** ✅ 2024 年 9 月宣布，已可用

| 项目 | 详情 |
|------|------|
| 模型 | Llama 3.2 1B / 3B |
| 芯片 | Snapdragon 8 Gen 3 / 8 Elite |
| 分发 | Qualcomm AI Hub |

### 3.4 Qualcomm × 多家

**Snapdragon 8 Gen 3** 已支持超过 20 个模型在手机端运行，涵盖：
- 大语言模型（LLM）：Llama 2/3、Gemma、Phi-3
- 语言视觉模型（LVM）：Qwen-VL、LLaVA
- 自动语音识别（ASR）：Whisper
- 图像生成：Stable Diffusion

---

## 四、通义千问（Qwen）系列 — 国内端侧龙头

阿里在端侧部署上布局最完整，形成 **模型 → 芯片合作 → 推理引擎** 全闭环。

### 4.1 Qwen3 系列（主力端侧阵容）

| 模型 | 参数量 | 手机端状态 | 特点 |
|------|:------:|:----------:|------|
| **Qwen3-0.6B** | 600M | ✅ 量产可用 | 最小尺寸，量化后 ~300MB |
| **Qwen3-1.7B** | 1.7B | ✅ 量产可用 | 天玑 9400 优化，实测 **37 tok/s** |
| **Qwen3-4B-Instruct** | 4B | ✅ 已开源 | 256K 上下文，量化后 ~4GB |
| **Qwen3-4B-Thinking** | 4B | ✅ 已开源 | 推理能力媲美 30B MoE |
| **Qwen3-VL 4B/8B** | 4B/8B | ✅ 视觉语言 | 骁龙 8 Gen3 / A17 上 **10+ fps** |

**关键时间线：**
- 2025.04 → 天玑 9400 完成 Qwen3 端侧部署
- 2025.08 → Qwen3-4B 系列开源，性能**超越 GPT-4.1-nano**
- 2025.10 → Qwen3-VL 发布
- 2026 Q1 → Qwen3-VL 集成消费级产品

### 4.2 Qwen2.5 系列（成熟稳定版）

| 模型 | 参数量 | 部署方案 | 实测性能 |
|------|:------:|---------|:--------:|
| Qwen2.5-0.5B | 500M | ExecuTorch / ONNX | Pixel 8 ~**40 tok/s** |
| Qwen2.5-1.5B | 1.5B | LiteRT / ONNX | S25 Ultra 解码 **34 tok/s** |
| Qwen2.5-3B | 3B | ONNX q4f32 | Kirin 990 ~**20 tok/s** |
| Qwen2.5-Coder 1.5B | 1.5B | GGUF / MNN | ~900MB，手机侧代码生成 |

### 4.3 Qwen2.5-Omni（全模态端侧旗舰）

| 指标 | 数据 |
|------|:----:|
| 参数量 | 7B |
| 架构 | Thinker-Talker 双核 |
| 模态 | 文本 + 图像 + 音频 + 视频 **端到端** |
| 地位 | 登顶 Hugging Face 全球开源总榜 |
| 端侧 | 官方宣称 "手机轻松部署"，实际体验依赖量化 |

### 4.4 MNN — 阿里端侧推理引擎

MNN 是本 benchmark 项目的主力后端之一，与 Qwen 深度集成：

```
Qwen 模型 → MNNConvert 转换 → INT4/INT8/AWQ 量化 → MNN Chat APP 加载推理
```

- 提供全链路工具链（转换 → 量化 → 测试）
- 支持 INT4/INT8/AWQ 多种精度优化
- MNN Chat App 可直接下载 Qwen 模型运行
- **Android + iOS 双平台支持**

---

## 五、第三方性能基准 — 真实设备实测

### 5.1 MobileAIBench（Salesforce AI Research）

**论文：** arXiv:2406.10290
**测试设备：** iPhone 14（真实设备）

**核心结论：**

| 结论 | 详情 |
|------|------|
| 7B = 手机实际天花板 | 即使经过量化，7B 是旗舰手机可管理的上限 |
| 13B 不可用 | Q4_K_M 量化后 ~7.82GB 存储 + ~10.32GB 峰值内存 |
| 评估范围 | LLM 任务 + LMM 任务 + 可信安全维度 |

### 5.2 多框架覆盖全平台实测数据

| 设备 | 芯片 | 模型 | 方案 | 速度（tok/s） |
|:----|:----|:----|:----:|:----:|
| Pixel 10 Pro | Tensor G5 | Gemini Nano v3 | 原生 | **940**（prefix） |
| iPhone 15 Pro | A17 Pro | Qwen3-0.6B | ExecuTorch | ~**40** |
| Pixel 8 | Tensor G3 | Qwen3-0.6B | ExecuTorch | ~**40** |
| Vivo X200 Pro | 天玑 9400 | Qwen3-1.7B | ONNX q4f32 | **37** |
| Samsung S25 Ultra | 骁龙 8 Elite | Qwen2.5-1.5B | LiteRT int8 | **34** |
| OnePlus 13R | 骁龙 8 Gen 3 | LLaVA-1.5 7B | llama.cpp | 较慢（无 NPU） |
| Huawei P40 | Kirin 990 5G | Qwen2.5-1.5B | ONNX q4f32 | **20.5** |
| iPhone 15 Pro | A17 Pro | Qwen2.5-Coder 1.5B | GGUF Metal | ~**15** |

---

## 六、关键瓶颈分析

### 6.1 🟡 软件生态落后于硬件

**EPFL 论文（arXiv 2507.08505, 2025.07）在 OnePlus 13R（Snapdragon 8 Gen 3）测试发现：**

| 框架 | 测试模型 | Hexagon NPU 使用 |
|------|---------|:----------------:|
| llama.cpp | LLaVA-1.5 7B | ❌ 完全未使用 |
| MLC-Imp | MobileVLM-3B | ❌ 完全未使用 |
| mllm | Imp-v1.5 3B | ❌ 完全未使用 |

**现状：** NPU 硬件已就位，但主流推理框架的标准部署流程尚未对接

**缓解进展：** llama.cpp 已于 2025 下半年合入实验性 Hexagon 后端

### 6.2 🟡 模型精度数据缺失

厂商普遍只宣传速度（tok/s）指标，**缺乏标准化的精度对比**：
- Gemini Nano v3 的 MMLU / VQA-Science 等基准数据未公开
- 端侧模型 vs 云端模型的精度差距缺乏量化
- 不同量化方案（INT4/INT8/AWQ）的实际精度损失对比不足

### 6.3 🟡 厂商营销数据水份

- AWQ 论文中的 "mobile GPU" 实际指 **NVIDIA Jetson Orin 64GB**（嵌入式 AI SoM），**不是手机 SoC**
- 联发科 >800 tok/s prefill 为自报数据，缺第三方复现
- 高通 "7B 20 tok/s 业界最快" 的说法已被多源否决

---

## 七、模型压缩技术全景

### 7.1 四大技术路径（Dantas et al., 2025 综述）

| 技术 | 原理 | 压缩比 | 精度损失 |
|------|------|:------:|:--------:|
| **量化（Quantization）** | FP16 → INT4/INT8 | 模型大小 ↓50%+ | 1-3% |
| **剪枝（Pruning）** | 移除不重要连接 | 参数量大幅下降 | 视比例 |
| **知识蒸馏（KD）** | 大模型教小模型 | 可达 40x | 可接受 |
| **NAS（神经架构搜索）** | 搜索高效架构 | 定制化 | 优秀 |

### 7.2 AWQ 量化（MLSys 2024 最佳论文）

- 权重感知量化，关注 **1% 的显著权重通道**
- 声称让 70B Llama-2 可在 "mobile GPU" 运行
- ⚠️ 语境限定：指 Jetson Orin，非手机 SoC
- 但 AWQ 思想已广泛应用于手机端模型量化（MNN 支持）

### 7.3 W4A4 量化

- 权重 + 激活同时 4-bit 量化
- Qwen2.5-1.5B INT4 的 TTFT 低至 **58.4ms**
- 是当前端侧部署的研究热点

---

## 八、国内厂商生态总览

| 厂商 | 模型 | 芯片合作 | 推理引擎 | 端侧进展 |
|:----:|:----:|:--------:|:--------:|:--------:|
| **阿里** | Qwen3 全系列 | 天玑 9400 | **MNN**（自研） | ⭐ 国内最完善 |
| **百度** | 文心一言小模型 | 高通 | Paddle-Lite | ⭐ 有布局 |
| **华为** | 盘古系列 | 麒麟 NPU | MindSpore Lite | ⭐ 芯片优势 |
| **小米** | 自研 AI 大模型 | 高通/联发科 | MNN 等 | ⭐ 合作路线 |
| **OPPO/vivo** | 合作方案 | 天玑深度合作 | 多框架 | ⭐ 紧跟芯片 |
| **字节跳动** | 豆包小模型 | - | 自研 | ⭐ 有布局 |
| **面壁智能** | MiniCPM | 多平台 | llama.cpp/MNN | ⭐ 端侧先行者 |

---

## 九、Qwen 端侧多框架生态

| 框架 | 支持的 Qwen 模型 | 平台 |
|------|:---------------:|:----:|
| **MNN**（阿里） | Qwen3 / Qwen2.5-Coder | Android / iOS |
| **ExecuTorch**（Meta） | Qwen3-0.6B / Qwen2.5 | Android / iOS |
| **LiteRT/TFLite**（Google） | Qwen2.5-1.5B | Android / iOS |
| **ONNX Runtime**（MS） | Qwen2.5 0.5B~3B | Android |
| **GGUF/llama.cpp** | Qwen2.5-Coder | 通用 |
| **Onde Inference**（Flutter） | Qwen2.5-1.5B/Coder | iOS(~15 tok/s) / Android |

---

## 十、未解决问题

1. **精度 vs 速度的权衡：** Gemini Nano v3 和 Qwen3-4B 的标准化基准（MMLU, VQA-Science）数据缺失，无法客观对比端侧模型的推理质量
2. **Apple 端侧策略不透明：** OpenELM 暗示了方向，但 iPhone 端侧真实推理性能无公开数据
3. **三星 Galaxy AI 技术栈：** 是否基于 Gemini Nano 还是自研模型？未知
4. **NPU 接入后的量化收益：** 一旦主流框架接入 Hexagon NPU，延迟和能效的具体改善幅度缺乏数据
5. **国内厂商真实落地进度：** 多为行业综述级别信息，缺一手技术细节和实测数据

---

## 十一、对本 benchmark 项目的启示

1. **Qwen 系列是最合适的 benchmark 候选模型** — 模型规格齐全（0.5B~7B），MNN 原生支持，实测数据可验证
2. **MNN vs ExecuTorch 对比** 将是本项目的差异化价值
3. **NPU 利用率** 应成为 benchmark 的一项关键指标
4. **7B 量化模型的精度退化量** 值得纳入测试维度
5. **天玑 9400 + Qwen3** 的组合是重要的测试场景

---

## 参考资料

### 官方来源
- Google Gemini Nano v3 — [developers.googleblog.com](https://developers.googleblog.com/2025/08/the-latest-gemini-nano-with-on-device-ml-kit-genai-apis.html)
- ML Kit Prompt API — [android-developers.googleblog.com](https://android-developers.googleblog.com/2025/10/ml-kit-genai-prompt-api-alpha-release.html)
- MediaTek × Qwen3 — [developer.mediatek.com](https://developer.mediatek.com/ai/681dc0083648cc23f27eae09.html)
- MediaTek × Phi-4-mini — [mediatek.com](https://www.mediatek.com/tek-talk-blogs/unleash-next-gen-ai-on-mediatek-npus-with-microsofts-phi-4-mini-models)
- Qualcomm Snapdragon 8 Gen 3 — [qualcomm.com](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-3-mobile-platform)
- MNN × Qwen 实战 — [developer.aliyun.com](https://developer.aliyun.com/article/1688650)

### 学术论文
- MobileAIBench（arXiv:2406.10290）— 真实 iPhone 14 端侧 LLM 基准
- EPFL VLM 端侧部署（arXiv:2507.08505）— NPU 利用率分析
- LLM 压缩综述（Springer, 2025）— Dantas et al.
- AWQ 量化（MLSys 2024 Best Paper）

### 社区与工具
- Unsloth 手机部署教程 — [unsloth.ai](https://unsloth.ai/docs/zh/ji-chu/inference-and-deployment/deploy-llms-phone)
- Onde Inference（Flutter）— [pub.dev](https://pub.dev/packages/onde_inference)
- Qualcomm Nexa SDK — [github.com/qualcomm/nexa-sdk](https://github.com/qualcomm/nexa-sdk)
