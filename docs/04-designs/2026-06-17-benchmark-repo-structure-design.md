# Benchmark 仓库结构重整设计方案

**日期**: 2026-06-17
**状态**: 已批准
**目标**: 对 ARM 端侧推理框架性能基准测试项目进行文件级仓库整理，优化目录结构、消除冗余、统一命名规范。

---

## 1. 概述

当前仓库经过半年多的迭代开发，积累了大量模型文件、文档和脚本，存在以下问题：

- **models/** 目录 52G，源文件与导出文件混合，LLM 分散在 `nlp/` 和 `llm/` 两个目录
- **docs/** 文件命名中英文混用、层级混乱，superpowers 设计文档混在子目录中
- **scripts/** 40+ 个文件平铺在根目录，功能类型混杂
- **.gitignore** 有重复规则、未覆盖新模型格式

本次整理为 **文件级操作**（`git mv` / 重命名 / 定期删除），不改 git 历史，不改源码实现逻辑。

---

## 2. models/ — 模型目录重整

### 2.1 当前结构

```
models/
├── classification/       ← 8 个子模型目录，混合了源模型 + 各框架导出产物
│   ├── mobilenetv2/      ← .onnx .pt .mnn .tflite .ncnn.bin .tnnmodel .ms .pnnx.* ...
│   ├── resnet50/         ← 同上，多种格式混杂
│   ├── resnet18/
│   ├── mobilevit_s/
│   ├── shufflenet_v2/
│   ├── squeezenet/
│   └── efficientnet_lite0/
├── detection/
│   └── yolov8n/
├── nlp/                  ← 旧 LLM (gitignore 内，不在跟踪中)
│   ├── bert/
│   ├── qwen2_0.5b/
│   ├── qwen2_1.5b/
│   └── qwen2.5_1.5b/
├── llm/                  ← 新 LLM Qwen3-4B (不在跟踪中)
├── single_ops/           ← 单算子模型
├── single_ops_extracted/ ← 算子提取产物
├── single_ops_gemm/      ← GEMM 算子 (onnx)
├── single_ops_gemm_mnn/  ← GEMM 算子 (mnn) — 与上面重复
├── single_ops_stair/     ← 阶梯算子
└── speech/
```

### 2.2 目标结构

```
models/
├── source/                          # 📦 原始/源模型（canonical 格式）
│   ├── classification/
│   │   ├── mobilenetv2/             # 保留 .onnx .pt (源模型)
│   │   ├── resnet50/                # 保留 .onnx .pt
│   │   ├── resnet18/
│   │   ├── mobilevit_s/
│   │   ├── shufflenet_v2/
│   │   ├── squeezenet/
│   │   └── efficientnet_lite0/
│   ├── detection/
│   │   └── yolov8n/                 # 保留 .onnx
│   └── nlp/
│       ├── bert/                    # 保留 .onnx
│       ├── qwen2_0.5b/hf_model/
│       └── qwen2.5_1.5b/hf_model/
│
├── exported/                        # 🏭 各框架导出产物（可重新生成）
│   ├── mnn/
│   │   ├── mobilenetv2.mnn
│   │   ├── resnet50.mnn
│   │   ├── mobilevit_s.mnn
│   │   └── ...
│   ├── ort/                         # ONNX Runtime 优化版（如有）
│   └── tvm/                         # TVM 编译产物 .so
│
├── single_ops/                      # 🔬 单算子测试模型
│   ├── basic/                       # ← 原 single_ops/ + single_ops_extracted/
│   ├── gemm/                        # ← 原 single_ops_gemm/ + single_ops_gemm_mnn/
│   └── stair/                       # ← 原 single_ops_stair/
│
├── llm/                             # 🧠 LLM 大模型（不在 git 跟踪）
│   ├── qwen2_0.5b/                  # ← 从 models/nlp/ 移入
│   ├── qwen2.5_1.5b/                # ← 从 models/nlp/ 移入
│   ├── qwen3-4b/
│   │   ├── gguf/                    # GGUF 量化版本
│   │   ├── hf/                      # HuggingFace 权重
│   │   └── mnn/                     # MNN 导出格式
│   └── ...
│
└── speech/
```

### 2.3 操作清单

| 操作 | 说明 |
|------|------|
| `git mv models/classification/* models/source/classification/` | 移动 tracked 的源文件 |
| 删除 `.tflite` `.ncnn.bin` `.tnnmodel` `.pnnx.*` `.ms` 等冗余导出文件 | 每个模型只保留 1-2 种 canonical 格式 |
| 将 `.mnn` 文件移至 `exported/mnn/` | 集中管理 |
| 合并 `single_ops_gemm/` + `single_ops_gemm_mnn/` → `single_ops/gemm/` | 去重 |
| 合并 `single_ops_extracted/` → `single_ops/basic/` | 归并 |
| 移动 `models/nlp/` 中的 Qwen 系列到 `models/llm/` | 统一 LLM 目录 |
| 更新 `models.json` 和 `src/models/model_info.h` 中的路径映射 | 保证编译和运行不受影响 |

---

## 3. docs/ — 文档目录重整

### 3.1 当前结构

```
docs/
├── ARM推理Benchmark项目指导.md       ← 中文名
├── 手机端大模型部署全景调研.md        ← URL 编码中文名
├── 项目完整计划.md                   ← 中文名
├── profiling_guide.md               ← OK
├── tvm_deployment_guide.md          ← OK
├── FAQ.md, cfi_explained.md, ...    ← OK
├── superpowers/plans/               ← 4 个实施计划
├── superpowers/specs/               ← 3 个设计文档 + HTML
└── figures/
```

### 3.2 目标结构

```
docs/
├── 01-guides/                       # 使用指南
│   ├── android-benchmark-guide.md   ← 原 ARM推理Benchmark项目指导.md
│   ├── profiling-guide.md
│   ├── profiling-setup-guide.md
│   ├── flamegraph-guide.md
│   ├── tvm-deployment-guide.md
│   ├── strip-symbols-guide.md
│   └── faq.md
│
├── 02-analysis/                     # 分析报告
│   ├── ort-profiling-report.md      ← 原 ort_profiling_report_20260430.md
│   ├── mnn-profiling-analysis.md
│   ├── profiling-status.md
│   ├── cfi-explained.md
│   └── results-sm8250.md
│
├── 03-plans/                        # 项目计划
│   ├── 2026-05-10-benchmark-skills-implementation.md
│   ├── 2026-05-25-mobile-bench-plugin-implementation.md
│   ├── 2026-06-06-benchmark-full-plan.md
│   ├── 2026-06-17-benchmark-repo-structure-design.md  ← 本文档
│   └── tvm-integration-design.md
│
├── 04-designs/                      # 设计方案
│   ├── 2026-05-10-benchmark-skills-design.md
│   ├── 2026-05-25-mobile-bench-plugin-design.md
│   └── mobile-llm-survey.md         ← 原 手机端大模型部署全景调研.md
│
├── full-plan.md                     ← 原 项目完整计划.md（放根目录或迁至 03-plans/）
└── figures/
```

### 3.3 命名规范

- 全部英文小写 + 连字符
- 数字前缀 `01-` `02-` 表示类别，非排序
- 文档内部用 frontmatter 记录日期/状态

### 3.4 操作清单

| 操作 | 说明 |
|------|------|
| `git mv` 重命名中文文名为英文 | 固定大小写和分隔符 |
| 建立 `01-guides/` `02-analysis/` `03-plans/` `04-designs/` | 按功能分类 |
| 迁移 `superpowers/plans/` → `03-plans/` | 统一计划文档 |
| 迁移 `superpowers/specs/` → `04-designs/` | 统一设计文档 |
| 删除 `superpowers/` 空目录 | |
| 更新 `CLAUDE.md` 和 `README.md` 中的文档引用 | 保证链接可访问 |

---

## 4. scripts/ — 脚本目录重整

### 4.1 当前问题

- 40+ 文件平铺根目录
- 2 个 `.deprecated` 文件干扰阅读
- `__pycache__` 目录
- 功能类型混在一起

### 4.2 目标结构

```
scripts/
├── build/                           # 构建
│   ├── build-android.sh
│   ├── build-host-tools.sh
│   └── build.sh
│
├── benchmark/                       # 基准测试
│   ├── run-android.sh
│   ├── run-benchmark.sh
│   ├── run-comprehensive.sh
│   ├── run-single-op.sh
│   ├── run-single-op-tests.sh
│   ├── bench-qwen3-4b.sh
│   ├── build-and-run.sh
│   └── tvm-benchmark-compare.sh
│
├── convert/                         # 模型转换
│   ├── convert-models.sh
│   ├── convert-pnnx.py
│   ├── convert-tflite.py
│   ├── export-qwen3-4b-mnn.sh
│   ├── export-mobilevit.py
│   └── download-pretrained.py (或脚本)
│
├── profile/                         # 性能分析
│   ├── profile-benchmark.sh
│   ├── framework-profiling.sh
│   ├── simpleperf-profile.sh
│   ├── profiling-utils.sh
│   ├── atrace-capture.sh
│   └── perfetto-trace.sh
│
├── analyze/                         # 结果分析
│   ├── generate-report.py
│   ├── analyze-roofline.py
│   ├── analyze-single-op.py
│   ├── validate-results.py
│   ├── extract-op-summary.py
│   ├── extract-model-operators.py
│   └── verify-yolov8.py
│
├── setup/                           # 环境设置
│   ├── setup-test-env.sh
│   ├── restore-test-env.sh
│   └── setup-deps.sh
│
└── utils/                           # 工具
    ├── generate-model-header.py
    ├── generate-single-ops.py
    ├── tvm-docker-build.sh
    └── tvm-build-model.py
```

### 4.3 操作清单

| 操作 | 说明 |
|------|------|
| 创建 7 个子目录 | |
| `git mv` 按功能归并脚本 | 保持可执行权限 |
| 删除 `__pycache__/` 和 `.deprecated` 文件 | |
| 检查重复脚本，合并或删除功能重叠的 | 如 `generate_single_ops.py` vs `generate_single_operator_models.py` |
| 更新脚本内部相互 `source` 的路径 | 保证跨目录引用有效 |
| 更新 `CLAUDE.md` 中脚本路径 | |

---

## 5. src/ + .gitignore + CLAUDE.md

### 5.1 src/ 源码

**原则：不改实现代码，只做确认和路径适配。**

| 操作 | 说明 |
|------|------|
| 更新 `models.json` | 路径匹配新的 models/ 结构 |
| 更新 `src/models/model_info.h` | 路径映射更新 |
| 确认 disabled 后端（TNN/TFLite/QNN）的 `#ifdef` 守卫已到位 | 不改代码，仅确认 |
| 检查并更新 `CMakeLists.txt` 注释 | 使其反映 NCNN/MindSpore Lite 的真实状态 |
| 删除 `models/source/nlp/bert/bert_base/`（如存在且冗余） | 仅清理未跟踪文件 |

### 5.2 .gitignore

| 新增规则 | 原因 |
|----------|------|
| `.zed/` | IDE 配置文件 |
| `*.gguf` `*.safetensors` `*.mtok` `*.mv` `*.msc` | LLM 模型格式 |
| `models/llm/.cache/` `models/llm/.cache_ms/` | HuggingFace 缓存目录 |
| 清理重复的 `.DS_Store` / `.vscode` / `*.swp` 条目 | 去重 |
| 更新 models/ 路径规则 | 匹配新的 source/exported/llm 结构 |

### 5.3 CLAUDE.md

| 更新内容 |
|----------|
| 框架状态表：加入 NCNN (ON) 和 MindSpore Lite (ON) 的实际状态 |
| 项目结构：更新 models/ docs/ scripts/ 新路径 |
| 关键源文件：更新 `model_info.h` 路径和说明 |
| 核心命令：更新脚本路径（`scripts/benchmark/run-android.sh`） |

---

## 6. 影响范围与回滚方案

### 6.1 会影响什么

| 影响面 | 风险 | 处理方式 |
|--------|------|----------|
| 脚本中硬编码的路径 | 中 | 全局 grep 后批量替换 |
| `models.json` 中的路径 | 高 | 必须同步更新 |
| `CLAUDE.md` / README 中的路径 | 低 | 同步更新 |
| 已部署到手机的运行脚本 | 低 | 手机端通过新版本更新 |
| 未跟踪的模型文件（llm/ nlp/） | 低 | 纯移动不涉及引用 |

### 6.2 回滚方案

由于是文件级整理（无 git 历史改写），回滚方式：

```bash
# 恢复到整理前状态
git checkout HEAD -- path/to/file
```

---

## 7. 实施步骤

建议按以下顺序执行：

1. **备份阶段**：确认当前 git 状态，确保工作区干净
2. **.gitignore 先行**：先更新 .gitignore 避免新模型文件意外被跟踪
3. **scripts/** 整理：归类移动脚本，更新内部引用
4. **docs/** 整理：重命名 + 分类 + 更新链接
5. **models/** 整理：路径移动 + 删除冗余 + 更新配置文件
6. **CLAUDE.md + README** 更新
7. **验证阶段**：编译确认 + 路径确认
8. **清理阶段**：删除 __pycache__、.deprecated
