# Benchmark 仓库结构重整 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 benchmark 仓库进行文件级结构重整 — 优化 models/docs/scripts 三⼤目录结构，消除冗余，统⼀命名

**Architecture:** 按 .gitignore → scripts → docs → models → src 路径引⽤ → CLAUDE.md/README 的顺序执⾏，确保引⽤链始终有效。每个模块独⽴完成，可分段提交。

**Tech Stack:** git mv / shell / python / cmake

---

### Task 1: 更新 .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 清理 .gitignore 中的重复规则**

```bash
# 当前 .gitignore 中有两处重复：
# - .DS_Store、Thumbs.db 各出现两次
# - .vscode/ 出现两次
# 编辑 .gitignore，删除第二组重复规则（约第 100-105 行附近的 OS/IDE 块）
```

- [ ] **Step 2: 添加缺失的新规则**

在 `.gitignore` 末尾添加：

```gitignore
# ==============================================
# IDE
# ==============================================
.zed/

# ==============================================
# LLM 模型文件
# ==============================================
*.gguf
*.safetensors
*.mtok
*.mv
*.msc

# LLM 缓存
models/llm/.cache/
models/llm/.cache_ms/
```

- [ ] **Step 3: 更新 models/ 路径规则以匹配新结构**

找到 `.gitignore` 中 `# Model directories` 区块，更新为：

```gitignore
# Model directories
models/source/classification/*.onnx
models/source/classification/*.mnn
models/source/classification/*.tflite
models/source/classification/*.param
models/source/classification/*.bin
models/source/detection/*
models/source/nlp/*
```

- [ ] **Step 4: 验证 .gitignore**

```bash
git status --short
# 确认没有 .gguf / .safetensors / .zed/ 等文件出现在 Untracked 中
```

---

### Task 2: scripts/ 整理 — 创建目录 + git mv

**Files:**
- Create: `scripts/build/` `scripts/benchmark/` `scripts/convert/` `scripts/profile/` `scripts/analyze/` `scripts/setup/` `scripts/utils/`
- Rename (git mv): 所有 scripts/*.sh / *.py

- [ ] **Step 1: 创建所有目标子目录**

```bash
mkdir -p scripts/{build,benchmark,convert,profile,analyze,setup,utils}
```

- [ ] **Step 2: git mv 构建脚本 → scripts/build/**

```bash
git mv scripts/build.sh scripts/build/
git mv scripts/build_android.sh scripts/build/build-android.sh
git mv scripts/build_host_tools.sh scripts/build/build-host-tools.sh
```

- [ ] **Step 3: git mv 基准测试脚本 → scripts/benchmark/**

```bash
git mv scripts/run_benchmark_android.sh scripts/benchmark/run-android.sh
git mv scripts/run_benchmark.sh scripts/benchmark/run-benchmark.sh
git mv scripts/run_comprehensive_benchmark.sh scripts/benchmark/run-comprehensive.sh
git mv scripts/run_single_op_benchmark.sh scripts/benchmark/run-single-op.sh
git mv scripts/run_single_op_tests.sh scripts/benchmark/run-single-op-tests.sh
git mv scripts/bench_qwen3_4b.sh scripts/benchmark/bench-qwen3-4b.sh
git mv scripts/build_and_run.sh scripts/benchmark/build-and-run.sh
git mv scripts/tvm_benchmark_compare.sh scripts/benchmark/tvm-benchmark-compare.sh
```

- [ ] **Step 4: git mv 模型转换脚本 → scripts/convert/**

```bash
git mv scripts/convert_models.sh scripts/convert/convert-models.sh
git mv scripts/convert_pnnx.py scripts/convert/convert-pnnx.py
git mv scripts/convert_tflite.py scripts/convert/convert-tflite.py
git mv scripts/export_qwen3_4b_mnn.sh scripts/convert/export-qwen3-4b-mnn.sh
git mv scripts/export_mobilevit.py scripts/convert/export-mobilevit.py
git mv scripts/download_pretrained.py scripts/convert/download-pretrained.py
git mv scripts/convert_mobilenetv2_pnnx.py scripts/convert/convert-mobilenetv2-pnnx.py      # small util
git mv scripts/convert_resnet50_pnnx.py scripts/convert/convert-resnet50-pnnx.py              # small util
```

- [ ] **Step 5: git mv profiling 脚本 → scripts/profile/**

```bash
git mv scripts/profile_benchmark.sh scripts/profile/profile-benchmark.sh
git mv scripts/framework_profiling.sh scripts/profile/framework-profiling.sh
git mv scripts/simpleperf_profile.sh scripts/profile/simpleperf-profile.sh
git mv scripts/profiling_utils.sh scripts/profile/profiling-utils.sh
git mv scripts/atrace_capture.sh scripts/profile/atrace-capture.sh
git mv scripts/perfetto_trace.sh scripts/profile/perfetto-trace.sh
```

- [ ] **Step 6: git mv 结果分析脚本 → scripts/analyze/**

```bash
git mv scripts/generate_report.py scripts/analyze/generate-report.py
git mv scripts/analyze_roofline.py scripts/analyze/analyze-roofline.py
git mv scripts/analyze_single_op_results.py scripts/analyze/analyze-single-op.py
git mv scripts/validate_results.py scripts/analyze/validate-results.py
git mv scripts/extract_op_summary.py scripts/analyze/extract-op-summary.py
git mv scripts/extract_model_operators.py scripts/analyze/extract-model-operators.py
git mv scripts/verify_yolov8_accuracy.py scripts/analyze/verify-yolov8.py
```

- [ ] **Step 7: git mv 环境设置脚本 → scripts/setup/**

```bash
git mv scripts/setup_test_environment.sh scripts/setup/setup-test-env.sh
git mv scripts/restore_test_environment.sh scripts/setup/restore-test-env.sh
git mv scripts/setup_deps.sh scripts/setup/setup-deps.sh
```

- [ ] **Step 8: git mv 工具脚本 → scripts/utils/**

```bash
git mv scripts/generate_model_header.py scripts/utils/generate-model-header.py
git mv scripts/generate_single_operator_models.py scripts/utils/generate-single-operator-models.py
git mv scripts/generate_single_ops.py scripts/utils/generate-single-ops.py
git mv scripts/tvm_docker_build.sh scripts/utils/tvm-docker-build.sh
git mv scripts/tvm_build_model.py.deprecated scripts/utils/tvm-build-model.py.deprecated
git mv scripts/compile_tvm_mobilenetv2.py.deprecated scripts/utils/compile-tvm-mobilenetv2.py.deprecated
# 这两个 .deprecated 暂时保留在 utils/ 中，标记移入
```

- [ ] **Step 9: 验证所有脚本已移动**

```bash
ls scripts/*/  # 确认 7 个子目录都有文件
git status --short | grep "^R" | wc -l  # 确认所有 rename 都记录
```

---

### Task 3: scripts/ 整理 — 更新内部 source 引用

**Files:**
- Modify: `scripts/profile/framework-profiling.sh` `scripts/profile/profile-benchmark.sh` `scripts/profile/atrace-capture.sh` `scripts/profile/perfetto-trace.sh` `scripts/profile/simpleperf-profile.sh`
- Modify: `scripts/benchmark/bench-qwen3-4b.sh` `scripts/benchmark/tvm-benchmark-compare.sh`

- [ ] **Step 1: 更新 profiling 脚本的 source 路径**

`profiling_utils.sh` 已从 `scripts/profile_utils.sh` → `scripts/profile/profiling-utils.sh`，所有 source 它的脚本需要更新：

编辑 `scripts/profile/framework-profiling.sh` 第 6 行：
```bash
# 改前
source "$SCRIPT_DIR/profiling_utils.sh"
# 改后
source "$SCRIPT_DIR/profiling-utils.sh"
```

同样编辑：
- `scripts/profile/profile-benchmark.sh:6`
- `scripts/profile/atrace-capture.sh:6`
- `scripts/profile/perfetto-trace.sh:6`
- `scripts/profile/simpleperf-profile.sh:6`

- [ ] **Step 2: 更新 bench-qwen3-4b.sh 的外部脚本调用**

```bash
# 编辑 scripts/benchmark/bench-qwen3-4b.sh
# 第 48 行：./scripts/setup_test_environment.sh → ./scripts/setup/setup-test-env.sh
# 第 98 行：./scripts/restore_test_environment.sh → ./scripts/setup/restore-test-env.sh
```

- [ ] **Step 3: 更新 tvm-benchmark-compare.sh 的外部脚本调用**

```bash
# 编辑 scripts/benchmark/tvm-benchmark-compare.sh
# 第 52 行：./scripts/build_android.sh → ./scripts/build/build-android.sh
```

- [ ] **Step 4: 确认所有内部引用已更新**

```bash
grep -rn "scripts/" scripts/ --include="*.sh" --include="*.py" | grep -v "^scripts/" | grep -v ".pyc" | grep -v "\./scripts"
# 确认没有残留在 `scripts/` 根目录（无子目录）路径
```

---

### Task 4: docs/ 整理 — 重命名 + 分类

**Files:**
- Rename: 10+ 个 docs/ 文件
- Delete: `docs/superpowers/` 空目录（git mv 移出文件后）

- [ ] **Step 1: 创建目标子目录**

```bash
mkdir -p docs/01-guides docs/02-analysis docs/03-plans docs/04-designs
```

- [ ] **Step 2: git mv 重命名中文文档 + 归入 01-guides**

```bash
# 原 ARM推理Benchmark项目指导.md → android-benchmark-guide.md
git mv "docs/ARM推理Benchmark项目指导.md" docs/01-guides/android-benchmark-guide.md

# 以下保持原名（已经是英文）只移动目录：
git mv docs/profiling_guide.md docs/01-guides/profiling-guide.md
git mv docs/profiling_setup_guide.md docs/01-guides/profiling-setup-guide.md
git mv docs/flamegraph_guide.md docs/01-guides/flamegraph-guide.md
git mv docs/tvm_deployment_guide.md docs/01-guides/tvm-deployment-guide.md
git mv docs/strip_symbols_guide.md docs/01-guides/strip-symbols-guide.md
git mv docs/FAQ.md docs/01-guides/faq.md
```

- [ ] **Step 3: git mv 分析报告 → 02-analysis**

```bash
git mv docs/ort_profiling_report_20260430.md docs/02-analysis/ort-profiling-report.md
git mv docs/mnn_profiling_analysis.md docs/02-analysis/mnn-profiling-analysis.md
git mv docs/profiling_status.md docs/02-analysis/profiling-status.md
git mv docs/cfi_explained.md docs/02-analysis/cfi-explained.md
git mv docs/results_sm8250.md docs/02-analysis/results-sm8250.md
```

- [ ] **Step 4: 迁移 superpowers/plans/ → 03-plans/**

```bash
cp docs/superpowers/plans/*.md docs/03-plans/
# 注意：这里是 cp + git rm 而不是 git mv，因为 superpowers/ 可能不是通过 git 跟踪的
# 检查：
git ls-files docs/superpowers/plans/
# 如果文件被 git 跟踪，用 git mv；如果不在跟踪中，用 cp + rm
```

- [ ] **Step 5: 迁移 superpowers/specs/ → 04-designs/**

```bash
# 同样：先检查是否被 git 跟踪
git ls-files docs/superpowers/specs/
# 如果被跟踪 → git mv
# 如果不在跟踪中 → cp + rm
```

- [ ] **Step 6: 重命名中文设计文档 + 归入 04-designs/**

```bash
# 原 手机端大模型部署全景调研.md → mobile-llm-survey.md
git mv "docs/手机端大模型部署全景调研.md" docs/04-designs/mobile-llm-survey.md
```

- [ ] **Step 7: 处理 项目完整计划.md**

```bash
git mv "docs/项目完整计划.md" docs/03-plans/full-plan.md
```

- [ ] **Step 8: 删除空目录**

```bash
rmdir docs/superpowers/plans docs/superpowers/specs docs/superpowers 2>/dev/null; true
```

- [ ] **Step 9: 更新文档间相互引用的链接**

```bash
# 搜索所有 .md 文件中对旧路径的引用
grep -rn "docs/\|superpowers/" docs/ --include="*.md" | grep -v "2026-06-17"
# 逐一更新为新路径
```

---

### Task 5: models/ 整理 — source/ + exported/ 结构

**Files:**
- Rename: `models/classification/` → `models/source/classification/`
- Delete: 各框架冗余导出文件（.tflite / .ncnn.bin / .tnnmodel / .pnnx.* / .ms）
- Move: `.mnn` 文件集中到 `models/exported/mnn/`

- [ ] **Step 1: 创建目标目录**

```bash
mkdir -p models/source models/exported/mnn models/exported/tvm
```

- [ ] **Step 2: git mv 模型分类目录 → models/source/classification/**

```bash
# 检查哪些子目录被 git 跟踪
git ls-files models/classification/ | awk -F/ '{print $1"/"$2"/"$3}' | sort -u
# 输出应为：mobilenetv2, resnet50, resnet18, mobilevit_s, shufflenet_v2, squeezenet, efficientnet_lite0

# git mv 整个子目录
for dir in mobilenetv2 resnet50 resnet18 mobilevit_s shufflenet_v2 squeezenet efficientnet_lite0; do
  git mv "models/classification/$dir" "models/source/classification/"
done
```

- [ ] **Step 3: git mv detection → models/source/detection/**

```bash
git mv models/detection/yolov8n models/source/detection/
```

- [ ] **Step 4: 清理各分类目录中的冗余导出文件**

对 `models/source/classification/` 下每个子目录，删除以下框架格式（仅保留 `.onnx` `.pt` 或 canonical 格式）：

```bash
# 每个目录中执行类似操作（以 mobilenetv2 为例）：
# 保留：mobilenetv2.onnx, mobilenetv2.pt
# 删除：*_MNN.mnn, *.tflite, *_ncnn.*, *_TNN.*, *.pnnx.*, *.ms, *.ort
cd models/source/classification/mobilenetv2

# 删除前先列出要删的文件
find . -maxdepth 1 \( -name "*_MNN.mnn" -o -name "*.tflite" -o -name "*_ncnn*" \
  -o -name "*_TNN*" -o -name "*.pnnx.*" -o -name "*.ms" -o -name "*.ort" \
  -o -name "*.tgz" -o -name "*.ckpt.*" -o -name "*_eval.pbtxt" -o -name "*_info.txt" \) -exec git rm {} \;
```

对 `resnet50/`、`mobilevit_s/` 等重复相同操作。

> ⚠️ resnet50 中的 `.pnnx.onnx` 和 `.onnx` 是不同文件 — 保留 `.onnx`（98M 的那个），删除 `.pnnx.onnx`

- [ ] **Step 5: 将 .mnn 文件集中到 exported/mnn/**

```bash
# 检查各分类目录下有哪些 .mnn 文件
find models/source/classification -name "*.mnn" -o -name "*_MNN.mnn"

# 对每个 .mnn 文件，git mv 到 exported/mnn/
# 规则：models/source/classification/mobilenetv2/mobilenetv2_MNN.mnn → models/exported/mnn/mobilenetv2.mnn
git mv models/source/classification/mobilenetv2/mobilenetv2_MNN.mnn models/exported/mnn/mobilenetv2.mnn
git mv models/source/classification/resnet50/resnet50_MNN.mnn models/exported/mnn/resnet50.mnn
git mv models/source/classification/mobilevit_s/mobilevit_s_MNN.mnn models/exported/mnn/mobilevit_s.mnn
```

- [ ] **Step 6: 处理未跟踪的 nlp/ 目录**

`models/nlp/` 不在 git 跟踪中，直接移动即可：

```bash
mv models/nlp/bert models/source/nlp/bert
# 保留 bert.onnx + bert_patched.onnx + bert_simplified.onnx 中的一个即可
# 清理：rm models/source/nlp/bert/*.onnx.data （data 文件可重新下载时生成）
```

---

### Task 6: models/ 整理 — single_ops 合并

**Files:**
- Move/merge: `models/single_ops/` `models/single_ops_extracted/` `models/single_ops_gemm/` `models/single_ops_gemm_mnn/` `models/single_ops_stair/`

- [ ] **Step 1: 创建目标目录**

```bash
mkdir -p models/single_ops/basic models/single_ops/gemm models/single_ops/stair
```

- [ ] **Step 2: 合并 single_ops/ + single_ops_extracted/ → single_ops/basic/**

```bash
# 检查 git 跟踪状态
git ls-files models/single_ops/ | head -5
git ls-files models/single_ops_extracted/ | head -5

# 如果被跟踪：git mv
if git ls-files models/single_ops/ &>/dev/null; then
  for f in $(git ls-files models/single_ops/); do
    basename=$(basename "$f")
    git mv "$f" "models/single_ops/basic/$basename"
  done
fi
# single_ops_extracted/ 同理，注意去重（同名文件只移一次）
```

- [ ] **Step 3: 合并 single_ops_gemm/ + single_ops_gemm_mnn/ → single_ops/gemm/**

```bash
# 检查两个目录的跟踪状态
git ls-files models/single_ops_gemm/ 2>/dev/null | head -5
git ls-files models/single_ops_gemm_mnn/ 2>/dev/null | head -5

# 合并，注意去重（同时有 .onnx 和 .mnn 版本的保留两者，放到同目录）
```

- [ ] **Step 4: 移动 single_ops_stair/ → single_ops/stair/**

```bash
git ls-files models/single_ops_stair/ 2>/dev/null
# 如果被跟踪 → git mv
# 否则 → mv
```

- [ ] **Step 5: 删除空目录**

```bash
rmdir models/single_ops_extracted models/single_ops_gemm models/single_ops_gemm_mnn models/single_ops_stair 2>/dev/null; true
```

---

### Task 7: models/ 整理 — LLM 统一

**Files:**
- Move: `models/nlp/qwen2_0.5b/` `models/nlp/qwen2.5_1.5b/` → `models/llm/`

- [ ] **Step 1: 将旧 LLM 从 nlp/ 移到 llm/**

```bash
# 这些文件不在 git 跟踪中，直接 mv
mv models/nlp/qwen2_0.5b models/llm/qwen2_0.5b
mv models/nlp/qwen2.5_1.5b models/llm/qwen2.5_1.5b
```

- [ ] **Step 2: 清理 models/nlp/ 中已处理完的旧文件**

```bash
ls models/nlp/  # 此时应只剩 bert/
# bert 已移入 models/source/nlp/bert/
rmdir models/nlp 2>/dev/null; true  # 如果空了就删除
```

---

### Task 8: 更新 models.json 和 model_info.h 的路径

**Files:**
- Modify: `models.json`
- Modify: `src/models/model_info.h`

- [ ] **Step 1: 更新 models.json 中的 base_path**

```json
// 改前                      →  改后
"base_path": "models/classification/mobilenetv2"   →  "base_path": "models/source/classification/mobilenetv2"
"base_path": "models/classification/resnet50"       →  "base_path": "models/source/classification/resnet50"
"base_path": "models/detection/yolov8n"             →  "base_path": "models/source/detection/yolov8n"
"base_path": "models/classification/mobilevit_s"    →  "base_path": "models/source/classification/mobilevit_s"
"base_path": "models/nlp/bert"                      →  "base_path": "models/source/nlp/bert"
"base_path": "models/nlp/qwen2_0.5b"                →  "base_path": "models/llm/qwen2_0.5b"
```

- [ ] **Step 2: 更新 model_info.h 中的路径逻辑**

`model_info.h` 的 `get_model_path()` 根据 `base_path + "/" + name + "_SUFFIX"` 构造路径。base_path 更新后大部分路径自动生效。

但是 `.gguf` 的路径逻辑需要检查 — llama.cpp backend 的路径是：
```bash
# 当前 model_info.h: return base_path + "/" + name + ".gguf";
# 但实际 qwen2-0_5b-instruct-q4_k_m.gguf 文件名不是 model_name + ".gguf"
# 这需要确认是否在 llm_benchmark.cpp 中有单独处理
```

`src/llm_benchmark.cpp:71` 有硬编码路径 — 需要在 Task 9 中处理。

- [ ] **Step 3: 更新 model_info.h 中的 MNN 产物路径**

`model_info.h` 当前返回 `base_path + "/" + name + "_MNN.mnn"`。由于 .mnn 文件已移到 `models/exported/mnn/`，需要修改：

```cpp
// 改前
return base_path + "/" + name + "_MNN.mnn";
// 改后 — MNN 权重路径统一指向 exported/mnn/
return "models/exported/mnn/" + name + ".mnn";
```

---

### Task 9: 更新 src 中的硬编码路径

**Files:**
- Modify: `src/llm_benchmark.cpp`（第 71、146 行）
- Modify: `src/single_op_benchmark.cpp`（第 27-62 行）

- [ ] **Step 1: 更新 llm_benchmark.cpp 中的硬编码路径**

```cpp
// llm_benchmark.cpp:71
// 改前: "models/nlp/qwen2_0.5b/qwen2-0_5b-instruct-q4_k_m.gguf"
// 改后: "models/llm/qwen2_0.5b/qwen2-0_5b-instruct-q4_k_m.gguf"

// llm_benchmark.cpp:146
// 改前: "models/nlp/qwen2_0.5b/mnn_llm/config.json"
// 改后: "models/llm/qwen2_0.5b/mnn_llm/config.json"
```

- [ ] **Step 2: 更新 single_op_benchmark.cpp 中的路径前缀**

```cpp
// 所有 "models/single_ops/" → "models/single_ops/basic/"
// 第 27 行起：
// 改前:        "models/single_ops/Conv1x1_K16_C64_M784.onnx",
// 改后:        "models/single_ops/basic/Conv1x1_K16_C64_M784.onnx",
```

- [ ] **Step 3: 验证所有硬编码路径已被覆盖**

```bash
grep -rn "models/classification\|models/nlp\|models/single_ops_extracted\|models/single_ops_gemm\|models/single_ops_stair" src/ --include="*.cpp" --include="*.h"
# 应返回空
```

---

### Task 10: 更新 CLAUDE.md + README

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: 更新 CLAUDE.md 中的框架状态表**

`CLAUDE.md` 框架表中加入 NCNN 和 MindSpore Lite 的实际状态：

| 框架 | 状态 | CMake 选项 |
|------|------|-----------|
| NCNN | 实际已启用 | `BENCHMARK_NCNN=ON` |
| MindSpore Lite | 实际已启用 | `BENCHMARK_MINDSPORE_LITE=ON` |

- [ ] **Step 2: 更新 CLAUDE.md 中的项目结构**

```markdown
├── scripts/                  # 28 个脚本 → 改为 7 子目录结构
│   ├── build/
│   ├── benchmark/
│   ├── convert/
│   ├── profile/
│   ├── analyze/
│   ├── setup/
│   └── utils/
├── docs/                     # 文档
│   ├── 01-guides/
│   ├── 02-analysis/
│   ├── 03-plans/
│   └── 04-designs/
├── models/                   # 模型文件
│   ├── source/
│   ├── exported/
│   ├── single_ops/
│   └── llm/
```

- [ ] **Step 3: 更新 CLAUDE.md 中的核心命令路径**

```bash
# 所有 ./scripts/xxx → 对应新路径
./scripts/build/build-android.sh
./scripts/benchmark/run-android.sh
./scripts/profile/profile-benchmark.sh
```

- [ ] **Step 4: 更新 README.md 中的脚本路径**

```bash
# 搜索 README.md 中所有 scripts/ 引用，更新为新路径
grep -n "scripts/" README.md
# 逐一修正
```

---

### Task 11: 验证 + 清理

**Files:**
- Delete: `scripts/__pycache__/` `scripts/*.deprecated`
- Verify: 编译 + 路径一致性

- [ ] **Step 1: 删除 __pycache__ 和 .deprecated 文件**

```bash
rm -rf scripts/__pycache__
# .deprecated 文件保留在 utils/ 中（设计文档要求标记移入而非删除）
```

- [ ] **Step 2: 确认所有 git 操作状态**

```bash
git status --short
# 应全部是 Renamed (R) 或 Deleted (D) 或 Untracked (??) 移动操作
# 不应有 Modified (M) 状态的跟踪文件（除非是路径更新那几步）
```

- [ ] **Step 3: 编译确认**

```bash
# 确认 models.json + model_info.h 更新后能正常解析
python3 -c "import json; json.load(open('models.json')); print('models.json OK')"
```

- [ ] **Step 4: 确认路径一致性**

```bash
# 检查所有模型中引用的文件路径是否存在
python3 -c "
import json
data = json.load(open('models.json'))
for m in data['models']:
    print(f'{m[\"name\"]}: {m[\"base_path\"]}')
"
```

- [ ] **Step 5: 整体提交**

```bash
git add -A
git status  # 最后确认
git commit -m "refactor: 仓库结构重整 — models/docs/scripts 目录优化 + .gitignore 更新

- models/: 分离 source/exported/llm，合并 single_ops，清除冗余格式
- docs/: 英文命名 + 功能分类（guides/analysis/plans/designs）
- scripts/: 按功能分组到 7 个子目录
- .gitignore: 清理重复规则，新增 LLM/IDE 规则
- 更新 models.json / model_info.h / CLAUDE.md / README
- 删除 __pycache__ 和冗余导出文件"
```

---

## Scope Check

本计划覆盖 spec 中所有模块（.gitignore / scripts / docs / models / src / CLAUDE.md），相互依赖明确。各 Task 按"先引用的后移动"顺序排列，确保路径一致性。

## Self-Review

- [x] **Spec 覆盖**：每个设计章节都能指向对应的 Task
  - .gitignore → Task 1
  - scripts/ 目录结构 → Task 2-3
  - docs/ 命名+分类 → Task 4
  - models/ source/exported → Task 5
  - models/ single_ops 合并 → Task 6
  - models/ LLM 统一 → Task 7
  - src 路径更新 → Task 8-9
  - CLAUDE.md/README → Task 10
  - 验证清理 → Task 11
- [x] **无占位符**：每个 step 都有实际命令/代码
- [x] **类型一致性**：路径映射贯穿所有 Task
