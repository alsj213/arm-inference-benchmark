# mobile-bench 插件实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建 `mobile-bench` Claude Code 插件 + 改进 benchmark 仓库使其可被插件对接

**Architecture:** 分两阶段执行。Phase A 改进 benchmark 仓库（8项，解耦硬编码、标准化 CLI），Phase B 创建独立 mobile-bench npm 插件仓库（skills/agents/rules/scripts）

**Tech Stack:** Shell, Python 3, C++17, CMake, YAML

---

## Phase A: benchmark 仓库改进

### Task A1: 创建 `.benchmarkrc.yml` 配置文件

**Files:**
- Create: `/home/liu/project/newwork/benchmark/.benchmarkrc.yml`
- Create: `/home/liu/project/newwork/benchmark/.benchmarkrc.yml.example`

- [ ] **Step 1: 创建配置文件**

```bash
cat > /home/liu/project/newwork/benchmark/.benchmarkrc.yml << 'EOF'
device:
  adb: /mnt/e/andorid/adb/adb.exe
  id: b08dee23
project:
  build_dir: build_android
  binary: benchmark_inference
  models_dir: models
backends:
  - mnn
  - onnxruntime
models:
  - mobilenetv2
  - resnet50
  - yolov8n
  - bert
ndk:
  path: /home/liu/android-ndk
EOF
```

- [ ] **Step 2: 验证 YAML 格式**

```bash
python3 -c "import yaml; yaml.safe_load(open('.benchmarkrc.yml'))" && echo "YAML OK"
```

- [ ] **Step 3: 创建模板 + gitignore**

```bash
cp .benchmarkrc.yml .benchmarkrc.yml.example
echo ".benchmarkrc.yml" >> .gitignore
```

- [ ] **Step 4: Commit**

```bash
git add .benchmarkrc.yml.example .gitignore
git commit -m "feat: add .benchmarkrc.yml config template for mobile-bench plugin"
```

---

### Task A2: 二进制添加 `--version` 输出

**Files:**
- Modify: `/home/liu/project/newwork/benchmark/src/main.cpp`
- Modify: `/home/liu/project/newwork/benchmark/src/CMakeLists.txt`

- [ ] **Step 1: 读取当前 src/CMakeLists.txt 找到 add_executable 位置**

- [ ] **Step 2: 在 src/CMakeLists.txt 的 add_executable 之后添加编译定义**

```cmake
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_COMMIT_HASH)
    set(GIT_COMMIT_HASH "unknown")
endif()
target_compile_definitions(benchmark_inference PRIVATE
    GIT_COMMIT_HASH="${GIT_COMMIT_HASH}"
)
```

- [ ] **Step 3: 在 main.cpp 的 CommandLineArgs 中添加 `bool show_version = false;`**

- [ ] **Step 4: 在 main.cpp 的 parse_args() 中添加 `--version` 解析**

```cpp
} else if (arg == "--version") {
    args.show_version = true;
```

- [ ] **Step 5: 在 main.cpp 的 main() 开头处理 `--version`**

```cpp
if (args.show_version) {
    printf("benchmark_inference %s\n", GIT_COMMIT_HASH);
    return 0;
}
```

- [ ] **Step 6: 编译验证**

```bash
export ANDROID_NDK=/home/liu/android-ndk
./scripts/build_android.sh
```

- [ ] **Step 7: Commit**

```bash
git add src/main.cpp src/CMakeLists.txt
git commit -m "feat: add --version flag showing git commit hash"
```

---

### Task A3: 二进制添加 `--json` 结构化输出

**Files:**
- Modify: `/home/liu/project/newwork/benchmark/src/common/benchmark.h`
- Modify: `/home/liu/project/newwork/benchmark/src/common/benchmark.cpp`
- Modify: `/home/liu/project/newwork/benchmark/src/main.cpp`

- [ ] **Step 1: 在 benchmark.h 的 BenchmarkResult 结构体中添加 to_json() 声明**

```cpp
std::string to_json() const;
```

- [ ] **Step 2: 在 benchmark.cpp 中实现 to_json()**

```cpp
#include <sstream>

std::string BenchmarkResult::to_json() const {
    std::ostringstream ss;
    ss << "{";
    ss << "\"backend\":\"" << backend_name << "\",";
    ss << "\"model\":\"" << model_name << "\",";
    ss << "\"precision\":\"" << (precision == Precision::FP32 ? "fp32" : precision == Precision::FP16 ? "fp16" : "int8") << "\",";
    ss << "\"threads\":" << num_threads << ",";
    ss << "\"gpu\":" << (use_gpu ? "true" : "false") << ",";
    ss << "\"init_time_ms\":" << init_time_ms << ",";
    ss << "\"peak_memory_kb\":" << peak_memory_kb << ",";
    ss << "\"latency\":{";
    ss << "\"min\":" << latency_stats.min_ms << ",";
    ss << "\"max\":" << latency_stats.max_ms << ",";
    ss << "\"mean\":" << latency_stats.mean_ms << ",";
    ss << "\"p50\":" << latency_stats.p50_ms << ",";
    ss << "\"p90\":" << latency_stats.p90_ms << ",";
    ss << "\"p95\":" << latency_stats.p95_ms << ",";
    ss << "\"p99\":" << latency_stats.p99_ms << ",";
    ss << "\"std_dev\":" << latency_stats.std_dev;
    ss << "},";
    ss << "\"throughput_fps\":" << throughput_fps << ",";
    ss << "\"accuracy\":{";
    ss << "\"passed\":" << (accuracy.passed ? "true" : "false") << ",";
    ss << "\"cosine_similarity\":" << accuracy.cosine_similarity << ",";
    ss << "\"mean_absolute_error\":" << accuracy.mean_absolute_error << ",";
    ss << "\"max_absolute_error\":" << accuracy.max_absolute_error;
    ss << "}";
    ss << "}";
    return ss.str();
}
```

- [ ] **Step 3: 在 main.cpp 的 CommandLineArgs 中添加 `bool json_output = false;`**

- [ ] **Step 4: 在 parse_args() 中添加 `--json` 解析**

```cpp
} else if (arg == "--json") {
    args.json_output = true;
```

- [ ] **Step 5: 在 main() 中根据 json_output 切换输出模式**

在性能结果输出处，json 模式输出 `JSON_RESULT:` 行，非 json 保持原样：

```cpp
if (json_mode) {
    printf("JSON_RESULT: %s\n", result.to_json().c_str());
} else {
    // 保持原有 printf 输出
    printf("\n--- Performance Results ---\n");
    printf("  Init time:  %.2f ms\n", result.init_time_ms);
    utils::print_stats(result.latency_stats);
    printf("  Throughput: %.2f FPS\n", result.throughput_fps);
    printf("  Peak mem:   %zu KB\n", result.peak_memory_kb);
}
```

- [ ] **Step 6: 编译验证**

```bash
export ANDROID_NDK=/home/liu/android-ndk
./scripts/build_android.sh
```

- [ ] **Step 7: Commit**

```bash
git add src/common/benchmark.h src/common/benchmark.cpp src/main.cpp
git commit -m "feat: add --json output mode for structured result parsing"
```

---

### Task A4: 模型信息外部化到 models.json

**Files:**
- Create: `/home/liu/project/newwork/benchmark/models.json`
- Create: `/home/liu/project/newwork/benchmark/scripts/generate_model_header.py`
- Modify: `/home/liu/project/newwork/benchmark/src/main.cpp`
- Modify: `/home/liu/project/newwork/benchmark/src/CMakeLists.txt`

- [ ] **Step 1: 创建 models.json**

```bash
cat > /home/liu/project/newwork/benchmark/models.json << 'EOF'
{
  "models": [
    {"name": "mobilenetv2",     "input_shape": [1, 3, 224, 224], "base_path": "models/classification/mobilenetv2"},
    {"name": "resnet50",        "input_shape": [1, 3, 224, 224], "base_path": "models/classification/resnet50"},
    {"name": "shufflenet_v2",   "input_shape": [1, 3, 224, 224], "base_path": "models/classification/shufflenet_v2"},
    {"name": "yolov8n",         "input_shape": [1, 3, 640, 640], "base_path": "models/detection/yolov8n"},
    {"name": "bert",            "input_shape": [1, 128],          "base_path": "models/nlp/bert"},
    {"name": "whisper_tiny",    "input_shape": [1, 80, 3000],     "base_path": "models/speech/whisper_tiny"}
  ]
}
EOF
```

- [ ] **Step 2: 创建代码生成脚本 scripts/generate_model_header.py**

```bash
cat > /home/liu/project/newwork/benchmark/scripts/generate_model_header.py << 'PYEOF'
#!/usr/bin/env python3
"""从 models.json 生成 C++ 模型注册表代码"""
import json, sys

def generate(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)
    lines = [
        '// Auto-generated by generate_model_header.py — DO NOT EDIT',
        '#include "models/model_info.h"',
        '#include <map>',
        '#include <string>',
        '#include <vector>',
        '',
    ]
    for m in data['models']:
        name = m['name']
        shape = ', '.join(str(d) for d in m['input_shape'])
        base = m['base_path']
        lines.append(f'static ModelInfo s_{name}_info{{"{name}", {{{shape}}}, "{base}"}};')
        lines.append(f'ModelInfo get_{name}_info() {{ return s_{name}_info; }}')
        lines.append('')
    lines.append('const std::map<std::string, ModelInfo>& get_all_models() {')
    lines.append('    static const std::map<std::string, ModelInfo> r = {')
    for m in data['models']:
        lines.append(f'        {{"{m["name"]}", s_{m["name"]}_info}},')
    lines.append('    };')
    lines.append('    return r;')
    lines.append('}')
    lines.append('')
    lines.append('ModelInfo get_model_info(const std::string& name) {')
    lines.append('    auto& r = get_all_models();')
    lines.append('    auto it = r.find(name);')
    lines.append('    return it != r.end() ? it->second : ModelInfo{};')
    lines.append('}')
    lines.append('')
    lines.append('std::vector<std::string> get_all_model_names() {')
    lines.append('    return {')
    for m in data['models']:
        lines.append(f'        "{m["name"]}",')
    lines.append('    };')
    lines.append('}')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Generated {output_path} ({len(data["models"])} models)')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: generate_model_header.py <models.json> <output.h>')
        sys.exit(1)
    generate(sys.argv[1], sys.argv[2])
PYEOF
chmod +x /home/liu/project/newwork/benchmark/scripts/generate_model_header.py
```

- [ ] **Step 3: 在 src/CMakeLists.txt 中添加代码生成步骤**

读取 src/CMakeLists.txt，在 add_executable 之前添加：

```cmake
set(MODELS_JSON ${CMAKE_SOURCE_DIR}/models.json)
set(MODEL_GEN_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/model_registry.h)
add_custom_command(
    OUTPUT ${MODEL_GEN_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_model_header.py ${MODELS_JSON} ${MODEL_GEN_HEADER}
    DEPENDS ${MODELS_JSON} ${CMAKE_SOURCE_DIR}/scripts/generate_model_header.py
    COMMENT "Generating model registry from models.json"
)
add_custom_target(generate_models DEPENDS ${MODEL_GEN_HEADER})
add_dependencies(benchmark_inference generate_models)
```

- [ ] **Step 4: 在 main.cpp 中使用生成的注册表**

删除 main.cpp 中的硬编码 `get_model_info()` 函数（约第86-97行），添加：
```cpp
#include "generated/model_registry.h"
```

修改 main() 中模型列表构造（约第123-128行），改为：
```cpp
std::vector<std::string> models_to_test;
if (args.model == "all") {
    models_to_test = get_all_model_names();
} else {
    models_to_test = {args.model};
}
```

- [ ] **Step 5: 编译验证**

```bash
python3 scripts/generate_model_header.py models.json /tmp/test_registry.h
echo "Header: $(wc -l < /tmp/test_registry.h) lines"
export ANDROID_NDK=/home/liu/android-ndk
./scripts/build_android.sh
```

- [ ] **Step 6: Commit**

```bash
git add models.json scripts/generate_model_header.py src/main.cpp src/CMakeLists.txt
git commit -m "feat: externalize model definitions to models.json via code generation"
```

---

### Task A5: 脚本读取 `.benchmarkrc.yml` 去硬编码

**Files:**
- Modify: `/home/liu/project/newwork/benchmark/scripts/run_benchmark_android.sh`
- Modify: `/home/liu/project/newwork/benchmark/scripts/build_and_run.sh`

- [ ] **Step 1: 修改 run_benchmark_android.sh 的 ADB 检测逻辑（第13-16行）**

替换为：
```bash
CONFIG_FILE="$PROJECT_ROOT/.benchmarkrc.yml"
if [ -f "$CONFIG_FILE" ]; then
    ADB_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c.get('device',{}).get('adb',''))" 2>/dev/null)
    if [ -n "$ADB_PATH" ] && [ -x "$ADB_PATH" ]; then
        adb() { "$ADB_PATH" "$@"; }
    fi
fi
if ! command -v adb &>/dev/null && [ -x "/mnt/e/andorid/adb/adb.exe" ]; then
    adb() { /mnt/e/andorid/adb/adb.exe "$@"; }
fi
```

- [ ] **Step 2: 修改 build_and_run.sh 的 ADB 检测逻辑（第119-121行）**

替换为与 Step 1 相同的配置读取逻辑。

- [ ] **Step 3: 验证**

```bash
python3 -c "import yaml; c=yaml.safe_load(open('.benchmarkrc.yml')); print('ADB:', c['device']['adb'])"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_benchmark_android.sh scripts/build_and_run.sh
git commit -m "refactor: read ADB path from .benchmarkrc.yml, fallback to PATH"
```

---

### Task A6: 按模型名推送对应模型目录

**Files:**
- Modify: `/home/liu/project/newwork/benchmark/scripts/run_benchmark_android.sh`

- [ ] **Step 1: 替换硬编码的模型推送（第100行）**

将 `adb push "$PROJECT_ROOT/models/classification" ...` 替换为智能推送逻辑：

```bash
MODEL_ARG="all"
prev=""
for arg in "$@"; do
    [ "$prev" = "--model" ] && MODEL_ARG="$arg" && break
    prev="$arg"
done

if [ "$MODEL_ARG" = "all" ]; then
    for dir in "$PROJECT_ROOT/models/"*/; do
        cat=$(basename "$dir")
        adb push "$dir" "/data/local/tmp/benchmark/models/$cat"
    done
else
    found=$(find "$PROJECT_ROOT/models" -type d -name "$MODEL_ARG" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        cat=$(echo "$found" | rev | cut -d/ -f2 | rev)
        adb push "$found" "/data/local/tmp/benchmark/models/$cat/$MODEL_ARG"
    else
        echo "WARNING: model dir not found for '$MODEL_ARG', pushing all"
        adb push "$PROJECT_ROOT/models" /data/local/tmp/benchmark/
    fi
fi
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_benchmark_android.sh
git commit -m "fix: push only requested model directory, not hardcoded classification"
```

---

### Task A7: 始终用 ORT 生成 reference output

**Files:**
- Modify: `/home/liu/project/newwork/benchmark/src/main.cpp`

- [ ] **Step 1: 移除 ORT 是否在测试列表中的前置条件**

将 main.cpp 第159-160行的条件：
```cpp
if (std::find(backends_to_test.begin(), backends_to_test.end(), "onnxrt") != backends_to_test.end() ||
    backends_to_test.size() > 1) {
```
改为：
```cpp
// Always attempt ORT reference; if ORT backend unavailable, skip silently
{
```

其余逻辑不变 — ORT 创建失败或 `infer_with_output` 返回 false 时 `has_reference` 为 false，不影响后续。

- [ ] **Step 2: 编译验证**

```bash
export ANDROID_NDK=/home/liu/android-ndk
./scripts/build_android.sh
```

- [ ] **Step 3: Commit**

```bash
git add src/main.cpp
git commit -m "fix: always attempt ORT reference regardless of backend test list"
```

---

### Task A8: Skill 合并去冗余（8→4）

**Files:**
- Create: `skills/benchmark-run/SKILL.md`
- Create: `skills/benchmark-model-prep/SKILL.md`
- Create: `skills/benchmark-profiling/SKILL.md`
- Create: `skills/benchmark-integrate/SKILL.md`
- Modify: `CLAUDE.md`
- Delete: 旧的 8 个 skill 目录

- [ ] **Step 1: 创建 skills/benchmark-run/SKILL.md** — 合并 run-benchmark + android-device-ops + test-environment-control + result-processor

- [ ] **Step 2: 创建 skills/benchmark-model-prep/SKILL.md** — 合并 model-pipeline + android-cross-compile

- [ ] **Step 3: 创建 skills/benchmark-profiling/SKILL.md** — 来自 performance-profiling

- [ ] **Step 4: 创建 skills/benchmark-integrate/SKILL.md** — 来自 integrate-framework

- [ ] **Step 5: 更新 CLAUDE.md 中技能列表** — 替换 8 个为 4 个

- [ ] **Step 6: 删除旧目录 + Commit**

```bash
rm -rf skills/android-device-ops skills/test-environment-control \
       skills/android-cross-compile skills/result-processor \
       skills/model-pipeline skills/run-benchmark \
       skills/performance-profiling skills/integrate-framework
git add skills/ CLAUDE.md
git commit -m "refactor: merge 8 skills into 4, remove hardcoded paths"
```

---

## Phase B: mobile-bench 插件仓库创建

### Task B1: 初始化 npm 包

**Files:**
- Create: `~/project/mobile-bench/package.json`

- [ ] **Step 1: 创建目录并初始化 git**

```bash
mkdir -p ~/project/mobile-bench && cd ~/project/mobile-bench && git init
```

- [ ] **Step 2: 创建 package.json**

```json
{
  "name": "claude-code-mobile-bench",
  "version": "0.1.0",
  "description": "Claude Code plugin for mobile device ML inference benchmarking",
  "keywords": ["claude-code", "plugin", "benchmark", "inference", "onnx", "mnn", "android"],
  "license": "MIT",
  "claude": {
    "type": "plugin",
    "skills": ["skills/*/SKILL.md"],
    "agents": ["agents/*.md"],
    "rules": ["rules/*.md"]
  },
  "files": ["skills/", "agents/", "rules/", "scripts/", "schema/"]
}
```

- [ ] **Step 3: Commit**

```bash
git add package.json && git commit -m "chore: initialize mobile-bench plugin package"
```

---

### Task B2: 创建 4 个 Skill

**Files:**
- Create: `~/project/mobile-bench/skills/mobile-bench-run/SKILL.md`
- Create: `~/project/mobile-bench/skills/mobile-bench-model-prep/SKILL.md`
- Create: `~/project/mobile-bench/skills/mobile-bench-profiling/SKILL.md`
- Create: `~/project/mobile-bench/skills/mobile-bench-integrate/SKILL.md`

- [ ] **Step 1-4: 创建 4 个 SKILL.md**

每个 skill 基于 Phase A Task A8 中合并后的 skill 改写，关键区别：
- 去掉所有 `~` 和绝对路径引用
- 用 `from .benchmarkrc.yml` 代替具体路径
- 引用插件的 scripts/ 工具

- [ ] **Step 5: Commit**

```bash
git add skills/ && git commit -m "feat: add 4 mobile-bench skills"
```

---

### Task B3: 创建 Agent

**Files:**
- Create: `~/project/mobile-bench/agents/mobile-bench-agent.md`

- [ ] **Step 1: 基于 benchmark 仓库的 benchmark-agent.md 改写**

关键改动：
- 去掉硬编码的工作目录 `/home/liu/project/newwork/benchmark`
- 去掉硬编码的 `b08dee23`、ADB 路径
- 所有 Step 中的命令改为从 `.benchmarkrc.yml` 读取配置
- 保留 10 条数据真实性红线
- 保留 7 步强制协议

- [ ] **Step 2: Commit**

```bash
git add agents/ && git commit -m "feat: add mobile-bench-agent with 7-step protocol"
```

---

### Task B4: 创建 Rules

**Files:**
- Create: `~/project/mobile-bench/rules/mobile-bench-integrity.md`

- [ ] **Step 1: 创建数据真实性规则文件**

提取 10 条红线中与路径无关的通用规则，作为自动加载的 rule。

- [ ] **Step 2: Commit**

```bash
git add rules/ && git commit -m "feat: add data integrity rules for benchmarking"
```

---

### Task B5: 创建 Python 工具脚本

**Files:**
- Create: `~/project/mobile-bench/scripts/parse_log.py`
- Create: `~/project/mobile-bench/scripts/generate_report.py`
- Create: `~/project/mobile-bench/scripts/compare_precision.py`

- [ ] **Step 1: 创建 parse_log.py** — 支持 JSON 行解析（优先）和 regex 回退

- [ ] **Step 2: 创建 generate_report.py** — 基于 benchmark 仓库的版本，增加 HTML 输出

- [ ] **Step 3: 创建 compare_precision.py** — 余弦相似度和 MAE 计算

- [ ] **Step 4: Commit**

```bash
git add scripts/ && git commit -m "feat: add Python tools for log parsing and report generation"
```

---

### Task B6: 创建配置 Schema

**Files:**
- Create: `~/project/mobile-bench/schema/benchmarkrc.schema.json`

- [ ] **Step 1: 创建 JSON Schema 用于验证 .benchmarkrc.yml**

- [ ] **Step 2: Commit**

```bash
git add schema/ && git commit -m "feat: add JSON Schema for .benchmarkrc.yml validation"
```

---

### Task B7: 创建 README

**Files:**
- Create: `~/project/mobile-bench/README.md`

- [ ] **Step 1: 编写 README** — 安装、配置、使用示例、链接 benchmark 参考实现

- [ ] **Step 2: Commit**

```bash
git add README.md && git commit -m "docs: add README with install and usage guide"
```

---

## 执行总结

| Phase | Task | 产物 |
|-------|------|------|
| A1 | 配置模板 | `.benchmarkrc.yml` + `.example` |
| A2 | --version | CMake 注入 git hash |
| A3 | --json | benchmark.h/cpp + main.cpp |
| A4 | 模型外部化 | models.json + generate_model_header.py |
| A5 | 去硬编码 | run_benchmark_android.sh, build_and_run.sh |
| A6 | 按模型推送 | run_benchmark_android.sh |
| A7 | ORT reference | main.cpp |
| A8 | Skill 合并 | 8→4 skills |
| B1 | npm 初始化 | package.json |
| B2 | Skills ×4 | 插件 skills/ |
| B3 | Agent ×1 | mobile-bench-agent.md |
| B4 | Rule ×1 | mobile-bench-integrity.md |
| B5 | 工具脚本 ×3 | parse/generate/compare |
| B6 | Schema ×1 | benchmarkrc.schema.json |
| B7 | README | README.md |
