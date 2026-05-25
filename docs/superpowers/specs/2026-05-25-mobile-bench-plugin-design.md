# mobile-bench 插件设计

## 概述

`mobile-bench` 是一个 Claude Code 插件（npm 包 `claude-code-mobile-bench`），封装手机端深度学习推理框架性能基准测试的完整经验。插件通过 `.benchmarkrc.yml` 配置文件与用户的 benchmark 项目解耦，不硬依赖任何具体仓库。

## 仓库关系

```
claude-code-mobile-bench/       # npm 插件（新建）
├── skills/      ← 领域知识 + 工作流
├── agents/      ← benchmark-agent（7步协议）
├── rules/       ← 数据真实性红线
├── scripts/     ← 纯 Python 工具（无硬编码）
└── schema/      ← .benchmarkrc.yml JSON Schema

        ↑ 通过 .benchmarkrc.yml 契约对接 ↓

benchmark/                      # 参考实现（当前仓库）
├── .benchmarkrc.yml            # 新增：用户配置
├── src/                        # C++ benchmark 二进制
├── scripts/                    # shell 构建/运行脚本
└── models/                     # 模型文件
```

## CLI 契约

插件要求用户的 benchmark 二进制遵守标准接口：

```bash
benchmark_inference --model <name> --backend <name> --threads N --runs N --json
```

必须支持的参数：`--model`, `--backend`, `--precision`, `--threads`, `--warmup`, `--runs`, `--gpu`, `--json`, `--version`

`--json` 输出标准结构化格式，供脚本解析。

## 配置文件契约

用户项目根目录的 `.benchmarkrc.yml`：

```yaml
device:
  adb: /path/to/adb
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
```

## 插件内容

### Skills（4个）

| Skill | 职责 |
|-------|------|
| `benchmark-run` | 7步完整流程：设备握手→版本→二进制→模型→执行→解析→报告 |
| `benchmark-model-prep` | 模型下载、格式转换、路径管理 |
| `benchmark-profiling` | 逐算子 profiling、simpleperf 火焰图、perfetto trace |
| `benchmark-integrate` | 添加新推理框架后端的完整步骤 |

### Agent（1个）

`benchmark-agent` — 可 dispatch 的子 agent，严格执行 7 步协议 + 10 条数据真实性红线。从 `.benchmarkrc.yml` 读取配置，所有数字从命令输出提取。

### Rules（1条）

`benchmark-integrity.md` — 全局生效的数据真实性红线：
- 不得编造性能数字
- 不得跳过设备握手
- 每行数据标注来源命令
- 温度 > 45°C 标注降频风险
- 必须 tee 保存原始日志

### Scripts（3个 Python）

| 脚本 | 功能 |
|------|------|
| `parse_log.py` | 从原始日志提取结构化指标 |
| `generate_report.py` | 生成 Markdown + HTML 报告 |
| `compare_precision.py` | 精度对比（余弦相似度） |

## benchmark 仓库改进清单

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| 阻塞 | 创建 `.benchmarkrc.yml` | 解耦硬编码的设备ID、ADB路径、NDK路径 |
| 阻塞 | 二进制添加 `--json` 输出 | 替代 grep 解析，供脚本标准化消费 |
| 阻塞 | 二进制添加 `--version` | 输出 git commit hash，保证可追溯 |
| 高 | 模型信息外部化 | 从 JSON/YAML 配置读取，加模型不重新编译 |
| 高 | 脚本去硬编码 | run_benchmark_android.sh、build_and_run.sh |
| 中 | 按模型名推送对应模型目录 | 当前只推 classification |
| 中 | 始终用 ORT 生成 reference | 不管 ORT 是否在测试列表中 |
| 低 | skill 合并去冗余 | 8→4，为提取到插件做准备 |
