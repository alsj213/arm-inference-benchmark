# mobile-bench 收口锚点：vendor 融合 + 上游保留决策记录

> 日期：2026-09-06
> 关联历史设计：[2026-05-25-mobile-bench-plugin-design.md](./2026-05-25-mobile-bench-plugin-design.md)
> 本文件是"以 benchmark 为主仓、mobile-bench 内容融合进本仓、上游独立仓保留"这一架构决策的**锚点**，用于后续维护时对齐双仓关系与同步方向。

## 1. 决策摘要

1. **以 `benchmark` 仓为主仓**（本仓拥有 build 纵深：C++ / NDK / 5 个 third_party 子模块 + 真实模型 + 实测数据）。
2. **mobile-bench 内容融合进本仓**，形态为 **现状 vendor 分布**（方案 1，不新建 `tools/mobile-bench/` 净副本），因为其已是"本仓定制活体"：`scripts/analyze/mobilebench/` 被真实报告流程引用，移动会造成断链与双份重复。
3. **mobile-bench 上游独立仓保留，不删除**。后续它将**整改为通用类型、与本项目零依赖**（可独立发布 / 供其他契约项目使用）。
4. 后期可按需重构 vendor 布局，本决策不锁死结构。

## 2. 两仓定位与关系

| 仓 | 定位 | 形态 |
|---|---|---|
| `benchmark`（本仓，主） | 实测引擎 + 跑数宿主 + mobile-bench 定制融合层 | C++/NDK/子模块深 build；vendor 快照命令改指本仓真实脚本 |
| `claude-code-mobile-bench`（上游，保留） | 通用协议源 / 插件，目标零本仓依赖 | 纯 md + py、零 build；`claude plugins install` 可分发的插件 |

关系：上游为**通用版源**，本仓为**定制版**。本仓 vendor 快照因路径重写（指向本仓真实脚本）与上游**不可直接 diff 合入**；同步一律经人工 diff + 本文件锚点核对。

上游锚点：`https://github.com/alsj213/claude-code-mobile-bench` · commit `beb0c6a` · package.json `v0.3.0`（本地 clone：`/home/liu/project/mobile-bench`）。

## 3. 本仓 vendor 分布清单（融合文件）

| 位置 | 内容 | 上游来源 |
|---|---|---|
| `.claude/skills/mobile-bench-{run,llm,model-prep,methodology,profiling,integrate,memory,power,thermal,regression}/` | 10 个 skill（SKILL.md） | `skills/` |
| `.claude/agents/mobile-bench-agent.md` | 协议 agent | `agents/` |
| `.claude/rules/mobile-bench-integrity.md` | 数据真实性规则 | `rules/` |
| `scripts/analyze/mobilebench/*.py`（7 工具） | parse/generate/stats/env/power/regression/compare | `scripts/` |
| `scripts/analyze/mobilebench/tests/` | 单测（52 通过） | `tests/` |
| `.benchmarkrc.schema.json` | 配置 schema | `schema/benchmarkrc.schema.json` |
| `scripts/benchmark/bench-qwen3-*.sh` | 本仓新增实测脚本（3way/2×2） | 本仓自增 |

差异性质：脚本/schema/rule 与上游逐字节一致；agent 与多数 skill 仅做路径改写；`mobile-bench-llm` 等按本仓真实工具链（`llm_benchmark` 统一二进制 + `--require-precision` 精度对齐）重写。

## 4. 收口动作记录（2026-09-06）

### 4.1 漂移命令修复（vendor 未对齐本仓真实脚本，本次已修）

| 文件 | 旧（漂移，不存在） | 新（本仓真实） |
|---|---|---|
| `mobile-bench-model-prep` | `./scripts/build_host_tools.sh` | `./scripts/build/build-host-tools.sh` |
| 同上 | `python3 scripts/compile_tvm_model.py` | `python3 tools/tvm/compile_model_relax.py <model>`（env 见脚本 docstring） |
| 同上 | `./scripts/setup_deps.sh` | `./scripts/setup/setup-deps.sh` |
| 同上 | `./converter_lite`（歧义） | 注明为 MindSpore Lite SDK 外部工具，非本仓脚本 |
| `mobile-bench-profiling` | `./scripts/profile_benchmark.sh` | `./scripts/profile/profile-benchmark.sh` |
| 同上 | `./scripts/simpleperf_profile.sh` | `./scripts/profile/simpleperf-profile.sh` |
| 同上 | `./scripts/atrace_capture.sh` | `./scripts/profile/atrace-capture.sh` |
| 同上 | `./scripts/perfetto_trace.sh` | `./scripts/profile/perfetto-trace.sh` |
| 同上 | `build_android.sh --debug`（缺前缀） | `./scripts/build/build-android.sh --debug` |

修复后已对两 SKILL 全文复查无残留漂移（`grep` 验证通过）。

## 5. 同步策略（维护约定）

- **本仓为权威定制版**：agent/skill 引用一律指向本仓真实脚本，不与上游逐字对齐。
- **上游改动回灌本仓**：人工 diff，仅合入与路径无关的实质逻辑（协议、参数表、规则），路径类一律按本仓改写。
- **本仓定制成果反哺上游（推荐，配合第 6 节）**：把本仓已验证的协议/参数改进整理回上游通用版，方向优先于上游直接改。

## 6. 重构 TODO（mobile-bench 上游 → 通用零依赖）

- [ ] 移除/参数化对 benchmark 特定脚本路径的硬引用（`build-android.sh`、`benchctl.sh`、`llm_benchmark`、`scripts/convert/*`、`scripts/analyze/mobilebench/*`），改为契约化占位（如 `<PROJECT_ROOT>/scripts/...` + `.benchmarkrc.yml` 解析）。
- [ ] 与上游自身 OpenSpec/ECC 开发工作区解耦，CLAUDE.md 恢复为插件自身说明。
- [ ] 升级完整插件形态评估：hooks（温度门禁/日志校验，强制执行数据真实性）→ 需运行时，仅插件形态可承载；commands（`/bench-run`）等。
- [ ] 决策提示：若上游升级带 hooks → 只能以插件安装，本仓 vendor 仅覆盖内容层，双仓须各自保持（见会话决策记录）。

## 7. 相关参考

- 会话决策过程：CLAUDE.md「Benchmark 执行协议」/「项目级 Skill」章节来源说明
- `2026-05-25-mobile-bench-plugin-design.md`（原始插件设计）
- 上游 README / package.json（`claude.type=plugin` 声明）
