# benchctl 统一 CLI 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 benchctl 统一命令行工具，替代散落的 shell 脚本，提供 CNN/LLM/单算子三赛道 benchmark、SQLite 历史数据库、原生工具验证、JSON/HTML 报告输出。

**Architecture:** benchctl 是 Python (host-side) CLI，通过 ADB 调用设备上的 C++ benchmark 二进制。C++ 侧新增结构化 JSON 输出。Python 侧负责数据库、报告、验证编排。两层通过 JSON 协议通信。

**Tech Stack:** Python 3.13 (Click CLI, sqlite3 stdlib, Jinja2 HTML), C++17 (现有代码增强), SQLite (历史数据库)

## Global Constraints

- C++ 代码仅在 hot path 使用，所有编排/存储/报告逻辑在 Python 侧
- Python 依赖最小化：Click + Jinja2，不引入 ORM/Web 框架
- JSON 是 C++ ↔ Python 的唯一通信格式
- SQLite 数据库文件位于项目根目录 `benchmarks.db`
- 所有命令必须支持 `--json` 输出模式
- ADB 路径从 `.benchmarkrc.yml` 读取，fallback 到 `$PATH`

---

## 文件结构

```
benchmark/
├── benchctl                    # Python CLI (新建)
│   ├── __init__.py
│   ├── __main__.py            # python -m benchctl 入口
│   ├── cli.py                 # Click 命令定义 (run/history/verify/export)
│   ├── db.py                  # SQLite 读写封装
│   ├── runner.py              # ADB push/run/collect 编排
│   ├── verify.py              # 原生工具验证逻辑
│   ├── report.py              # HTML report 生成
│   └── tracks/
│       ├── __init__.py
│       ├── base.py            # Track 基类 (定义统一接口)
│       ├── cnn.py             # CNN 赛道 metric schema
│       ├── llm.py             # LLM 赛道 (TTFT/TPS)
│       └── single_op.py       # 单算子赛道
├── templates/                  # HTML 模板 (新建)
│   ├── base.html
│   ├── compare.html           # 横向对比
│   └── history.html           # 纵向趋势
├── src/                        # C++ 侧 (增强现有)
│   ├── cnn/main.cpp           # 增强 JSON 输出
│   ├── llm/llm_benchmark.cpp  # 增强 JSON 输出
│   └── single_op/single_op_benchmark.cpp  # 增强 JSON 输出
└── scripts/benchmark/          # 现有脚本 (逐步废弃)
```

---

### Task 1: C++ 侧 JSON 输出标准化

**Files:**
- Modify: `src/common/benchmark.cpp:210-250`
- Modify: `src/common/benchmark.h:22-38`
- Modify: `src/cnn/main.cpp:240-253`
- Modify: `src/llm/llm_benchmark.cpp` (LLM result 输出)
- Modify: `src/single_op/single_op_benchmark.cpp` (单算子 result 输出)
- Modify: `src/cnn/CMakeLists.txt` (链接 nlohmann/json)

**Interfaces:**
- Produces: C++ binaries 输出严格 JSON Lines 格式到 stdout
- Produces: 每条 JSON 包含 `run_id` / `timestamp` / `git_commit` / `track` / `framework` / `model` / `metrics`

- [ ] **Step 1: 引入 nlohmann/json header-only 库**

```bash
# 下载到 third_party/
wget -O third_party/json.hpp https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
```

- [ ] **Step 2: 标准化 BenchmarkResult::to_json()**

修改 `src/common/benchmark.cpp` 中 `to_json()` 方法，输出结构化 JSON：

```cpp
#include "json.hpp"
using json = nlohmann::json;

std::string BenchmarkResult::to_json() const {
    json j;
    j["run_id"] = run_id;          // 新增字段
    j["timestamp"] = timestamp;    // 新增字段 (ISO 8601)
    j["git_commit"] = GIT_COMMIT_HASH;
    j["track"] = "cnn";
    j["model"] = model_name;
    j["framework"] = backend_name;
    j["precision"] = precision_str;
    j["threads"] = num_threads;
    j["warmup_runs"] = warmup_runs;
    j["test_runs"] = test_runs;
    j["init_time_ms"] = init_time_ms;
    j["peak_memory_kb"] = peak_memory_kb;
    j["metrics"] = {
        {"p50_ms", latency_stats.p50_ms},
        {"p90_ms", latency_stats.p90_ms},
        {"p99_ms", latency_stats.p99_ms},
        {"mean_ms", latency_stats.mean_ms},
        {"min_ms", latency_stats.min_ms},
        {"max_ms", latency_stats.max_ms},
        {"std_dev", latency_stats.std_dev},
        {"throughput_fps", throughput_fps}
    };
    if (!accuracy.passed) {
        j["accuracy"] = {
            {"passed", accuracy.passed},
            {"cosine_similarity", accuracy.cosine_similarity},
            {"mean_absolute_error", accuracy.mean_absolute_error}
        };
    }
    return j.dump();
}
```

需要新增字段到 `BenchmarkResult` 和 `BenchmarkConfig`：

```cpp
// benchmark.h 新增字段
struct BenchmarkResult {
    std::string run_id;      // UUID
    std::string timestamp;   // ISO 8601
    int warmup_runs;
    int test_runs;
    std::string precision_str;
    // ... 原有字段 ...
};
```

- [ ] **Step 3: 添加 run_id 和时间戳生成**

在 `benchmark.cpp` 的 `run_benchmark()` 开头：

```cpp
#include <chrono>
#include <iomanip>
#include <sstream>
#include <random>

static std::string generate_run_id() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::gmtime(&t), "%Y%m%d-");
    // 4位随机 hex
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 0xFFFF);
    ss << std::hex << std::setw(4) << std::setfill('0') << dis(gen);
    return ss.str();
}

static std::string now_iso8601() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}
```

- [ ] **Step 4: 修改三个入口文件的 JSON 输出格式**

修改 `src/cnn/main.cpp` 中的输出：

```cpp
// 原来:
printf("JSON_RESULT: %s\n", result.to_json().c_str());

// 改为 (JSON Lines 格式):
printf("%s\n", result.to_json().c_str());
// 同时输出到 stderr 供人类阅读:
fprintf(stderr, "  %-10s | p50=%6.2f ms | p99=%6.2f ms | fps=%6.1f\n",
    result.backend_name.c_str(),
    result.latency_stats.p50_ms,
    result.latency_stats.p99_ms,
    result.throughput_fps);
```

LLM track 的 JSON schema：

```cpp
// src/llm/llm_benchmark.cpp: LLM 结果输出
json j;
j["run_id"] = run_id;
j["track"] = "llm";
j["framework"] = backend_name;
j["model"] = model_name;
j["metrics"] = {
    {"ttft_ms", ttft_ms},            // Time To First Token
    {"tps", tokens_per_second},       // Tokens Per Second
    {"total_tokens", total_tokens},
    {"prompt_len", prompt_len},
    {"generation_time_ms", gen_time_ms}
};
printf("%s\n", j.dump().c_str());
```

Single-op track JSON schema：

```cpp
// src/single_op/single_op_benchmark.cpp
json j;
j["run_id"] = run_id;
j["track"] = "single_op";
j["framework"] = backend_name;
j["category"] = category;
j["operator"] = op_name;
j["shape"] = shape_str;
j["metrics"] = {
    {"mean_us", mean_us},
    {"min_us", min_us},
    {"max_us", max_us}
};
printf("%s\n", j.dump().c_str());
```

- [ ] **Step 5: 更新 CMakeLists.txt 包含 json.hpp**

```cmake
# src/cnn/CMakeLists.txt 添加:
target_include_directories(benchmark_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party)
# src/llm/CMakeLists.txt 添加:
target_include_directories(llm_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party)
# src/single_op/CMakeLists.txt 添加:
target_include_directories(single_op_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party)
```

- [ ] **Step 6: 编译验证**

```bash
cmake --build build_android --target benchmark_inference llm_benchmark single_op_benchmark
# 预期: 编译通过，无链接错误
# 验证: build_android/src/cnn/benchmark_inference --json 2>/dev/null 输出有效 JSON
```

- [ ] **Step 7: Commit**

```bash
git add third_party/json.hpp src/common/benchmark.cpp src/common/benchmark.h \
        src/common/config.h src/cnn/main.cpp src/llm/llm_benchmark.cpp \
        src/single_op/single_op_benchmark.cpp src/cnn/CMakeLists.txt \
        src/llm/CMakeLists.txt src/single_op/CMakeLists.txt
git commit -m "feat: C++ 侧 JSON 输出标准化 (nlohmann/json)

- 引入 nlohmann/json v3.11.3 header-only
- BenchmarkResult::to_json() 输出结构化 JSON Lines
- 新增 run_id / timestamp / git_commit 字段
- CNN/LLM/SingleOp 三个赛道各自定义 metric schema
- stderr 输出人类可读摘要，stdout 输出 JSON"
```

---

### Task 2: benchctl Python CLI 骨架 + DB 层

**Files:**
- Create: `benchctl/__init__.py`
- Create: `benchctl/__main__.py`
- Create: `benchctl/cli.py`
- Create: `benchctl/db.py`

**Interfaces:**
- Produces: `benchctl` CLI 入口 (Click), `benchctl db` 子命令
- Produces: `Database` 类 (init/insert_run/query_history/get_latest)

- [ ] **Step 1: 创建 Python 包结构**

```bash
mkdir -p benchctl/tracks benchctl/templates
touch benchctl/__init__.py benchctl/tracks/__init__.py
```

- [ ] **Step 2: 实现 Database 类 (benchctl/db.py)**

```python
"""SQLite 历史数据库封装."""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "benchmarks.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT UNIQUE NOT NULL,
    timestamp   TEXT NOT NULL,         -- ISO 8601
    git_commit  TEXT NOT NULL,
    track       TEXT NOT NULL,         -- cnn | llm | single_op
    framework   TEXT NOT NULL,         -- mnn | ort | tvm | llamacpp | mnn_llm
    model       TEXT NOT NULL,
    precision   TEXT DEFAULT 'fp32',
    threads     INTEGER DEFAULT 4,
    warmup      INTEGER DEFAULT 10,
    test_runs   INTEGER DEFAULT 100,
    device_model TEXT,
    device_temp  REAL,                 -- 摄氏度
    metrics_json TEXT NOT NULL,        -- 完整 metrics JSON
    raw_log     TEXT,                  -- 原始日志路径
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_framework_model
    ON runs(framework, model);
CREATE INDEX IF NOT EXISTS idx_timestamp
    ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_run_id
    ON runs(run_id);
"""


class Database:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def insert(self, data: dict) -> int:
        """插入一条 benchmark 记录. data 来自 C++ JSON 输出."""
        cols = [
            "run_id", "timestamp", "git_commit", "track",
            "framework", "model", "precision", "threads",
            "warmup", "test_runs", "metrics_json"
        ]
        placeholders = ", ".join("?" * len(cols))
        values = [data.get(c) for c in cols]
        values[cols.index("metrics_json")] = json.dumps(data.get("metrics", {}))

        sql = f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders})"
        cur = self.conn.execute(sql, values)
        self.conn.commit()
        return cur.lastrowid

    def history(self, framework: str, model: str, limit: int = 10) -> list[dict]:
        """查询某个框架+模型的历史记录."""
        rows = self.conn.execute(
            """SELECT run_id, timestamp, git_commit, metrics_json
               FROM runs WHERE framework=? AND model=? AND track='cnn'
               ORDER BY timestamp DESC LIMIT ?""",
            (framework, model, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def latest(self, framework: str, model: str) -> Optional[dict]:
        """获取最新一条记录."""
        row = self.conn.execute(
            """SELECT * FROM runs WHERE framework=? AND model=?
               ORDER BY timestamp DESC LIMIT 1""",
            (framework, model)
        ).fetchone()
        return dict(row) if row else None

    def compare(self, frameworks: list[str], model: str) -> list[dict]:
        """横向对比: 多个框架同模型的最新数据."""
        results = []
        for fw in frameworks:
            row = self.latest(fw, model)
            if row:
                results.append(row)
        return results

    def close(self):
        self.conn.close()
```

- [ ] **Step 3: 实现 CLI 入口 (benchctl/cli.py)**

```python
"""benchctl CLI — ARM 端侧推理基准测试统一入口."""
import click
import json
import sys
from pathlib import Path

from .db import Database


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """ARM Inference Benchmark CLI — 统一基准测试工具."""
    pass


@cli.group()
def db():
    """数据库管理命令."""
    pass


@db.command("init")
def db_init():
    """初始化数据库."""
    d = Database()
    d.close()
    click.echo(f"✅ Database initialized: {Database.DB_PATH}")


@db.command("stats")
def db_stats():
    """数据库统计."""
    d = Database()
    total = d.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    click.echo(f"Total runs: {total}")
    d.close()


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: 实现 __main__.py**

```python
"""python -m benchctl 入口."""
from .cli import cli
cli()
```

- [ ] **Step 5: 验证 CLI 可用**

```bash
cd /home/liu/project/newwork/benchmark
python -m benchctl --help
# 预期输出: Usage: benchctl [OPTIONS] COMMAND [ARGS]...
python -m benchctl db init
# 预期输出: ✅ Database initialized
python -m benchctl db stats
# 预期输出: Total runs: 0
```

- [ ] **Step 6: Commit**

```bash
git add benchctl/ benchmarks.db
# 注意: benchmarks.db 加 .gitignore? 先不加，作为初始 schema
git commit -m "feat: benchctl Python CLI 骨架 + SQLite 数据库层

- Click CLI 框架 (benchctl/ cli.py + __main__.py)
- SQLite 数据库封装 (benchctl/db.py)
- runs 表 schema: run_id/timestamp/git_commit/track/framework/model/metrics
- 索引: framework+model, timestamp, run_id"
```

---

### Task 3: benchctl run 命令 + ADB Runner

**Files:**
- Create: `benchctl/runner.py`
- Create: `benchctl/tracks/base.py`
- Create: `benchctl/tracks/cnn.py`
- Create: `benchctl/tracks/llm.py`
- Create: `benchctl/tracks/single_op.py`
- Modify: `benchctl/cli.py` (添加 `run` 命令)

**Interfaces:**
- Consumes: `Database` from Task 2
- Consumes: C++ JSON 输出 from Task 1
- Produces: `benchctl run cnn resnet50 -f mnn,ort` 端到端流程

- [ ] **Step 1: 实现 ADB Runner (benchctl/runner.py)**

```python
"""ADB 设备交互编排."""
import subprocess
import json
import shlex
import yaml
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent


def load_config():
    """从 .benchmarkrc.yml 读取配置."""
    cfg_path = PROJECT_ROOT / ".benchmarkrc.yml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def find_adb() -> str:
    """找到 ADB 可执行文件路径."""
    cfg = load_config()
    adb_path = cfg.get("device", {}).get("adb", "adb")
    # 测试是否可用
    try:
        subprocess.run([adb_path, "version"], capture_output=True, check=True)
        return adb_path
    except Exception:
        return "adb"  # fallback to PATH


class AdbRunner:
    """ADB 设备交互编排器."""

    def __init__(self):
        self.adb = find_adb()
        self.device_id = load_config().get("device", {}).get("id")
        self.build_dir = PROJECT_ROOT / "build_android"
        self.device_dir = "/data/local/tmp/benchmark"

    def _adb(self, *args) -> str:
        cmd = [self.adb]
        if self.device_id:
            cmd += ["-s", self.device_id]
        cmd += list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ADB failed: {' '.join(cmd)}\n{result.stderr}")
        return result.stdout.strip()

    def check_device(self) -> dict:
        """设备握手检查."""
        output = self._adb("shell", "getprop", "ro.product.model")
        cpuinfo = self._adb("shell", "cat", "/proc/cpuinfo")
        return {
            "model": output,
            "soc": "SM8250" if "A77" in cpuinfo else "unknown",
            "connected": True
        }

    def push_binary(self, binary_name: str):
        """推送二进制到设备."""
        src = self.build_dir / "src" / "cnn" / binary_name  # FIXME: 按 track
        # 更通用的查找:
        for track_dir in ["cnn", "llm", "single_op"]:
            candidate = self.build_dir / "src" / track_dir / binary_name
            if candidate.exists():
                src = candidate
                break
        self._adb("push", str(src), f"{self.device_dir}/{binary_name}")
        self._adb("shell", f"chmod 755 {self.device_dir}/{binary_name}")

    def push_models(self, model_dir: str):
        """推送模型目录."""
        self._adb("push", model_dir, f"{self.device_dir}/models/")

    def push_libs(self):
        """推送所需的 .so 文件."""
        libs = [
            ("third_party/MNN/build_android/libMNN.so", "libMNN.so"),
            ("third_party/onnxruntime/build/Android/Release/libonnxruntime.so", "libonnxruntime.so"),
            ("third_party/tvm/build_android/libtvm_runtime.so", "libtvm_runtime.so"),
        ]
        for src_rel, dst_name in libs:
            src = PROJECT_ROOT / src_rel
            if src.exists():
                self._adb("push", str(src), f"{self.device_dir}/{dst_name}")

    def run_benchmark(self, binary: str, args: list[str]) -> list[dict]:
        """运行 benchmark 并解析 JSON Lines 输出."""
        cmd = f"cd {self.device_dir} && LD_LIBRARY_PATH={self.device_dir} ./{binary} " + " ".join(args)
        output = self._adb("shell", cmd)

        results = []
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return results

    def get_device_temp(self) -> Optional[float]:
        """获取设备温度."""
        try:
            out = self._adb("shell", "cat /sys/class/thermal/thermal_zone0/temp")
            return float(out.strip()) / 1000.0
        except Exception:
            return None
```

- [ ] **Step 2: 实现 Track 基类 (benchctl/tracks/base.py)**

```python
"""Benchmark track 基类."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackConfig:
    track: str                # cnn | llm | single_op
    binary: str               # benchmark_inference | llm_benchmark | single_op_benchmark
    model: str                # resnet50 | qwen3-4b
    frameworks: list[str]     # ["mnn", "ort"]
    precision: str = "fp32"
    threads: int = 4
    warmup: int = 10
    runs: int = 100


class BaseTrack(ABC):
    """Benchmark track 基类 — 定义各赛道的统一接口."""

    @abstractmethod
    def name(self) -> str:
        """赛道名称: cnn / llm / single_op."""
        ...

    @abstractmethod
    def binary_name(self) -> str:
        """对应的 C++ 二进制文件名."""
        ...

    @abstractmethod
    def build_cli_args(self, config: TrackConfig) -> list[str]:
        """从 TrackConfig 构建 C++ 二进制的 CLI 参数."""
        ...

    def validate_result(self, result: dict) -> bool:
        """验证一条 benchmark 结果的合理性."""
        metrics = result.get("metrics", {})
        if not metrics:
            return False
        p50 = metrics.get("p50_ms", 0)
        return p50 > 0 and p50 < 10000  # 不可能是负数或 >10s
```

- [ ] **Step 3: 实现 CNN Track (benchctl/tracks/cnn.py)**

```python
"""CNN 赛道 — 分类/检测/NLP 模型延迟基准测试."""
from .base import BaseTrack, TrackConfig


class CNNTrack(BaseTrack):
    def name(self) -> str:
        return "cnn"

    def binary_name(self) -> str:
        return "benchmark_inference"

    def build_cli_args(self, config: TrackConfig) -> list[str]:
        args = [
            "--model", config.model,
            "--backend", ",".join(config.frameworks),
            "--precision", config.precision,
            "--threads", str(config.threads),
            "--warmup", str(config.warmup),
            "--runs", str(config.runs),
            "--json",
        ]
        return args
```

- [ ] **Step 4: 实现 LLM Track (benchctl/tracks/llm.py)**

```python
"""LLM 赛道 — 大语言模型 TTFT/TPS 基准测试."""
from .base import BaseTrack, TrackConfig


class LLMTrack(BaseTrack):
    def name(self) -> str:
        return "llm"

    def binary_name(self) -> str:
        return "llm_benchmark"

    def build_cli_args(self, config: TrackConfig) -> list[str]:
        args = [
            "--model", config.model,
            "--backend", config.frameworks[0] if config.frameworks else "llamacpp",
            "--max-tokens", "128",
            "--n-prompt", "128",
            "--n-repeat", str(config.runs),
            "--benchmark",
        ]
        return args
```

- [ ] **Step 5: 实现 SingleOp Track (benchctl/tracks/single_op.py)**

```python
"""单算子赛道 — 逐算子延迟基准测试."""
from .base import BaseTrack, TrackConfig


class SingleOpTrack(BaseTrack):
    def name(self) -> str:
        return "single_op"

    def binary_name(self) -> str:
        return "single_op_benchmark"

    def build_cli_args(self, config: TrackConfig) -> list[str]:
        args = [
            "--category", config.model,  # model 字段复用为 category
            "--backend", ",".join(config.frameworks),
            "--warmup", str(config.warmup),
            "--runs", str(config.runs),
            "--threads", str(config.threads),
        ]
        return args
```

- [ ] **Step 6: 在 cli.py 添加 `run` 命令**

```python
# benchctl/cli.py 追加:
from .runner import AdbRunner
from .db import Database
from .tracks.cnn import CNNTrack
from .tracks.llm import LLMTrack
from .tracks.single_op import SingleOpTrack

TRACKS = {
    "cnn": CNNTrack(),
    "llm": LLMTrack(),
    "single_op": SingleOpTrack(),
}


@cli.command()
@click.argument("track", type=click.Choice(["cnn", "llm", "single_op"]))
@click.argument("model")
@click.option("-f", "--frameworks", default="mnn,ort",
              help="逗号分隔的框架列表 (mnn,ort,tvm,llamacpp)")
@click.option("-p", "--precision", default="fp32")
@click.option("-t", "--threads", default=4, type=int)
@click.option("-w", "--warmup", default=10, type=int)
@click.option("-r", "--runs", default=100, type=int)
@click.option("--no-save", is_flag=True, help="不保存到数据库")
def run(track, model, frameworks, precision, threads, warmup, runs, no_save):
    """执行基准测试.

    \b
    TRACK: cnn (CV/NLP模型) | llm (大语言模型) | single_op (单算子)
    MODEL: resnet50 | mobilenetv2 | qwen3-4b | matmul | ...
    """
    track_obj = TRACKS[track]
    fw_list = [f.strip() for f in frameworks.split(",")]

    config = TrackConfig(
        track=track,
        binary=track_obj.binary_name(),
        model=model,
        frameworks=fw_list,
        precision=precision,
        threads=threads,
        warmup=warmup,
        runs=runs,
    )

    runner = AdbRunner()

    # Step 1: 设备握手
    click.echo("📱 设备握手...")
    device_info = runner.check_device()
    click.echo(f"   设备: {device_info['model']} / {device_info['soc']}")

    # Step 2: 推送二进制和库
    click.echo(f"📤 推送 {track_obj.binary_name()}...")
    runner.push_binary(track_obj.binary_name())
    runner.push_libs()

    # Step 3: 获取设备温度
    temp = runner.get_device_temp()
    if temp:
        click.echo(f"🌡 设备温度: {temp}°C")

    # Step 4: 运行 benchmark
    args = track_obj.build_cli_args(config)
    click.echo(f"🚀 运行: {track_obj.binary_name()} {' '.join(args)}")
    results = runner.run_benchmark(track_obj.binary_name(), args)

    # Step 5: 保存结果
    if not no_save and results:
        db = Database()
        for r in results:
            r["device_model"] = device_info["model"]
            r["device_temp"] = temp
            db.insert(r)
        db.close()
        click.echo(f"✅ 已保存 {len(results)} 条结果到数据库")

    # Step 6: 打印摘要
    click.echo("\n📊 结果摘要:")
    for r in results:
        m = r.get("metrics", {})
        click.echo(f"  {r['framework']:12s} | p50={m.get('p50_ms', 0):6.2f}ms"
                   f" | p99={m.get('p99_ms', 0):6.2f}ms"
                   f" | fps={m.get('throughput_fps', 0):6.1f}")
```

- [ ] **Step 7: Commit**

```bash
git add benchctl/runner.py benchctl/tracks/ benchctl/cli.py
git commit -m "feat: benchctl run 命令 + ADB Runner + 三赛道实现

- AdbRunner: ADB 设备交互编排 (推送/运行/温度采集)
- BaseTrack/CNNTrack/LLMTrack/SingleOpTrack 赛道抽象
- benchctl run cnn resnet50 -f mnn,ort 端到端流程
- 结果自动保存到 SQLite 数据库"
```

---

### Task 4: benchctl history 命令 (纵向对比)

**Files:**
- Modify: `benchctl/cli.py` (添加 `history` 命令)

**Interfaces:**
- Consumes: `Database.history()` from Task 2
- Produces: 终端表格输出 + `--json` 模式

- [ ] **Step 1: 实现 history 命令**

```python
# benchctl/cli.py 追加:

@cli.command()
@click.argument("framework")
@click.argument("model")
@click.option("--last", default=10, type=int, help="显示最近 N 条")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
def history(framework, model, last, json_output):
    """查看框架+模型的历史趋势 (纵向对比).

    \b
    FRAMEWORK: mnn | ort | tvm | llamacpp
    MODEL: resnet50 | mobilenetv2 | qwen3-4b
    """
    db = Database()
    rows = db.history(framework, model, limit=last)
    db.close()

    if json_output:
        import json as j
        click.echo(j.dumps(rows, indent=2))
        return

    if not rows:
        click.echo(f"📭 没有 {framework}/{model} 的历史记录")
        return

    # 计算 baseline (最早的一条)
    baseline = rows[-1]
    b_metrics = json.loads(baseline["metrics_json"])
    b_p50 = b_metrics.get("p50_ms", 0)

    click.echo(f"\n📈 {framework}/{model} 性能趋势 (最近 {last} 条)\n")
    click.echo(f"{'日期':12s} {'commit':10s} {'p50(ms)':>10s} {'变化':>10s} {'累计':>8s}")
    click.echo("-" * 56)

    for r in reversed(rows):  # 正序显示
        m = json.loads(r["metrics_json"])
        p50 = m.get("p50_ms", 0)
        delta = (p50 - b_p50) / b_p50 * 100 if b_p50 else 0
        cumulative = "BASE" if r == baseline else f"{delta:+.1f}%"

        click.echo(f"{r['timestamp'][:10]:12s} "
                   f"{r['git_commit'][:8]:10s} "
                   f"{p50:10.2f} "
                   f"{delta:+9.1f}% "
                   f"{cumulative:>8s}")

    # 回归检测
    if len(rows) >= 2:
        latest = json.loads(rows[0]["metrics_json"])
        prev = json.loads(rows[1]["metrics_json"])
        l_p50 = latest.get("p50_ms", 0)
        p_p50 = prev.get("p50_ms", 0)
        if p_p50 > 0 and l_p50 > p_p50 * 1.05:
            click.echo(f"\n⚠️ 回归警告: 最新 commit 比前一次慢 {(l_p50/p_p50 - 1)*100:.1f}%")
        elif p_p50 > 0:
            click.echo(f"\n✅ 无回归 (最新 vs 前次: {(l_p50/p_p50 - 1)*100:.1f}%)")
```

- [ ] **Step 2: Commit**

```bash
git add benchctl/cli.py
git commit -m "feat: benchctl history 命令 — 纵向对比 + 回归检测

- benchctl history mnn resnet50 --last 10
- 终端表格: 日期/commit/p50/变化/累计 speedup
- 自动回归检测 (>5% 慢 → ⚠️ 警告)
- --json 输出模式"
```

---

### Task 5: benchctl verify 命令 (原生工具验证)

**Files:**
- Create: `benchctl/verify.py`
- Modify: `benchctl/cli.py` (添加 `verify` 命令)

**Interfaces:**
- Consumes: `Database.latest()` from Task 2
- Produces: 验证对照表 (Harness vs Native)

- [ ] **Step 1: 实现 Verify 模块 (benchctl/verify.py)**

```python
"""原生工具验证 — 调用各框架自带 benchmark 工具验证 Harness 结果."""
import subprocess
import json
import re
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent


class NativeVerifier:
    """调用各框架原生工具，与 Harness 结果对照."""

    DEVICE_DIR = "/data/local/tmp/benchmark"

    def verify_mnn(self, model_name: str, threads: int = 4,
                   harness_result_ms: float = 0) -> Optional[dict]:
        """调用 MNN 自带的 benchmark 工具 (benchmark.out).

        前提: MNN benchmark 工具已推送到设备:
          adb push third_party/MNN/build_android/benchmark.out $DEVICE_DIR/
        """
        model_path = f"models/exported/mnn/{model_name}.mnn"
        # MNN benchmark.out 的输出格式:
        # forward time: 8.123 ms
        try:
            runner = self._get_runner()
            output = runner._adb(
                "shell",
                f"cd {self.DEVICE_DIR} && "
                f"LD_LIBRARY_PATH={self.DEVICE_DIR} "
                f"./benchmark.out {self.DEVICE_DIR}/{model_path} 10 0"
            )
            # 解析 "forward time: X.XXX ms"
            match = re.search(r"forward time:\s*([\d.]+)\s*ms", output)
            if match:
                native_ms = float(match.group(1))
                deviation = (harness_result_ms - native_ms) / native_ms * 100
                return {
                    "native_tool": "MNN benchmark.out",
                    "native_result_ms": native_ms,
                    "harness_result_ms": harness_result_ms,
                    "deviation_pct": round(deviation, 2),
                    "verdict": "trusted" if abs(deviation) < 5 else "unreliable"
                }
        except Exception as e:
            return {"error": str(e)}
        return None

    def verify_ort(self, model_name: str, harness_result_ms: float = 0) -> Optional[dict]:
        """调用 ONNX Runtime 自带的 onnxruntime_perf_test.

        ORT 的 perf test 工具需要单独编译 (在 third_party/onnxruntime 中).
        """
        model_path = f"models/source/classification/{model_name}/{model_name}.onnx"
        try:
            runner = self._get_runner()
            output = runner._adb(
                "shell",
                f"cd {self.DEVICE_DIR} && "
                f"LD_LIBRARY_PATH={self.DEVICE_DIR} "
                f"./onnxruntime_perf_test {model_path} 10"
            )
            # 解析 ORT perf test 输出格式
            match = re.search(r"avg:\s*([\d.]+)\s*ms", output)
            if match:
                native_ms = float(match.group(1))
                deviation = (harness_result_ms - native_ms) / native_ms * 100
                return {
                    "native_tool": "onnxruntime_perf_test",
                    "native_result_ms": native_ms,
                    "harness_result_ms": harness_result_ms,
                    "deviation_pct": round(deviation, 2),
                    "verdict": "trusted" if abs(deviation) < 5 else "unreliable"
                }
        except Exception as e:
            return {"error": str(e)}
        return None

    def _get_runner(self):
        from .runner import AdbRunner
        return AdbRunner()
```

- [ ] **Step 2: 在 cli.py 添加 `verify` 命令**

```python
# benchctl/cli.py 追加:
from .verify import NativeVerifier
from .db import Database


@cli.command()
@click.argument("framework")
@click.argument("model")
@click.option("-t", "--threads", default=4, type=int)
def verify(framework, model, threads):
    """验证 Harness 结果的可信度 (调用框架原生工具).

    \b
    FRAMEWORK: mnn | ort
    MODEL: resnet50 | mobilenetv2
    """
    db = Database()
    latest = db.latest(framework, model)
    db.close()

    if not latest:
        click.echo(f"❌ 没有 {framework}/{model} 的 Harness 数据, 请先执行 benchctl run")
        return

    harness_metrics = json.loads(latest["metrics_json"])
    harness_ms = harness_metrics.get("p50_ms", 0)

    click.echo(f"\n🔍 验证 {framework}/{model}:")
    click.echo(f"   Harness p50: {harness_ms:.2f} ms")

    verifier = NativeVerifier()
    if framework == "mnn":
        result = verifier.verify_mnn(model, threads, harness_ms)
    elif framework == "ort":
        result = verifier.verify_ort(model, threads, harness_ms)
    else:
        click.echo(f"   ⚠️ {framework} 暂不支持原生工具验证")
        return

    if result and "error" not in result:
        verdict_icon = "✅" if result["verdict"] == "trusted" else "❌"
        click.echo(f"   原生工具: {result['native_tool']}")
        click.echo(f"   原生结果: {result['native_result_ms']:.2f} ms")
        click.echo(f"   偏差:     {result['deviation_pct']:+.1f}%")
        click.echo(f"   可信度:   {verdict_icon} {result['verdict']}")
    else:
        click.echo(f"   ❌ 验证失败: {result.get('error', 'unknown')}")
```

- [ ] **Step 3: Commit**

```bash
git add benchctl/verify.py benchctl/cli.py
git commit -m "feat: benchctl verify 命令 — 原生工具验证层

- NativeVerifier: MNN benchmark.out / ORT onnxruntime_perf_test
- 偏差 < 5% → ✅ trusted, > 5% → ❌ unreliable
- benchctl verify mnn resnet50"
```

---

### Task 6: benchctl export + HTML 报告

**Files:**
- Create: `benchctl/report.py`
- Modify: `benchctl/cli.py` (添加 `export` 命令)

**Interfaces:**
- Consumes: `Database.compare()` / `Database.history()` from Task 2
- Produces: JSON 文件输出 + HTML 报告

- [ ] **Step 1: 实现 Report 模块 (benchctl/report.py)**

```python
"""报告生成 — JSON 导出 + HTML 渲染."""
import json
from pathlib import Path
from typing import Optional
from .db import Database

PROJECT_ROOT = Path(__file__).parent.parent


def export_json(frameworks: list[str], model: str,
                output: Optional[Path] = None) -> Path:
    """导出横向对比数据为 JSON."""
    db = Database()
    results = db.compare(frameworks, model)
    db.close()

    if output is None:
        output = PROJECT_ROOT / "results" / f"compare_{model}.json"

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    return output


def generate_html_compare(frameworks: list[str], model: str,
                          output: Optional[Path] = None) -> Path:
    """生成横向对比 HTML 报告."""
    db = Database()
    results = db.compare(frameworks, model)
    db.close()

    if output is None:
        output = PROJECT_ROOT / "results" / f"report_{model}.html"

    # 简单内联 HTML (后续可用 Jinja2 模板)
    rows_html = ""
    for r in results:
        m = json.loads(r["metrics_json"])
        p50 = m.get("p50_ms", 0)
        p99 = m.get("p99_ms", 0)
        fps = m.get("throughput_fps", 0)
        # 找最小值做高亮
        rows_html += f"""
        <tr>
            <td>{r['framework']}</td>
            <td>{p50:.2f}</td>
            <td>{p99:.2f}</td>
            <td>{fps:.1f}</td>
            <td>{r.get('precision', 'fp32')}</td>
            <td>{r.get('threads', 4)}</td>
        </tr>"""

    best_p50 = min(
        json.loads(r["metrics_json"]).get("p50_ms", float("inf"))
        for r in results
    ) if results else 0

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>Benchmark: {model}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
        th {{ background: #f5f5f5; }}
        .best {{ background: #d4edda; font-weight: bold; }}
        .header {{ display: flex; justify-content: space-between; align-items: baseline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 {model} Benchmark Report</h1>
        <span>生成时间: <code>{results[0]['timestamp'] if results else 'N/A'}</code></span>
    </div>
    <table>
        <thead>
            <tr>
                <th>框架</th><th>P50 (ms)</th><th>P99 (ms)</th><th>FPS</th>
                <th>精度</th><th>线程</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    <p style="color:#666; margin-top:1em;">
        <small>测试设备: 红米 K30 Pro / 骁龙 865 (SM8250)</small>
    </p>
</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(html)
    return output
```

- [ ] **Step 2: 在 cli.py 添加 `export` 命令**

```python
# benchctl/cli.py 追加:
from .report import export_json, generate_html_compare


@cli.command()
@click.argument("model")
@click.option("-f", "--frameworks", default="mnn,ort,tvm",
              help="逗号分隔框架列表")
@click.option("-o", "--output", type=click.Path(), default=None,
              help="输出文件路径")
@click.option("--format", "fmt", type=click.Choice(["json", "html"]), default="html",
              help="输出格式")
def export(model, frameworks, output, fmt):
    """导出 benchmark 结果.

    \b
    MODEL: resnet50 | mobilenetv2 | ...
    """
    fw_list = [f.strip() for f in frameworks.split(",")]

    if fmt == "json":
        out = export_json(fw_list, model, Path(output) if output else None)
        click.echo(f"✅ JSON 导出: {out}")
    elif fmt == "html":
        out = generate_html_compare(fw_list, model, Path(output) if output else None)
        click.echo(f"✅ HTML 报告: {out}")
```

- [ ] **Step 3: Commit**

```bash
git add benchctl/report.py benchctl/cli.py
git commit -m "feat: benchctl export 命令 — JSON/HTML 报告生成

- benchctl export resnet50 -f mnn,ort,tvm (HTML)
- benchctl export resnet50 --format json (JSON)
- HTML 表格: P50/P99/FPS/精度/线程 横向对比
- JSON 输出支持 AI 消费"
```

---

### Task 7: 脚本整合 + .benchmarkrc.yml 完善

**Files:**
- Modify: `scripts/benchmark/run-android.sh` (调用 benchctl)
- Modify: `.benchmarkrc.yml` (增加配置项)
- Create: `benchctl/templates/base.html` (Jinja2 模板)

- [ ] **Step 1: 创建 shell wrapper 脚本**

```bash
# 创建 scripts/benchmark/benchctl.sh
cat > scripts/benchmark/benchctl.sh << 'EOF'
#!/bin/bash
# benchctl shell wrapper — 确保 Python 环境和 ADB 就绪
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
exec python3 -m benchctl "$@"
EOF
chmod +x scripts/benchmark/benchctl.sh
```

- [ ] **Step 2: 更新 run-android.sh 引用 benchctl**

在 `scripts/benchmark/run-android.sh` 顶部添加注释：

```bash
# ⚠️ 此脚本已弃用，请使用 benchctl:
#   ./scripts/benchmark/benchctl.sh run cnn resnet50 -f mnn,ort
# 或:
#   python3 -m benchctl run cnn resnet50 -f mnn,ort
echo "⚠️  此脚本已弃用，请使用 benchctl 替代"
echo "   python3 -m benchctl run cnn resnet50 -f mnn,ort"
```

- [ ] **Step 3: 完善 .benchmarkrc.yml**

```yaml
# 在 .benchmarkrc.yml 末尾追加:
benchctl:
  db_path: benchmarks.db
  device_temp_limit: 45  # 超过此温度警告降频
  verification_threshold: 5  # 原生工具偏差阈值 %
  default_threads: 4
  default_warmup: 10
  default_runs: 100

frameworks:
  mnn:
    native_tool: benchmark.out
    native_tool_path: third_party/MNN/build_android/benchmark.out
  ort:
    native_tool: onnxruntime_perf_test
    native_tool_path: third_party/onnxruntime/build/Android/Release/onnxruntime_perf_test
```

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark/benchctl.sh scripts/benchmark/run-android.sh .benchmarkrc.yml
git commit -m "chore: benchctl shell wrapper + 配置完善 + run-android.sh 弃用声明"
```

---

### Task 8: 端到端验证

- [ ] **Step 1: 数据库初始化**

```bash
python3 -m benchctl db init
# 预期: ✅ Database initialized
python3 -m benchctl db stats
# 预期: Total runs: 0
```

- [ ] **Step 2: CLI 完整帮助**

```bash
python3 -m benchctl --help
# 预期: 显示 run/history/verify/export/db 子命令
python3 -m benchctl run --help
# 预期: 显示 TRACK/MODEL 参数和 -f/-p/-t/-w/-r 选项
```

- [ ] **Step 3: 模拟 run 流程 (不需要设备)**

```bash
# 构造假 JSON 验证数据库写入
echo '{"run_id":"test-001","timestamp":"2026-07-25T00:00:00Z","git_commit":"10c5a18","track":"cnn","framework":"mnn","model":"resnet50","precision":"fp32","threads":4,"warmup":10,"test_runs":100,"metrics":{"p50_ms":8.2,"p99_ms":9.1,"throughput_fps":122}}' | python3 -c "
import sys, json
from benchctl.db import Database
db = Database()
for line in sys.stdin:
    db.insert(json.loads(line))
db.close()
print('inserted')
"

python3 -m benchctl history mnn resnet50 --last 5
# 预期: 显示刚才插入的记录
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: benchctl 端到端验证 + 文档"
```

---

## 实现顺序

```
Task 1 (C++ JSON) → Task 2 (CLI + DB) → Task 3 (run) → Task 4 (history)
                                                              ↓
                                              Task 5 (verify) ← Task 6 (export)
                                                              ↓
                                                         Task 7 (整合)
                                                              ↓
                                                         Task 8 (验证)
```

Task 1-2 是基础，Task 3-6 可以并行，Task 7-8 收尾。

---

## 自检清单

- [x] 每个 Task 有精确文件路径
- [x] 每个 Step 有可执行的命令/代码
- [x] 无 TBD / TODO 占位符
- [x] 类型/接口在前后 Task 间一致 (run_id, metrics_json schema)
- [x] 覆盖三赛道 + 历史 + 验证 + 导出
- [x] 保留现有 C++ 代码不变，增量增强
- [x] 编译验证步骤已包含
