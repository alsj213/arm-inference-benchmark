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
            "warmup", "test_runs", "device_model", "device_temp",
            "metrics_json",
        ]
        placeholders = ", ".join("?" * len(cols))
        values = [data.get(c) for c in cols]
        # normalize framework/model to lowercase for case-insensitive queries
        fw_idx = cols.index("framework")
        values[fw_idx] = (data.get("framework") or "").lower()
        values[cols.index("metrics_json")] = json.dumps(data.get("metrics", {}))

        sql = f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders})"
        cur = self.conn.execute(sql, values)
        self.conn.commit()
        return cur.lastrowid

    def history(self, framework: str, model: str, limit: int = 10, track: str = None) -> list[dict]:
        """查询某个框架+模型的历史记录. track 为 None 时不过滤赛道."""
        if track:
            rows = self.conn.execute(
                """SELECT run_id, timestamp, git_commit, metrics_json
                   FROM runs WHERE framework=? AND model=? AND track=?
                   ORDER BY timestamp DESC LIMIT ?""",
                (framework, model, track, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT run_id, timestamp, git_commit, metrics_json
                   FROM runs WHERE framework=? AND model=?
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
