"""Benchmark track 基类."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TrackConfig:
    track: str  # cnn | llm | single_op
    binary: str  # benchmark_inference | llm_benchmark | single_op_benchmark
    model: str  # resnet50 | qwen3-4b
    frameworks: list[str]  # ["mnn", "ort"]
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
