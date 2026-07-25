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
