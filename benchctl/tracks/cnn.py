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
