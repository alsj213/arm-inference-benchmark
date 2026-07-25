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
            "--backend",
            config.frameworks[0] if config.frameworks else "llamacpp",
            "--max-tokens", "128",
            "--n-prompt", "128",
            "--n-repeat", str(config.runs),
            "--benchmark",
            "--json",
        ]
        return args
