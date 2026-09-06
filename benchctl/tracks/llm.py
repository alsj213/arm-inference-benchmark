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
        # 精度对齐：显式指定 LLM 量化级别时透传 --require-precision（fp32 是 CNN 默认，
        # 对 LLM 无意义，不转发以避免误拒绝已量化的模型）
        LLM_LEVELS = {"f32", "f16", "q2", "q3", "q4", "q5", "q6", "q8", "iq"}
        if config.precision in LLM_LEVELS:
            args += ["--require-precision", config.precision]
        return args
