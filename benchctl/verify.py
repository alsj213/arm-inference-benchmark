"""原生工具验证 — 调用各框架自带 benchmark 工具验证 Harness 结果."""
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
        try:
            runner = self._get_runner()
            output = runner._adb(
                "shell",
                f"cd {self.DEVICE_DIR} && "
                f"LD_LIBRARY_PATH={self.DEVICE_DIR} "
                f"./benchmark.out {self.DEVICE_DIR}/{model_path} 10 0 {threads}"
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
