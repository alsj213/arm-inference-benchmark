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
        model_folder = f"{self.DEVICE_DIR}/models/exported/mnn"
        try:
            runner = self._get_runner()
            output = runner._adb(
                "shell",
                f"cd {self.DEVICE_DIR} && "
                f"LD_LIBRARY_PATH={self.DEVICE_DIR} "
                f"./benchmark.out {model_folder} 10 0 0 {threads}"
            )
            # 解析 "[ - ] mobilenetv2.mnn   max = X.XXX ms  min = X.XXX ms  avg = X.XXX ms"
            pattern = rf"{model_name}\.mnn\s+max\s*=\s*([\d.]+)\s*ms\s+min\s*=\s*([\d.]+)\s*ms\s+avg\s*=\s*([\d.]+)\s*ms"
            match = re.search(pattern, output)
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

    def verify_ort(self, model_name: str, threads: int = 4,
                   harness_result_ms: float = 0) -> Optional[dict]:
        """调用 ONNX Runtime 自带的 onnxruntime_perf_test.

        ORT 的 perf test 工具需要单独编译 (在 third_party/onnxruntime 中).
        """
        # Model path mapping: classification/ for most, detection/ for yolov8n, nlp/ for bert
        if model_name == "yolov8n":
            model_path = f"{self.DEVICE_DIR}/models/source/detection/{model_name}/{model_name}.onnx"
        elif model_name == "bert":
            model_path = f"{self.DEVICE_DIR}/models/source/nlp/{model_name}/{model_name}.onnx"
        else:
            model_path = f"{self.DEVICE_DIR}/models/source/classification/{model_name}/{model_name}.onnx"

        try:
            runner = self._get_runner()
            output = runner._adb(
                "shell",
                f"cd {self.DEVICE_DIR} && "
                f"LD_LIBRARY_PATH={self.DEVICE_DIR} "
                f"./onnxruntime_perf_test -I -r 50 -s -e cpu {model_path}",
                timeout=300
            )
            # 解析 ORT perf test 输出: "P50 Latency: X.XXXXXXX s"
            match = re.search(r"P50 Latency:\s*([\d.]+)\s*s", output)
            if match:
                native_ms = float(match.group(1)) * 1000  # s → ms
                deviation = (harness_result_ms - native_ms) / native_ms * 100
                return {
                    "native_tool": "onnxruntime_perf_test",
                    "native_result_ms": round(native_ms, 2),
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
