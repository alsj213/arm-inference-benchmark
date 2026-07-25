"""ADB 设备交互编排."""
import subprocess
import json
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

    def _adb(self, *args, timeout: int = 60) -> str:
        cmd = [self.adb]
        if self.device_id:
            cmd += ["-s", self.device_id]
        cmd += list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"ADB failed: {' '.join(cmd)}\n{result.stderr}"
            )
        return result.stdout.strip()

    def check_device(self) -> dict:
        """设备握手检查."""
        output = self._adb("shell", "getprop", "ro.product.model")
        cpuinfo = self._adb("shell", "cat", "/proc/cpuinfo")
        return {
            "model": output,
            "soc": "SM8250" if "A77" in cpuinfo else "unknown",
            "connected": True,
        }

    def push_binary(self, binary_name: str):
        """推送二进制到设备."""
        src = None
        for track_dir in ["cnn", "llm", "single_op"]:
            candidate = self.build_dir / "src" / track_dir / binary_name
            if candidate.exists():
                src = candidate
                break
        if src is None:
            raise FileNotFoundError(
                f"Binary '{binary_name}' not found under {self.build_dir / 'src'}"
            )
        self._adb("push", str(src), f"{self.device_dir}/{binary_name}")
        self._adb("shell", "chmod", "755", f"{self.device_dir}/{binary_name}")

    def push_models(self, model_dir: str):
        """推送模型目录."""
        self._adb("push", model_dir, f"{self.device_dir}/models/")

    def push_libs(self):
        """推送所需的 .so 文件."""
        libs = [
            (
                "third_party/MNN/build_android/libMNN.so",
                "libMNN.so",
            ),
            (
                "third_party/onnxruntime/build/Android/Release/libonnxruntime.so",
                "libonnxruntime.so",
            ),
            (
                "third_party/tvm/build_android/libtvm_runtime.so",
                "libtvm_runtime.so",
            ),
        ]
        for src_rel, dst_name in libs:
            src = PROJECT_ROOT / src_rel
            if src.exists():
                self._adb(
                    "push", str(src), f"{self.device_dir}/{dst_name}"
                )

    def run_benchmark(
        self, binary: str, args: list[str]
    ) -> list[dict]:
        """运行 benchmark 并解析 JSON Lines 输出."""
        cmd = (
            f"cd {self.device_dir}"
            f" && LD_LIBRARY_PATH={self.device_dir}"
            f" ./{binary} " + " ".join(args)
        )
        output = self._adb("shell", cmd, timeout=600)

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
            out = self._adb(
                "shell", "cat /sys/class/thermal/thermal_zone0/temp"
            )
            return float(out.strip()) / 1000.0
        except Exception:
            return None
