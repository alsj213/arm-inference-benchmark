# Contributing to ARM Inference Benchmark

Thanks for your interest! This guide covers how to add a new device, framework, model, or fix a bug.

## Quick Links

- [Methodology](docs/METHODOLOGY.md)
- [Architecture](README.md#architecture)
- [Submodule Tags](README.md#submodule-tags)

---

## Adding a New Device

### Prerequisites

- ARM64 Android device (Android 10+)
- ADB debugging enabled
- Root access (optional, for CPU frequency lock)

### Steps

1. **Record device info:**

```bash
adb shell getprop ro.product.model              # e.g. "Redmi K30 Pro"
adb shell cat /proc/cpuinfo | grep -E "CPU|A77|A78|X1"
adb shell cat /sys/class/thermal/thermal_zone0/temp
```

2. **Add to benchmark device registry** (create `docs/devices/<device-name>.md`):

```markdown
# Device Name

| Property | Value |
|----------|-------|
| Model | Redmi K30 Pro |
| SoC | Snapdragon 865 (SM8250) |
| CPU | 1xA77@2.84GHz + 3xA77@2.42GHz + 4xA55@1.8GHz |
| ISA | ARMv8.2-A, FP16 |
```

3. **Run benchmark:**

```bash
export ANDROID_DEVICE_ID=<your-device-id>
benchctl run cnn mobilenetv2 -f mnn -t 4 -r 50
```

4. **Submit a PR** with your device doc + benchmark results.

---

## Adding a New Framework

1. **Implement `BenchmarkBackend` interface** in `src/backends/newfw_backend.cpp`:

```cpp
class NewFWBackend : public BenchmarkBackend {
public:
    bool init(const BenchmarkConfig& config) override;
    bool infer(const std::vector<float>& input) override;
    bool infer_with_output(const std::vector<float>& input, std::vector<float>& output) override;
    void deinit() override;
    std::string name() const override { return "newfw"; }
};
```

2. **Add CMake option** in `CMakeLists.txt`
3. **Update `third_party/CMakeLists.txt`** with submodule config
4. **Add model converter** under `scripts/convert/` if needed
5. **Submit a PR**

---

## Development Setup

### Build

```bash
export ANDROID_NDK=/path/to/android-ndk
cmake -B build_android \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
    -DBENCHMARK_MNN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_android -j$(nproc)
```

### Code Style

- C++: `clang-format --style=file` (`.clang-format` at repo root)
- Python: `ruff check`

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, all CI green |
| `feat/xxx` | Feature development |
| `fix/xxx` | Bug fixes |
| `backup/xxx` | Historical snapshots |

### Commit Convention

```
<type>: <description>
Types: feat, fix, refactor, docs, test, chore, perf, ci
```

---

## Dataset

Benchmark result datasets are published with each release:

```bash
benchctl export mobilenetv2 -f mnn,ort --format json
# → results/compare_mobilenetv2.json
```

For CI automation, see `.github/workflows/build.yml`.

---

Open an issue at [github.com/alsj213/arm-inference-benchmark/issues](https://github.com/alsj213/arm-inference-benchmark/issues).
