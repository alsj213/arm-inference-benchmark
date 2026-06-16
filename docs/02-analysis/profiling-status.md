# Profiling Framework Status

## Current State

### 1. Debug/Release Variable Control ✅

The framework already supports Debug/Release variable control through the following scripts:

| Script | Flag | Default | Description |
|--------|------|---------|-------------|
| `build_android.sh` | `--debug` / `--release` | Release | Build with debug symbols or optimized |
| `run_benchmark_android.sh` | `--build-type <type>` | Release | Run benchmark with specified build type |
| `simpleperf_profile.sh` | `--build-type <type>` | Debug | Flame graph analysis (Debug for symbols) |
| `profile_benchmark.sh` | `--build-type <type>` | Release | Performance testing (Release for speed) |

**Usage Examples:**

```bash
# Build Debug version (for flame graph analysis)
./scripts/build_android.sh --debug

# Build Release version (for performance testing)
./scripts/build_android.sh --release

# Run benchmark with Release build
./scripts/run_benchmark_android.sh --build-type release --backend mnn --model mobilenetv2

# Run flame graph analysis with Debug build
./scripts/simpleperf_profile.sh --build-type debug --backend mnn --model mobilenetv2
```

### 2. Framework Profiling ✅

The `framework_profiling.sh` script supports ORT and MNN profiling:

#### MNN Profiling
- Uses environment variables: `MNN_PROFILING=1` and `MNN_PROFILING_FILE=<file>`
- Output: Layer-by-layer timing in text format
- Build type: Uses Release by default (can be changed with `--build-type`)

#### ONNX Runtime Profiling
- Uses the `--profiling` flag which calls `EnableProfiling()` API
- Output: JSON file viewable in chrome://tracing
- Build type: Uses Release by default (can be changed with `--build-type`)
- **Fixed Issue**: Removed `session_.release()` call in `deinit()` which was preventing profiling data from being written

### 3. ORT Profiling Fix ✅

**Issue 1**: ORT profiling JSON files were empty (0 bytes)

**Root Cause 1**: The `deinit()` function was calling `session_.release()` which:
1. Just returns the pointer and sets internal pointer to nullptr
2. Doesn't actually destroy the session
3. Prevents the destructor from being called
4. Profiling data is only written when the session is destroyed

**Fix 1**: Removed the `session_.release()` call from `deinit()`. The session destructor will be called automatically when the backend is destroyed, which will write the profiling data.

**Issue 2**: ORT profiling files not generated due to incorrect file prefix

**Root Cause 2**: ONNX Runtime's `EnableProfiling()` expects a file **prefix**, not a full filename. The actual file is named `{prefix}_{timestamp}.json`. Passing `profiling/ort_profile.json` as the prefix resulted in files named `profiling/ort_profile.json_20260501_123456.json`.

**Fix 2**: 
1. Updated `main.cpp` to strip `.json` extension from profiling file path
2. Updated `framework_profiling.sh` to use `profiling/ort_profile` as prefix (without `.json`)
3. Updated file retrieval logic to find the generated file with timestamp suffix

**Example**:
- Input prefix: `profiling/ort_profile`
- Generated file: `profiling/ort_profile_20260501_123456.json`

### 4. Profiling Workflow

#### For Performance Testing (Release Build)
```bash
# Build Release version
./scripts/build_android.sh --release

# Run benchmark with profiling
# Note: ORT expects a file prefix (without .json), it will add _timestamp.json automatically
./scripts/run_benchmark_android.sh --build-type release \
    --backend onnxrt --model mobilenetv2 --threads 4 \
    --profiling profiling/ort_profile
```

#### For Flame Graph Analysis (Debug Build)
```bash
# Build Debug version
./scripts/build_android.sh --debug

# Run simpleperf profiling
./scripts/simpleperf_profile.sh --build-type debug \
    --backend mnn --model mobilenetv2 --threads 4
```

#### For Framework Profiling (ORT/MNN)
```bash
# ORT profiling
./scripts/framework_profiling.sh --backend onnxrt --model mobilenetv2 --threads 4

# MNN profiling
./scripts/framework_profiling.sh --backend mnn --model mobilenetv2 --threads 4
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/build_android.sh` | Build Android binaries with Debug/Release support |
| `scripts/run_benchmark_android.sh` | Run benchmark with build type selection |
| `scripts/simpleperf_profile.sh` | CPU sampling analysis with flame graph generation |
| `scripts/profile_benchmark.sh` | Integrated benchmark + profiling |
| `scripts/framework_profiling.sh` | Framework-specific profiling (ORT/MNN) |
| `scripts/profiling_utils.sh` | Common profiling utilities |
| `src/backends/ort_backend.cpp` | ORT backend with profiling support |
| `src/backends/mnn_backend.cpp` | MNN backend with profiling support |

## Framework Requirements Verification ✅

### Requirement 1: Debug/Release Dynamic Library Variable Control ✅

**Status**: FULLY SUPPORTED

The framework supports Debug/Release variable control through multiple scripts:

| Script | Flag | Default | Purpose |
|--------|------|---------|---------|
| `build_android.sh` | `--debug` / `--release` | Release | Build with debug symbols or optimized |
| `run_benchmark_android.sh` | `--build-type <type>` | Release | Run benchmark with specified build type |
| `simpleperf_profile.sh` | `--build-type <type>` | Debug | Flame graph analysis (Debug for symbols) |
| `profile_benchmark.sh` | `--build-type <type>` | Release | Performance testing (Release for speed) |
| `framework_profiling.sh` | `--build-type <type>` | Release | Framework profiling (ORT/MNN) |

**Usage Examples**:

```bash
# Performance testing (Release build - no recompilation needed)
./scripts/run_benchmark_android.sh --build-type release --backend mnn --model mobilenetv2

# Flame graph analysis (Debug build - no recompilation needed)
./scripts/simpleperf_profile.sh --build-type debug --backend mnn --model mobilenetv2

# Framework profiling (ORT/MNN)
./scripts/framework_profiling.sh --backend onnxrt --model mobilenetv2
```

### Requirement 2: Operator-Level Profiling for ORT and MNN ✅

**Status**: FULLY SUPPORTED AND FIXED

**MNN Profiling**:
- Uses environment variables: `MNN_PROFILING=1` and `MNN_PROFILING_FILE=<file>`
- Output: Layer-by-layer timing in text format
- Command: `./scripts/framework_profiling.sh --backend mnn --model mobilenetv2`

**ORT Profiling**:
- Uses `EnableProfiling()` API with file prefix
- Output: JSON file viewable in chrome://tracing
- **Fixed Issues**:
  1. Removed `session_.release()` call in `deinit()` (prevented destructor from writing profiling data)
  2. Fixed file prefix handling (ORT expects prefix without `.json` extension)
- Command: `./scripts/framework_profiling.sh --backend onnxrt --model mobilenetv2`

### Requirement 3: No Recompilation Needed ✅

**Status**: FULLY SUPPORTED

- Debug and Release builds are stored in separate directories:
  - Release: `build_android/src/benchmark_inference`
  - Debug: `build_android_debug/src/benchmark_inference`
- Scripts automatically select the correct binary based on `--build-type` flag
- No recompilation needed when switching between Debug and Release

## Next Steps

1. **Test ORT profiling with the fix** to verify profiling data is now generated correctly
2. **Analyze the dumped operator timing data** from both MNN and ORT
3. **Compare profiling results** between Debug and Release builds
4. **Verify flame graph generation** with Debug build and simpleperf
