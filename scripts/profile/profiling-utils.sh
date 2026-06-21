#!/bin/bash
# profiling_utils.sh - 性能分析公共工具库
# 供 simpleperf_profile.sh, atrace_capture.sh, perfetto_trace.sh 等脚本使用

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# 检测 ADB 路径（兼容 WSL2）
detect_adb() {
    # 优先使用环境变量
    if [ -n "$ADB_PATH" ]; then
        ADB_CMD="$ADB_PATH"
        return 0
    fi

    # WSL2 环境：Windows ADB
    if [ -f "/mnt/e/andorid/adb/adb.exe" ]; then
        ADB_CMD="/mnt/e/andorid/adb/adb.exe"
        return 0
    fi

    # 检查 PATH 中的 adb
    if command -v adb &> /dev/null; then
        ADB_CMD="adb"
        return 0
    fi

    log_error "ADB not found. Set ADB_PATH or add adb to PATH"
    return 1
}

# 统一 ADB 命令包装
adb_cmd() {
    $ADB_CMD "$@"
}

# 检查设备连接
check_device() {
    local devices
    devices=$(adb_cmd devices 2>/dev/null | grep -w "device" | head -1)
    if [ -z "$devices" ]; then
        log_error "No Android device connected"
        return 1
    fi
    DEVICE_SERIAL=$(echo "$devices" | awk '{print $1}')
    log_info "Device connected: $DEVICE_SERIAL"
    return 0
}

# 获取设备信息
get_device_info() {
    local output_file="${1:-device_info.txt}"

    echo "=== Device Info ===" > "$output_file"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')" >> "$output_file"
    echo "" >> "$output_file"

    # 设备型号
    echo "Model: $(adb_cmd shell getprop ro.product.model 2>/dev/null | tr -d '\r')" >> "$output_file"
    echo "Android: $(adb_cmd shell getprop ro.build.version.release 2>/dev/null | tr -d '\r')" >> "$output_file"
    echo "SDK: $(adb_cmd shell getprop ro.build.version.sdk 2>/dev/null | tr -d '\r')" >> "$output_file"
    echo "ABI: $(adb_cmd shell getprop ro.product.cpu.abi 2>/dev/null | tr -d '\r')" >> "$output_file"
    echo "" >> "$output_file"

    # CPU 信息
    echo "=== CPU Info ===" >> "$output_file"
    adb_cmd shell cat /proc/cpuinfo 2>/dev/null | head -30 >> "$output_file"
    echo "" >> "$output_file"

    # CPU 频率
    echo "=== CPU Frequencies ===" >> "$output_file"
    for i in 0 1 2 3 4 5 6 7; do
        local freq=$(adb_cmd shell cat /sys/devices/system/cpu/cpu${i}/cpufreq/scaling_cur_freq 2>/dev/null | tr -d '\r')
        if [ -n "$freq" ]; then
            echo "CPU${i}: ${freq} KHz" >> "$output_file"
        fi
    done
    echo "" >> "$output_file"

    # 温度
    echo "=== Thermal Zones ===" >> "$output_file"
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        local temp=$(adb_cmd shell cat "$zone" 2>/dev/null | tr -d '\r')
        if [ -n "$temp" ] && [ "$temp" -gt 0 ] 2>/dev/null; then
            local zone_name=$(basename $(dirname "$zone"))
            echo "${zone_name}: $((temp / 1000))°C" >> "$output_file"
        fi
    done

    log_info "Device info saved to $output_file"
}

# 检查 root 状态
check_root() {
    local id_output
    id_output=$(adb_cmd shell id 2>/dev/null | tr -d '\r')
    if echo "$id_output" | grep -q "uid=0"; then
        IS_ROOT=true
        log_info "Device is rooted"
    else
        IS_ROOT=false
        log_warn "Device is not rooted (some profiling features may be limited)"
    fi
}

# 推送 simpleperf 到设备
push_simpleperf() {
    local device_path="/data/local/tmp/simpleperf"

    # 检查设备上是否已有 simpleperf
    if adb_cmd shell "[ -f $device_path ]" 2>/dev/null; then
        log_info "simpleperf already exists on device"
        return 0
    fi

    # 从 NDK 查找 simpleperf
    if [ -z "$ANDROID_NDK" ]; then
        log_error "ANDROID_NDK not set, cannot push simpleperf"
        return 1
    fi

    # NDK r25+ 路径结构
    local simpleperf_path="$ANDROID_NDK/simpleperf/bin/android/arm64/simpleperf"
    if [ ! -f "$simpleperf_path" ]; then
        # 旧版 NDK 路径
        simpleperf_path="$ANDROID_NDK/simpleperf/arm64/simpleperf"
    fi
    if [ ! -f "$simpleperf_path" ]; then
        log_error "simpleperf not found at $simpleperf_path"
        return 1
    fi

    log_info "Pushing simpleperf to device..."
    adb_cmd push "$simpleperf_path" "$device_path"
    adb_cmd shell chmod 755 "$device_path"
    log_info "simpleperf pushed successfully"
}

# 创建结果目录
create_result_dir() {
    local model="${1:-unknown}"
    local backend="${2:-unknown}"
    local base_dir="results/profiling"
    local enabled_tools="${3:-}"  # 可选：启用的工具列表，逗号分隔

    local timestamp=$(date '+%Y%m%d_%H%M%S')
    RESULT_DIR="${base_dir}/${timestamp}_${model}_${backend}"

    # 如果指定了启用的工具，只创建对应的目录
    if [ -n "$enabled_tools" ]; then
        IFS=',' read -ra tools <<< "$enabled_tools"
        for tool in "${tools[@]}"; do
            case "$tool" in
                simpleperf) mkdir -p "$RESULT_DIR/simpleperf" ;;
                atrace) mkdir -p "$RESULT_DIR/atrace" ;;
                perfetto) mkdir -p "$RESULT_DIR/perfetto" ;;
                framework) mkdir -p "$RESULT_DIR/framework" ;;
                all)
                    mkdir -p "$RESULT_DIR/simpleperf"
                    mkdir -p "$RESULT_DIR/atrace"
                    mkdir -p "$RESULT_DIR/perfetto"
                    mkdir -p "$RESULT_DIR/framework"
                    ;;
            esac
        done
    else
        # 默认创建所有目录
        mkdir -p "$RESULT_DIR/simpleperf"
        mkdir -p "$RESULT_DIR/atrace"
        mkdir -p "$RESULT_DIR/perfetto"
        mkdir -p "$RESULT_DIR/framework"
    fi

    log_info "Result directory: $RESULT_DIR"
}

# 推送 benchmark 二进制和模型
push_benchmark_files() {
    local device_dir="/data/local/tmp/benchmark"
    local binary_path="${1:-build_android/src/cnn/benchmark_inference}"
    local models_path="${2:-models}"
    local build_type="${3:-Release}"  # 新增参数：Release 或 Debug

    # 根据构建类型选择正确的二进制路径
    if [ "$build_type" = "Debug" ]; then
        binary_path="${1:-build_android_debug/src/cnn/benchmark_inference}"
    else
        binary_path="${1:-build_android/src/cnn/benchmark_inference}"
    fi

    log_info "Using build type: $build_type"
    log_info "Binary path: $binary_path"

    # 创建设备目录
    adb_cmd shell "mkdir -p $device_dir/models/classification $device_dir/models/detection"

    # 检查二进制是否需要更新
    local need_push_binary=true
    if adb_cmd shell "[ -f $device_dir/benchmark_inference ]" 2>/dev/null; then
        local remote_size=$(adb_cmd shell "stat -c%s $device_dir/benchmark_inference" 2>/dev/null | tr -d '\r')
        local local_size=$(stat -c%s "$binary_path" 2>/dev/null)
        if [ "$remote_size" = "$local_size" ]; then
            need_push_binary=false
            log_info "Binary already up to date, skipping push"
        fi
    fi

    # 推送二进制
    if [ "$need_push_binary" = true ] && [ -f "$binary_path" ]; then
        log_info "Pushing benchmark binary..."
        adb_cmd push "$binary_path" "$device_dir/"
        adb_cmd shell chmod 755 "$device_dir/benchmark_inference"
    elif [ ! -f "$binary_path" ]; then
        log_error "Binary not found: $binary_path"
        log_error "Please run ./scripts/build_android.sh --$(echo "$build_type" | tr '[:upper:]' '[:lower:]') first"
        return 1
    fi

    # 检查模型是否需要更新
    local need_push_models=true
    if adb_cmd shell "ls $device_dir/models/classification/mobilenetv2/mobilenetv2.onnx" &>/dev/null; then
        need_push_models=false
        log_info "Models already exist, skipping push"
    fi

    # 推送模型
    if [ "$need_push_models" = true ] && [ -d "$models_path" ]; then
        log_info "Pushing models..."
        adb_cmd push "$models_path/." "$device_dir/models/"
    fi

    # 推送第三方库（检查是否存在）
    for lib in third_party/*/lib-android/aarch64/*.so; do
        if [ -f "$lib" ]; then
            local lib_name=$(basename "$lib")
            if ! adb_cmd shell "[ -f $device_dir/$lib_name ]" 2>/dev/null; then
                adb_cmd push "$lib" "$device_dir/"
            fi
        fi
    done
}

# WSL 路径转 Windows 路径
wsl_to_win_path() {
    local wsl_path="$1"
    # /mnt/e/... -> E:\...
    echo "$wsl_path" | sed 's|^/mnt/\([a-z]\)/|\U\1:\\|' | sed 's|/|\\|g'
}
