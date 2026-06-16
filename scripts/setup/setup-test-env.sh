#!/bin/bash
# 设置测试环境以确保硬件一致性（支持 root 和非 root 设备）

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

# 使用 adb 从 Windows (WSL2)
if [ -x "/mnt/e/andorid/adb/adb.exe" ]; then
    adb() {
        /mnt/e/andorid/adb/adb.exe "$@"
    }
fi

# 创建结果目录
mkdir -p "$PROJECT_ROOT/results"

echo "=== 设置测试环境 ==="

# 检测 root 权限
echo "检测设备 root 权限..."
ROOT_CHECK=$(adb shell "su -c 'echo root' 2>/dev/null" || echo "no_root")

if [ "$ROOT_CHECK" = "root" ]; then
    echo "✅ 检测到 root 权限，启用硬件控制..."

    # 1. 设置 CPU 调度器为 performance 模式
    echo "设置 CPU 调度器为 performance..."
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance | adb shell "su -c 'cat > $cpu'" 2>/dev/null || true
    done

    # 2. 记录初始 CPU 频率
    echo "记录初始 CPU 频率..."
    adb shell "su -c 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'" > "$PROJECT_ROOT/results/cpu_freq_before.txt" 2>/dev/null || echo "N/A" > "$PROJECT_ROOT/results/cpu_freq_before.txt"

    # 3. 清理系统缓存
    echo "清理系统缓存..."
    adb shell "su -c 'sync && echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null || echo "⚠️  无法清理缓存（权限不足）"

    # 4. 停止不必要的后台服务（可选）
    echo "停止后台服务..."
    adb shell "su -c 'stop zygote'" 2>/dev/null || echo "⚠️  无法停止服务（权限不足）"

    echo "✅ 硬件控制已启用"
else
    echo "⚠️  未检测到 root 权限，仅启用软件控制..."
fi

# 5. 设置进程优先级（不需要 root）
echo "设置进程优先级..."
if adb shell "nice -n -20 echo test" >/dev/null 2>&1; then
    echo "✅ nice 值设置可用"
else
    echo "⚠️  nice 值设置受限"
fi

# 6. 记录设备状态
echo "记录设备状态..."
adb shell "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" > "$PROJECT_ROOT/results/governor_before.txt" 2>/dev/null || echo "N/A" > "$PROJECT_ROOT/results/governor_before.txt"
adb shell "cat /sys/class/thermal/thermal_zone0/temp" > "$PROJECT_ROOT/results/temp_before.txt" 2>/dev/null || echo "N/A" > "$PROJECT_ROOT/results/temp_before.txt"

echo "=== 环境设置完成 ==="
echo "CPU 调度器: $(cat $PROJECT_ROOT/results/governor_before.txt 2>/dev/null || echo 'N/A')"
echo "CPU 频率: $(cat $PROJECT_ROOT/results/cpu_freq_before.txt 2>/dev/null || echo 'N/A') kHz"
echo "温度: $(cat $PROJECT_ROOT/results/temp_before.txt 2>/dev/null || echo 'N/A')"
