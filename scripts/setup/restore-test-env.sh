#!/bin/bash
# 恢复测试环境（支持 root 和非 root 设备）

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

# 使用 adb 从 Windows (WSL2)
if [ -n "${ADB:-}" ] && [ -x "$ADB" ]; then
    adb() {
        $ADB "$@"
    }
fi

echo "=== 恢复测试环境 ==="

# 检测 root 权限
ROOT_CHECK=$(adb shell "su -c 'echo root' 2>/dev/null" || echo "no_root")

if [ "$ROOT_CHECK" = "root" ]; then
    echo "✅ 检测到 root 权限，恢复硬件设置..."

    # 1. 恢复 CPU 调度器为默认模式
    echo "恢复 CPU 调度器..."
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo schedutil | adb shell "su -c 'cat > $cpu'" 2>/dev/null || true
    done

    # 2. 记录恢复后的 CPU 频率
    echo "记录恢复后 CPU 频率..."
    adb shell "su -c 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'" > "$PROJECT_ROOT/results/cpu_freq_after.txt" 2>/dev/null || echo "N/A" > "$PROJECT_ROOT/results/cpu_freq_after.txt"

    # 3. 重启服务
    echo "重启服务..."
    adb shell "su -c 'start zygote'" 2>/dev/null || echo "⚠️  无法重启服务（权限不足）"

    echo "✅ 硬件设置已恢复"
else
    echo "⚠️  未检测到 root 权限，跳过硬件恢复..."
fi

# 4. 记录恢复后的设备状态
echo "记录恢复后设备状态..."
adb shell "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" > "$PROJECT_ROOT/results/governor_after.txt" 2>/dev/null || echo "N/A" > "$PROJECT_ROOT/results/governor_after.txt"
adb shell "cat /sys/class/thermal/thermal_zone0/temp" > "$PROJECT_ROOT/results/temp_after.txt" 2>/dev/null || echo "N/A" > "$PROJECT_ROOT/results/temp_after.txt"

echo "=== 环境恢复完成 ==="
echo "CPU 调度器: $(cat $PROJECT_ROOT/results/governor_after.txt 2>/dev/null || echo 'N/A')"
echo "CPU 频率: $(cat $PROJECT_ROOT/results/cpu_freq_after.txt 2>/dev/null || echo 'N/A') kHz"
echo "温度: $(cat $PROJECT_ROOT/results/temp_after.txt 2>/dev/null || echo 'N/A')"
