---
name: android-device-ops
description: Use when needing to interact with Android test device — check ADB connection, push binaries/libraries/models, execute remote commands, pull test results
---

# Android Device Ops

## Overview

管理 Android 测试设备的 ADB 通信操作。测试设备为红米 K30 Pro（骁龙 865），通过 WSL2 连接。

## 环境

- ADB: `/mnt/e/andorid/adb/adb.exe`
- 设备 ID: `b08dee23`
- 别名: `alias adb='/mnt/e/andorid/adb/adb.exe'`

## 快速参考

| 操作 | 命令 |
|------|------|
| 检查连接 | `adb devices` |
| 推送二进制 | `adb push build_android/src/benchmark_inference /data/local/tmp/` |
| 推送 ORT so | `adb push third_party/onnxruntime/build/Android/Release/libonnxruntime.so /data/local/tmp/` |
| 推送模型 | `adb push models /data/local/tmp/` |
| 远程执行 | `adb shell "cd /data/local/tmp && ./benchmark_inference --help"` |
| 拉取结果 | `adb pull /data/local/tmp/results.txt ./results/` |
| 设置权限 | `adb shell chmod +x /data/local/tmp/benchmark_inference` |

## 常用工作流

推送并运行：

```bash
adb shell mkdir -p /data/local/tmp/models
adb push build_android/src/benchmark_inference /data/local/tmp/
adb shell chmod +x /data/local/tmp/benchmark_inference
adb push third_party/onnxruntime/build/Android/Release/libonnxruntime.so /data/local/tmp/
adb push models/classification/mobilenetv2 /data/local/tmp/models/
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./benchmark_inference --model mobilenetv2"
```

## 常见问题

- **device not found**: 检查 `adb devices`，确认设备 ID `b08dee23` 在列表中
- **permission denied**: 推送后 `chmod +x`
- **ORT so 找不到**: 设置 `LD_LIBRARY_PATH=.`
- **空间不足**: `adb shell rm -rf /data/local/tmp/*`