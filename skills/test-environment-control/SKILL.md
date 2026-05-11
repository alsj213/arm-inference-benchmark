---
name: test-environment-control
description: Use when running benchmarks or profiling to ensure consistent test conditions — lock CPU frequency, clear caches, restore environment after testing
---

# Test Environment Control

## Overview

在 Android 设备上保证测试环境一致性：锁 CPU 频率、清缓存，测试完成后恢复调度器。

## 注意

当前设备**未 root**，部分高级功能不可用。脚本会自动检测 root 状态并降级。

## 快速参考

| 操作 | 命令 | 需要 root |
|------|------|-----------|
| 设置测试环境 | `./scripts/setup_test_environment.sh` | 自动检测 |
| 恢复环境 | `./scripts/restore_test_environment.sh` | 自动检测 |

## 脚本行为

### setup_test_environment.sh（自动检测 root）

**有 root 权限时：**
- CPU 设为 performance governor
- 清理缓存（`echo 3 > /proc/sys/vm/drop_caches`）
- 记录初始 CPU 频率和温度
- 停止 zygote 后台服务

**无 root 权限时：**
- 跳过硬件控制，仅记录设备状态
- 设置进程优先级（nice）

### restore_test_environment.sh（自动检测 root）

**有 root 权限时：**
- CPU 恢复为 schedutil governor
- 重启 zygote 服务
- 记录恢复后的频率和温度

**无 root 权限时：**
- 仅记录设备状态

## 工作流

```bash
# 设置测试环境（自动检测 root）
./scripts/setup_test_environment.sh

# 运行测试...

# 恢复环境
./scripts/restore_test_environment.sh
```

## 常见问题

- **非 root 设备效果有限**: 无 root 无法设置 performance governor 或清理缓存，但脚本仍会记录设备状态供参考
- **测试前建议清缓存**: root 设备脚本自动处理；非 root 设备可手动重启
- **测试后必须恢复**: 避免 performance 模式导致发热/耗电（root 设备）