# Strip 符号指南 — 什么是 Strip、为什么要 Strip、如何操作

---

## 1. 什么是 Strip？

**Strip**（剥离）就是从 binary 文件中**移除符号表和调试信息**，让文件变小。

```
Strip 前：
┌─────────────────────────────────────┐
│  机器码（CPU 执行的指令）              │  ← 保留
│  符号表（函数名 → 地址的映射）          │  ← 移除
│  调试信息（源码行号、变量名等）          │  ← 移除
│  .eh_frame（调用栈展开信息）           │  ← 可选移除
└─────────────────────────────────────┘

Strip 后：
┌─────────────────────────────────────┐
│  机器码（CPU 执行的指令）              │  ← 保留
└─────────────────────────────────────┘
```

### Strip 的效果

| 项目 | Strip 前 | Strip 后 | 减少 |
|------|----------|----------|------|
| 文件大小 | 50 MB | 5 MB | 90% |
| 函数名 | 有 | 无（变成地址） | - |
| 调试信息 | 有 | 无 | - |
| 能否性能分析 | 能 | 不能（没有符号） | - |
| 能否调试 | 能 | 不能 | - |

---

## 2. 什么时候要 Strip？什么时候不要？

| 场景 | 是否 Strip | 原因 |
|------|-----------|------|
| **生产发布** | 要 Strip | 减小体积、保护代码 |
| **性能分析** | **不要 Strip** | 需要符号来生成火焰图 |
| **调试 bug** | **不要 Strip** | 需要调试信息来定位问题 |
| **CI/CD 测试** | 看情况 | 如果需要分析性能，不要 Strip |

---

## 3. 如何 Strip

### 3.1 使用 `strip` 命令

```bash
# Linux / macOS
strip your_binary                    # strip ELF binary
strip -x your_binary.dylib           # strip macOS dylib

# Android NDK 提供的交叉编译 strip
$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip your_binary

# 或者用 aarch64-linux-android-strip
$ANDROID_NDK/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-strip your_binary
```

### 3.2 使用 CMake 控制 Strip

```cmake
# 方法 1：编译后自动 strip（Release 模式默认行为）
set(CMAKE_BUILD_TYPE Release)

# 方法 2：手动控制 strip
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")      # -s 表示 strip
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

# 方法 3：禁止 strip（用于性能分析）
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_STRIP "")                # 设为空，不执行 strip
```

### 3.3 使用 Android Gradle 控制

```groovy
// build.gradle
android {
    buildTypes {
        release {
            // 启用 strip（默认）
            minifyEnabled true
            shrinkResources true

            // 或者禁用 strip
            // packagingOptions {
            //     doNotStrip '**/*.so'
            // }
        }
        debug {
            // 禁用 strip（默认）
            minifyEnabled false
        }
    }
}
```

### 3.4 手动 strip .so 文件

```bash
# strip 前
ls -lh libonnxruntime.so
# -rw-r--r-- 1 user user 14.6M Apr 27 20:26 libonnxruntime.so

# 执行 strip
$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip libonnxruntime.so

# strip 后
ls -lh libonnxruntime.so
# -rw-r--r-- 1 user user 5.2M Apr 27 20:26 libonnxruntime.so
# 大小从 14.6M 减少到 5.2M
```

---

## 4. 如何检查是否被 Strip

### 4.1 使用 `file` 命令

```bash
# Linux
file your_binary
# 输出: ELF 64-bit LSB executable, ... not stripped     ← 有符号
# 输出: ELF 64-bit LSB executable, ... stripped          ← 已 strip

# Android
adb shell "file /data/local/tmp/benchmark/benchmark_inference"
```

### 4.2 使用 `nm` 命令

```bash
# 查看符号表
nm -D your_binary | head -5
# 如果输出函数名，说明有符号表
# 如果报 "no symbols"，说明被 strip 了

# Android
adb shell "nm -D /data/local/tmp/benchmark/libonnxruntime.so"
```

### 4.3 使用 `readelf` 命令

```bash
# 查看 section 列表
readelf -S your_binary | grep -E "symtab|debug|strtab"

# 有符号的情况：
#   [Nr] Name              Type             Address           Offset
#   [25] .symtab           SYMTAB           0000000000000000  000313664
#   [26] .strtab           STRTAB           0000000000000000  000731834
#   [28] .debug_loc        PROGBITS         0000000000000000  000313664

# 已 strip 的情况（没有 .symtab 和 .debug_* 段）：
#   只有 .dynsym（动态符号表）
```

### 4.4 使用 `objdump` 命令

```bash
# 查看调试信息
objdump --debugging your_binary | head -20
# 如果有输出，说明有调试信息
# 如果报 "no debugging information"，说明已 strip
```

---

## 5. Strip 的层次

Strip 不是一刀切，可以分层次：

```
┌─────────────────────────────────────────────┐
│  Level 0: 完整（不 strip）                    │
│  包含：机器码 + 符号表 + 调试信息 + .eh_frame   │
│  用途：调试、性能分析                          │
│  大小：最大                                   │
├─────────────────────────────────────────────┤
│  Level 1: strip --strip-debug                │
│  包含：机器码 + 符号表 + .eh_frame             │
│  移除：调试信息（.debug_* 段）                 │
│  用途：性能分析（不需要源码行号）               │
│  大次：较大                                   │
├─────────────────────────────────────────────┤
│  Level 2: strip（默认）                       │
│  包含：机器码 + 动态符号表（.dynsym）           │
│  移除：符号表 + 调试信息                       │
│  用途：生产部署                               │
│  大小：较小                                   │
├─────────────────────────────────────────────┤
│  Level 3: strip --strip-all                  │
│  包含：仅机器码                               │
│  移除：所有符号和调试信息                       │
│  用途：最小体积（可能破坏动态链接）              │
│  大小：最小                                   │
└─────────────────────────────────────────────┘
```

### 各层次命令

```bash
# Level 0: 不 strip（保留一切）
# 什么都不做

# Level 1: 只移除调试信息（保留符号表）
strip --strip-debug your_binary
# 或
strip -g your_binary

# Level 2: 移除符号表和调试信息（保留动态符号表）
strip your_binary
# 或
strip --strip-unneeded your_binary

# Level 3: 移除所有（包括动态符号表，可能破坏 .so）
strip --strip-all your_binary
# 危险！可能导致 .so 无法加载
```

---

## 6. 性能分析时的 Strip 策略

### 最佳实践

```
开发/调试阶段：
├── 编译选项：RelWithDebInfo
├── Strip：不 strip
├── 目的：可以随时 profile 和 debug
└── 文件：较大，但功能完整

性能分析阶段：
├── 编译选项：RelWithDebInfo + -fno-omit-frame-pointer
├── Strip：不 strip（或只 strip --strip-debug）
├── 目的：生成火焰图、分析热点
└── 文件：较大，但有完整符号

生产发布阶段：
├── 编译选项：Release
├── Strip：strip（Level 2）
├── 目的：最小体积、保护代码
└── 文件：最小
```

### 分离符号文件

如果不想在生产 binary 中保留符号，可以**分离符号文件**：

```bash
# 1. 编译带符号的版本
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make

# 2. 复制一份带符号的（用于 profiling）
cp libyour.so libyour_with_symbols.so

# 3. strip 生产版本
strip libyour.so

# 4. 部署 strip 后的版本到设备
adb push libyour.so /data/local/tmp/

# 5. 做性能分析时，用带符号的版本
# simpleperf 可以自动匹配符号文件
simpleperf report -i perf.data --symdir ./symbols/
```

---

## 7. 常见问题

### Q1: strip 后还能做性能分析吗？

**不能**（没有函数名）。但可以用 `--symdir` 指定符号文件：

```bash
# 在设备上采集（用 strip 后的 binary）
simpleperf record -p PID --duration 10 -o perf.data

# 在电脑上生成报告（用带符号的 binary）
simpleperf report -i perf.data --symdir ./symbols/
```

### Q2: strip 会影响性能吗？

**不会**。Strip 只移除元数据（符号、调试信息），不影响机器码。程序运行速度完全一样。

### Q3: strip 会破坏 .so 的动态链接吗？

**可能会**（如果用 `--strip-all`）。推荐用 `strip`（默认）或 `--strip-unneeded`，保留 `.dynsym` 动态符号表。

### Q4: 如何只 strip 某些 .so？

```bash
# 只 strip 特定文件
strip libonnxruntime.so
strip libtensorflowlite_jni.so

# 或者在 CMake 中指定
set_target_properties(your_lib PROPERTIES
    CMAKE_STRIP_COMMAND "${CMAKE_STRIP} --strip-unneeded"
)
```

---

## 8. 快速参考

| 命令 | 作用 | 适用场景 |
|------|------|----------|
| `strip binary` | 移除符号表和调试信息 | 生产发布 |
| `strip -g binary` | 只移除调试信息 | 保留符号做 profiling |
| `strip --strip-all binary` | 移除所有（危险） | 最小体积 |
| `file binary` | 检查是否 strip | 验证 |
| `nm -D binary` | 查看动态符号 | 验证 |
| `readelf -S binary` | 查看所有 section | 深入检查 |
| `cmake -DCMAKE_BUILD_TYPE=Release` | 自动 strip | 编译时控制 |
| `cmake -DCMAKE_STRIP=""` | 禁止 strip | 性能分析 |
