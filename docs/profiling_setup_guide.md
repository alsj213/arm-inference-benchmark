# 火焰图采集准备教程 — 从编译到出图

---

## 1. 总览：火焰图采集的完整流程

```
① 编译准备（带符号、正确的编译选项）
        ↓
② 推送到设备（binary + simpleperf + 模型）
        ↓
③ 启动推理 + 采集数据（simpleperf record）
        ↓
④ 拉取数据 + 生成报告（report-sample）
        ↓
⑤ 生成火焰图（FlameGraph 工具）
        ↓
⑥ 用 SpeedScope 查看分析
```

**关键点**：火焰图的质量**完全取决于第①步**。如果编译时没有正确的符号和调试信息，后面再怎么努力都白费。

---

## 2. 编译准备（最重要！）

### 2.1 三种编译模式对比

| 编译模式 | 符号表 | 调试信息 | Frame Pointer | 适用场景 |
|----------|--------|----------|---------------|----------|
| `Release` | 可能被 strip | 无 | 默认省略 | 生产环境 |
| `RelWithDebInfo` | 保留 | 有 | 默认省略 | **推荐：性能分析** |
| `Debug` | 保留 | 完整 | 有 | 调试 bug |

**性能分析推荐用 `RelWithDebInfo`**：保留符号和调试信息，但仍有优化，不影响性能表现。

### 2.2 CMake 编译选项

```cmake
# 推荐的性能分析编译配置
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# 强制保留 frame pointer（ARM64 特别重要！）
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")

# 保留完整符号表（不要 strip）
set(CMAKE_STRIP "")
```

### 2.3 各编译选项的作用

#### `-fno-omit-frame-pointer`（关键！）

**问题**：ARM64 编译器默认 `-fomit-frame-pointer`，不保留帧指针寄存器（x29）。这导致 simpleperf 无法正确展开调用栈。

**解决**：加 `-fno-omit-frame-pointer`，让编译器在每个函数入口保存 x29 寄存器。

```bash
# 验证是否生效
adb shell "readelf -S your_binary | grep eh_frame"
# 应该有 .eh_frame 段

# 验证 frame pointer
adb shell "objdump -d your_binary | grep 'stp.*x29'"
# 应该能看到保存 frame pointer 的指令
```

#### 符号表（Symbol Table）

**问题**：如果 binary 被 `strip` 了，函数名会变成地址（如 `0xb4b778`），火焰图无法阅读。

**检查方法**：
```bash
# 检查是否有符号表
adb shell "nm -D your_binary | head -5"
# 如果输出函数名，说明有符号表
# 如果报 "no symbols"，说明被 strip 了

# 或者用 file 命令
adb shell "file your_binary"
# "not stripped" = 有符号表
# "stripped" = 没有符号表
```

**解决**：
```bash
# 编译时不要 strip
cmake -DCMAKE_STRIP=OFF ..

# 或者手动保留符号
cp libyour.so libyour_with_symbols.so
# 用未 strip 的版本做 profiling
```

#### DWARF 调试信息

**问题**：DWARF 是调用栈展开的另一种方式（比 frame pointer 更准确）。如果 binary 没有 DWARF 信息，`--call-graph dwarf` 无法工作。

**检查方法**：
```bash
# 检查是否有 debug 段
adb shell "readelf -S your_binary | grep debug"
# 应该有 .debug_info, .debug_line, .debug_abbrev 等
```

**注意**：DWARF 信息会显著增大 binary 大小（10x-100x），只在 profiling 时使用。

### 2.4 NDK 工具链配置

```bash
# 设置 NDK 路径
export ANDROID_NDK=/home/liu/android-ndk

# CMake 工具链文件
# $ANDROID_NDK/build/cmake/android.toolchain.cmake

# 完整的 CMake 配置示例
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_FLAGS="-fno-omit-frame-pointer" \
    -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer"
```

---

## 3. 推理框架编译注意事项

### 3.1 MNN

MNN 使用手写汇编优化关键计算路径，这些汇编函数**没有 `.cfi` 指令**，导致调用栈在汇编函数处中断。

```bash
# MNN 编译选项
cd third_party/MNN
mkdir build_android && cd build_android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMNN_ARM=ON \
    -DMNN_USE_THREAD_POOL=ON

# 注意：即使加了 -fno-omit-frame-pointer，汇编函数仍可能没有调用栈
# 这是 MNN 的已知限制，约 30-40% 采样会有不完整的调用栈
```

### 3.2 ONNX Runtime

```bash
# ORT 需要编译未 strip 的版本
cd third_party/onnxruntime
./build.sh --android \
    --config RelWithDebInfo \    # 关键：不要用 Release
    --build_shared_lib \
    --android_ndk $ANDROID_NDK

# 编译后的 .so 会包含符号表
# 验证：nm -D libonnxruntime.so | head -5
```

### 3.3 ncnn

```bash
cd third_party/ncnn
mkdir build_android && cd build_android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DNCNN_RUNTIME_CPU=OFF \
    -DNCNN_ARM82=ON
```

### 3.4 TFLite

```bash
# TFLite 需要编译带符号的版本
cd third_party/tensorflow
bazel build -c opt \
    --copt=-g \                    # 保留调试信息
    --copt=-fno-omit-frame-pointer \
    //tensorflow/lite:libtensorflowlite.so
```

---

## 4. 设备端准备

### 4.1 simpleperf 二进制

```bash
# 从 NDK 推送 simpleperf（推荐）
# NDK r25+ 路径
adb push $ANDROID_NDK/simpleperf/bin/android/arm64/simpleperf /data/local/tmp/
adb shell "chmod 755 /data/local/tmp/simpleperf"

# 验证
adb shell "/data/local/tmp/simpleperf --version"
```

### 4.2 FlameGraph 工具

```bash
# 克隆 FlameGraph 工具（在电脑端，不是设备端）
git clone https://github.com/brendangregg/FlameGraph.git tools/FlameGraph

# 验证
ls tools/FlameGraph/flamegraph.pl
# 应该存在
```

### 4.3 设备状态检查

```bash
# 检查设备连接
adb devices

# 检查 root 状态（非 root 也能用 simpleperf，但功能受限）
adb shell "whoami"

# 检查 CPU 信息
adb shell "cat /proc/cpuinfo | grep processor"

# 检查 CPU 频率（避免降频影响结果）
adb shell "for i in 0 1 2 3; do echo -n \"CPU\$i: \"; cat /sys/devices/system/cpu/cpu\$i/cpufreq/scaling_cur_freq; done"

# 建议：关闭后台应用、开启飞行模式、让手机冷却
```

---

## 5. 采集数据

### 5.1 simpleperf record 参数详解

```bash
simpleperf record \
    -p $PID \                      # 目标进程 PID
    -e cpu-cycles \                # 采样事件（cpu-cycles, cache-misses, branch-misses）
    -f 4000 \                      # 采样频率（Hz），推荐 1000-4000
    --call-graph dwarf \           # 调用栈展开方式（dwarf 或 fp）
    --duration 10 \                # 采样时长（秒）
    --user-buffer-size 65536 \     # 用户态缓冲区大小（KB），增大可减少采样丢失
    -o profiling/perf.data         # 输出文件
```

#### 参数选择建议

| 场景 | 采样频率 | 时长 | 调用栈模式 |
|------|----------|------|------------|
| 快速定位热点 | 1000 Hz | 5s | dwarf |
| 详细分析 | 4000 Hz | 10-30s | dwarf |
| 高频采样（大模型） | 8000 Hz | 30s+ | dwarf,16384 |
| 非 root 设备 | 1000-4000 Hz | 10s | fp |

#### 调用栈模式选择

| 模式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| `--call-graph dwarf` | 更准确，能穿透大部分函数 | 需要 DWARF 信息，perf.data 更大 | **推荐默认使用** |
| `--call-graph fp` | 快速，perf.data 小 | ARM64 上汇编函数会中断 | 非 root 设备 |
| `--call-graph dwarf,16384` | dump 更多栈信息 | perf.data 非常大 | 调用栈不完整时尝试 |

### 5.2 采集流程

```bash
# 1. 启动推理（后台）
adb shell "cd /data/local/tmp/benchmark && \
    LD_LIBRARY_PATH=. nohup ./benchmark_inference \
    --backend mnn --model mobilenetv2 --threads 2 --warmup 10 --runs 1000 \
    > /dev/null 2>&1 &"

# 2. 等待 warmup 完成（让 CPU 频率稳定）
sleep 3

# 3. 获取 PID
PID=$(adb shell "pidof benchmark_inference" | tr -d '\r')
echo "PID: $PID"

# 4. 启动 simpleperf 采集
adb shell "/data/local/tmp/simpleperf record \
    -p $PID \
    -e cpu-cycles \
    -f 4000 \
    --call-graph dwarf \
    --duration 10 \
    -o /data/local/tmp/benchmark/profiling/perf.data"

# 5. 拉取数据
adb pull /data/local/tmp/benchmark/profiling/perf.data .

# 6. 生成报告
adb shell "/data/local/tmp/simpleperf report-sample \
    -i /data/local/tmp/benchmark/profiling/perf.data \
    --show-callchain" > report.txt

# 7. 生成火焰图（电脑端）
./tools/FlameGraph/flamegraph.pl out.folded > flamegraph.svg
```

---

## 6. 常见问题排查

### Q1: 火焰图全是 `unknown` 或地址

**原因**：binary 被 strip 了，没有符号表。

**解决**：
```bash
# 检查
adb shell "file /data/local/tmp/benchmark/benchmark_inference"
# 如果显示 "stripped"，需要重新编译，不要 strip

# 或者用 readelf 检查
adb shell "readelf -S /data/local/tmp/benchmark/benchmark_inference | grep symtab"
# 如果没有 .symtab 段，说明被 strip 了
```

### Q2: 调用栈只有 2 层深度

**原因**：ARM64 汇编函数没有 `.cfi` 指令，DWARF 展开中断。

**解决**：
```bash
# 方法 1：加 -fno-omit-frame-pointer（对 C++ 代码有效）
cmake -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" ..

# 方法 2：换 fp 模式试试
simpleperf record --call-graph fp ...

# 方法 3：这是框架的已知限制，部分调用链仍有价值
```

### Q3: 采样丢失率高

**原因**：采样频率太高或缓冲区太小。

**解决**：
```bash
# 降低采样频率
simpleperf record -f 1000 ...

# 增大缓冲区
simpleperf record --user-buffer-size 65536 ...

# 增大 mmap 页数
simpleperf record -m 16384 ...
```

### Q4: kernel 符号显示为地址

**原因**：非 root 设备无法访问内核符号。

**解决**：
```bash
# 方法 1：获取 root 权限
adb shell "su -c 'echo 0 > /proc/sys/kernel/kptr_restrict'"

# 方法 2：忽略内核符号（不影响用户态分析）
simpleperf report-sample --remove-unknown-kernel-symbols ...
```

### Q5: perf.data 文件太大

**原因**：DWARF 模式 + 高频采样 = 大文件。

**解决**：
```bash
# 降低采样频率
simpleperf record -f 1000 ...

# 缩短采样时长
simpleperf record --duration 5 ...

# 或者用 fp 模式（perf.data 更小）
simpleperf record --call-graph fp ...
```

---

## 7. 编译选项速查表

| 选项 | 作用 | 影响 | 是否推荐 |
|------|------|------|----------|
| `-O2` / `-O3` | 编译优化 | 性能最好，但调试困难 | 推荐 |
| `-g` | 保留调试信息 | binary 增大 10-100x | 推荐做 profiling |
| `-fno-omit-frame-pointer` | 保留帧指针 | 性能损失 ~1-3% | **ARM64 强烈推荐** |
| `-fomit-frame-pointer` | 省略帧指针（默认） | 调用栈不完整 | 不推荐做 profiling |
| `-DNDEBUG` | 禁用 assert | 性能更好 | 推荐 |
| `RelWithDebInfo` | Release + 调试信息 | 最佳平衡 | **推荐** |
| `Debug` | 完整调试，无优化 | 性能差，但信息最全 | 仅调试 bug |
| `Release` | 完全优化，strip 符号 | 性能最好，但无法 profiling | 不推荐做 profiling |

---

## 8. 完整示例：一键采集火焰图

使用本项目的脚本：

```bash
# 最简用法
./scripts/simpleperf_profile.sh --backend mnn --model mobilenetv2

# 指定线程数和采样时长
./scripts/simpleperf_profile.sh --backend mnn --model mobilenetv2 --threads 2 --duration 15

# 使用 fp 模式（非 root 设备）
./scripts/simpleperf_profile.sh --backend mnn --model mobilenetv2 --callgraph fp

# 集成 profiling（同时采集 simpleperf + atrace + perfetto）
./scripts/profile_benchmark.sh --backend mnn --model mobilenetv2 --profile all
```

输出文件：
```
results/profiling/YYYYMMDD_HHMMSS_mobilenetv2_mnn/
├── simpleperf/
│   ├── perf.data              # 原始采样数据
│   ├── report_functions.txt   # 热点函数报告
│   ├── report_dso.txt         # DSO 分布报告
│   ├── out.folded             # 火焰图折叠格式
│   └── flamegraph.svg         # 火焰图（自动生成）
├── benchmark_output.txt       # benchmark 原始输出
├── device_info.txt            # 设备信息
└── summary.md                 # 汇总报告
```

---

## 9. 检查清单

采集火焰图前，逐项检查：

```
□ binary 是否有符号表？（nm -D 或 file 命令检查）
□ 是否加了 -fno-omit-frame-pointer？（ARM64 特别重要）
□ simpleperf 是否已推送到设备？
□ FlameGraph 工具是否已克隆？
□ 设备是否已连接？（adb devices）
□ 是否关闭了后台应用？
□ 是否让手机冷却了？（避免降频）
□ CPU 频率是否稳定？（cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq）
□ 采样频率是否合适？（1000-4000 Hz）
□ 采样时长是否足够？（至少 10 秒）
□ 采样丢失率是否可接受？（< 10%）
```
