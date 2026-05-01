# 常见问题 (FAQ)

## ❓ 常见问题

### Q: 为什么 TVM 性能比其他框架差？

**A:** 当前 TVM 后端使用的是纯手工 C++ 实现，作为性能基线。真实的 TVM 编译器通过以下方式可以获得显著提升：
- NEON SIMD 指令优化
- 算子融合
- AutoTVM 自动调优
- 调度优化

启用完整 TVM 编译器后，预计性能可提升 2-4 倍。

---

### Q: 如何添加一个新的推理框架？

**A:** 参考以下步骤：
1. 在 `third_party/` 添加框架的源代码或库
2. 在 `src/backends/` 创建 `xxx_backend.h` 和 `xxx_backend.cpp`
3. 继承 `BenchmarkBackend` 并实现所有纯虚函数
4. 在 `src/common/benchmark.cpp` 的 `create_backend` 函数中注册
5. 在 `src/CMakeLists.txt` 添加编译选项

参考 `ncnn_backend` 或 `mnn_backend` 的实现方式。

---

### Q: 如何添加新的测试模型？

**A:** 
1. 将模型文件放入 `models/classification/` 或相应目录
2. 在 `src/models/` 添加模型的配置信息（输入输出形状等）
3. 更新 `src/main.cpp` 中的模型列表

---

### Q: 如何在不同的设备上测试？

**A:** 
```bash
# 确保设备已连接
adb devices

# 安装特定设备的驱动（如果需要）

# 推送并运行测试
./scripts/run_benchmark_android.sh --backend all --model mobilenetv2

# 查看设备信息
adb shell cat /proc/cpuinfo
adb shell getprop ro.product.model
```

---

### Q: 如何启用 GPU 加速？

**A:** 
1. 确保编译时启用了相应框架的 GPU 选项
2. 运行时添加 `--gpu` 参数：
```bash
./benchmark_inference --backend mnn --model mobilenetv2 --gpu
```

注意：不是所有框架都支持 GPU，需要参考具体框架的文档。

---

### Q: 如何复现测试结果？

**A:** 
为了确保测试结果可复现：
1. 关闭所有后台应用
2. 保持设备常温
3. 使用相同的操作系统版本
4. 使用相同的编译选项
5. 运行足够的 warmup 次数（建议 >= 10）

---

### Q: MNN 和 ncnn 哪个更好？

**A:** 
这取决于具体的使用场景：

| 维度 | MNN | ncnn |
|------|-----|------|
| **性能** | 稍好 (MobileNetV2: ~18.6ms) | 略逊 (MobileNetV2: ~19.6ms) |
| **生态** | 阿里系，支持多种前端 | 腾讯系，社区活跃 |
| **文档** | 较完善 | 中文文档丰富 |
| **工具链** | 较完整 | 有专门的模型转换工具 |

两者都是优秀的端侧推理框架，建议根据具体项目需求选择。

---

### Q: 如何进行 INT8 量化测试？

**A:** 
1. 准备量化后的模型（各框架有自己的量化工具）
2. 运行时指定精度：
```bash
./benchmark_inference --backend mnn --model mobilenetv2 --precision int8
```

注意：不是所有框架都支持 INT8，需要参考各框架文档。

---

### Q: 测试结果的内存占用数据准确吗？

**A:** 
当前内存统计使用的是 `get_memory_usage_kb()`，通过读取 `/proc/self/status` 获取 VmRSS。这个值包括：
- 模型加载后的内存
- 运行时的中间张量
- 框架本身的内存开销

可以作为相对比较的参考，但具体数值可能受系统影响。

---

### Q: 如何贡献测试结果？

**A:** 
1. 确保测试环境稳定
2. 运行完整测试
3. 将结果格式化为 Markdown 表格
4. 提交 PR 更新 `docs/results_xxx.md`

---

## 🔧 开发相关问题

### Q: 如何调试后端代码？

**A:** 
1. 使用 Debug 模式编译：
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

2. 添加调试打印
3. 使用 Android NDK 的 gdb 进行远程调试

### Q: 如何添加新的性能指标？

**A:** 
1. 在 `BenchmarkResult` 结构体中添加字段
2. 在 `run_benchmark` 函数中收集数据
3. 在 `main.cpp` 中打印输出

---

## 📞 还有问题？

如果以上 FAQ 没有解决您的问题：
1. 搜索现有 [Issues](https://github.com/your-org/arm-inference-benchmark/issues)
2. 提交新 Issue 并附上详细信息
