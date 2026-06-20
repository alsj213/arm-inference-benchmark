# ARM端推理Benchmark项目指导

## 一、项目整体定位

### 项目名称（简历用）
**ARM端多框架推理性能对标与优化实践**

### 技术栈
C++ / ARM Neon / ONNX Runtime (ORT) / MNN / TVM

### 硬件平台推荐
- RK3588 开发板（推荐，工业级常用）
- Raspberry Pi 4
- 安卓手机（高通/麒麟芯片）

---

## 二、Benchmark 设计

### 测试维度

| 测试维度 | 具体内容 |
|---------|---------|
| **单算子Benchmark** | Conv1x1/3x3/7x7、Matmul、Softmax、Pooling、Add、Mul |
| **整模型Benchmark** | MobileNetV2/V3、ResNet18/50、YOLOv5s |
| **多Batch测试** | batch=1（端侧最常用）、batch=4、batch=8 |
| **多线程测试** | 单线程、2线程、4线程 |
| **精度模式** | FP32、FP16、INT8 |

### 性能指标定义
```
- 端到端时延（ms）：均值、标准差、P95、P99
- 单算子耗时占比（%）
- 算力利用率：实际GMACS / 理论峰值GMACS
- 内存占用峰值
- 内存带宽利用率
- CPU占用率

测试流程：100次 warmup + 1000次 正式测试
```

### 控制变量（必须）
```
✅ 同一硬件平台，CPU固定频率（关闭DVFS）
✅ 同一编译器版本：gcc/clang版本固定
✅ 同一编译选项：-O3 -march=armv8.2-a
✅ 线程绑核：避免核迁移
✅ 相同线程数：ORT和MNN使用相同线程配置
✅ 精度模式统一：都用FP32/FP16/INT8
```

---

## 三、性能优化步骤

### 阶段一：基线测试与瓶颈分析
```
1. 编译 ONNX Runtime 和 MNN baseline 版本
2. 跑完整benchmark，建立性能基线
3. 性能Gap分析：
   - 用 perf record/report 看热点函数
   - 用 perf stat 分析 Cache Miss、指令数、分支预测
   - 输出结论：算子实现差？图优化差？调度差？
```

### 阶段二：ORT原生优化（不借助TVM）

#### 图优化层面
```
1. 检查ORT图优化是否全开：
   SessionOptions options;
   options.graph_optimization_level = GraphOptimizationLevel::ORT_ENABLE_ALL;

2. 对比ORT vs MNN的Graph Pass差异
3. 手动添加ORT缺失的融合Pattern
4. 内存复用策略调优：Arena分配器参数
```

#### 算子层面优化（你最擅长）
```
1. 找出ORT比MNN慢的Top 5算子
2. 用Neon指令重写ORT的算子实现
3. 实现ORT Custom OP替换
4. 记录每个算子优化前后的性能提升
```

#### 调度层面
```
1. 线程数调优：ORT默认线程池参数
2. 算子内并行 vs 算子间并行的权衡
3. 大小核调度策略
```

### 阶段三：TVM联合优化（高阶亮点）

#### 方案A：TVM + ORT Custom OP
```
1. 用TVM调优瓶颈算子
2. TVM生成算子二进制
3. 集成到ORT作为Custom OP
4. 对比：ORT原生 vs TVM优化性能
```

#### 方案B：TVM编译整模型
```
1. TVM编译整模型
2. 对比ORT/MNN性能
3. 分析TVM优缺点和适用场景
```

---

## 四、关键技术点（体现专业度）

### 必须做的专业分析

| 技术点 | 说明 | 专业度 |
|--------|------|--------|
| **ARM PMU性能计数器** | 用硬件计数器分析真实瓶颈 | ⭐⭐⭐ |
| **Roofline模型分析** | 标注每个算子是计算Bound还是访存Bound | ⭐⭐⭐⭐⭐ |
| **性能Gap根因分析** | 不仅仅说"慢"，要说清"为什么慢" | ⭐⭐⭐⭐ |
| **成本收益分析** | 优化花了多少时间，收益多少 | ⭐⭐⭐ |
| **自动化测试脚本** | Python脚本自动跑所有case生成报告 | ⭐⭐⭐ |

### 避免的误区
❌ 不要只放性能对比数字表格
❌ 不要只说"优化了xx%"
✅ 要有分析、有原因、有思考过程

---

## 五、避坑指南

| 坑 | 解决方案 |
|----|----------|
| ORT默认优化没全开 | SessionOptions手动开启所有优化 |
| CPU频率动态调频 | cpupower frequency-set -g performance |
| 核迁移影响结果 | 绑核：taskset -c 0-3 ./benchmark |
| MNN默认开了FP16 | 统一精度模式，对比时确保一致 |
| ORT线程池默认值 | 显式设置intra_op_num_threads |
| 内存分配器影响 | 统一用Arena分配器 |

---

## 六、简历呈现模板

```
★ ARM端多框架推理性能Benchmark与优化实践
项目使用技术：C++ / ARM Neon / ONNX Runtime / MNN / TVM
硬件平台：RK3588 ARM开发板（Cortex-A76 + A55大小核）
项目描述：建立端侧推理框架标准化性能Benchmark体系，以MNN为标杆对ONNX Runtime
进行系统性性能优化，探索TVM自动代码生成在特定算子上的应用价值

核心工作：
1. 搭建标准化Benchmark测试框架，覆盖单算子、整模型、多线程、多精度等维度
2. 基于ARM PMU性能计数器建立性能分析方法论，使用Roofline模型定位瓶颈
3. 对比分析ONNX Runtime与MNN的性能Gap，定位图优化和算子实现的差异
4. 针对Top5瓶颈算子进行Neon指令级优化，平均性能提升25%-40%
5. 探索TVM自动调优特定算子，集成到ORT中，Matmul算子性能对比原生提升50%+
6. 输出端侧推理框架技术选型报告，为项目技术路线决策提供支撑

项目成果：
- 建立包含20+单算子、5+常用模型的端侧推理性能数据库
- ORT整体推理性能优化后提升30%+，达到MNN性能的90%
- Conv1x1算子优化后时延从1.2ms降至0.7ms，性能反超MNN
- 自动化测试脚本一键生成性能报告和对比图表
```

---

## 七、执行计划

| 阶段 | 时间 | 核心目标 |
|------|------|---------|
| 第1周 | 环境搭建 | 编译ORT/MNN/TVM，跑通基础测试 |
| 第2周 | 基线测试 | 完成全量benchmark，输出Gap分析报告 |
| 第3周 | 算子优化 | 完成3-5个核心算子优化 |
| 第4周 | TVM探索 + 文档 | 集成TVM优化，整理报告 |

---

## 八、加分项

### 1. 可视化报告
```
用 Python + Matplotlib 生成：
- 算子性能对比柱状图
- 多线程性能曲线
- Roofline模型散点图
- 优化前后对比图
```

### 2. 开源贡献
```
把优化的算子PR提交到 ONNX Runtime 社区
github链接放简历，面试官直接看代码质量
```

---

## 九、面试准备话术

```
"我做这个项目的初衷，是发现业界缺少一个标准化的端侧推理框架
Benchmark。我从单算子到整模型建立了完整的测试体系，用ARM PMU硬件
计数器和Roofline模型做深度性能分析，定位出ORT在哪些地方比MNN慢，
然后针对性地做算子级和图优化层面的优化，最后还探索了TVM自动代码
生成在特定算子上的优势和局限性。

通过这个项目，我完整经历了推理引擎性能优化的全流程：
从性能测试 → 瓶颈定位 → 方案设计 → 优化落地 → 效果验证，
这和推理引擎工程师的日常工作几乎完全一致。"
```

---

## 十、参考资料

- ONNX Runtime 官方文档：https://onnxruntime.ai/
- MNN 性能测试工具：https://github.com/alibaba/MNN/tree/master/tools/benchmark
- TVM 官方教程：https://tvm.apache.org/docs/tutorial/
- ARM NEON 编程指南：ARM官方文档
- Roofline Model 论文：伯克利论文
