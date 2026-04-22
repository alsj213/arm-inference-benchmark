# 贡献指南

感谢您对本项目的关注！欢迎提交 Issue 和 Pull Request。

## 🚀 快速开始

### 提交 Issue

如果您发现 bug 或有新功能建议，请：

1. 首先在 [Issues](https://github.com/your-org/arm-inference-benchmark/issues) 中搜索是否已有类似问题
2. 新建 Issue 时请按照模板填写详细信息
3. 对于 bug 报告，请附上复现步骤和环境信息

### 提交 Pull Request

1. **Fork 本仓库**
   ```bash
   git clone https://github.com/your-username/arm-inference-benchmark.git
   cd arm-inference-benchmark
   ```

2. **创建特性分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **推送到远程**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **创建 Pull Request**
   - 在 GitHub 上打开 Pull Request
   - 详细描述更改内容
   - 关联相关的 Issue

## 📝 代码规范

### C++ 代码风格

- 遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- 文件名使用小写加下划线：`ncnn_backend.cpp`
- 类名使用大驼峰：`class NCNNBackend`
- 函数和变量使用小驼峰：`void runBenchmark()`
- 成员变量使用下划线后缀：`int input_size_`
- 使用 4 空格缩进

### Git 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型：
- `feat`: 新功能
- `fix`: bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具链相关

示例：
```
feat(ncnn): add GPU delegate support

Add Vulkan GPU acceleration for ncnn backend.
Improve inference speed by ~2x.

Fixes #123
```

## 🔧 开发指南

### 添加新的推理框架

参考 `src/backends/` 下现有的后端实现：

1. 创建 `xxx_backend.h` 和 `xxx_backend.cpp`
2. 继承 `BenchmarkBackend` 基类
3. 实现所有纯虚函数：`init()`, `infer()`, `deinit()`, `name()`
4. 在 `src/common/benchmark.cpp` 中注册新后端
5. 更新 `src/CMakeLists.txt` 添加编译选项

### 添加新的测试模型

1. 在 `src/models/` 下添加模型信息
2. 更新 `src/main.cpp` 中的模型列表
3. 添加模型下载脚本到 `models/` 目录

### 构建测试

```bash
# 本地构建
mkdir -p build && cd build
cmake ..
make -j8

# Android 交叉编译
export ANDROID_NDK=/path/to/ndk
./scripts/build_android.sh
```

## ✅ PR 检查清单

提交 PR 前请确认：

- [ ] 代码可以正常编译
- [ ] 遵循项目代码风格
- [ ] 相关文档已更新
- [ ] 测试验证通过
- [ ] 提交信息符合规范

## 📄 许可证

通过贡献代码，您同意您的贡献将根据项目的 [MIT 许可证](LICENSE) 发布。

## 💬 其他

如有任何疑问，欢迎通过 Issue 或 Discussion 交流！

---
感谢您的贡献！🎉
