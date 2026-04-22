# Qualcomm QNN SDK 安装指南

## ⚠️ 前置说明

QNN (Qualcomm AI Engine Direct) SDK 需要从 Qualcomm 官方网站下载，需要注册 Qualcomm Developer Network 账号。

## 📥 下载方式

### 方式一：通过 QPM 下载（推荐）

1. 访问 https://qpm.qualcomm.com/
2. 注册/登录 Qualcomm Developer Network 账号
3. 搜索 "Qualcomm AI Engine Direct SDK" 或 "QNN SDK"
4. 下载最新版本（建议 v2.2x 或更高）
5. 解压到 `third_party/QNN/` 目录

### 方式二：通过 Qualcomm Package Manager

```bash
# 安装 QPM CLI 工具（需要 Node.js）
npm install -g @qcom/qpm-cli

# 登录并下载 QNN SDK
qpm login
qpm install qnn-sdk
```

## 📁 目录结构

解压后的目录结构应该如下：

```
third_party/QNN/
├── include/
│   ├── QnnCommon.h
│   ├── QnnError.h
│   ├── QnnGraph.h
│   ├── QnnInterface.h
│   └── QnnSystemInterface.h
├── lib/
│   └── aarch64-android/
│       ├── libQnnSystem.so
│       ├── libQnnCpu.so
│       ├── libQnnHtp.so
│       └── ...
└── docs/
```

## 🔧 验证安装

```bash
# 检查头文件
ls third_party/QNN/include/QnnInterface.h

# 检查库文件
ls third_party/QNN/lib/aarch64-android/libQnnSystem.so

# 重新编译
cd build_android && cmake .. -DBENCHMARK_QNN=ON && make -j8
```

## 📱 测试

```bash
# 推送库文件到设备
adb push third_party/QNN/lib/aarch64-android/*.so /data/local/tmp/

# 运行测试
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./benchmark_inference --backend qnn --model mobilenetv2"
```

## ℹ️ 说明

- QNN SDK 体积较大（约 2GB+）
- 需要骁龙 8 系及以上芯片支持（HTP 后端）
- 支持量化模型（INT8）性能最佳
