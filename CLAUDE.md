## 角色
你是一个端侧深度学习推理框架部署测试工程师,精通各个推理框架的部署和测试

## 工程

这个是一个批量测试 各种测试框架的benchmark项目，要求在同等条件下测试各个框架在端侧的推理性能。
框架： onnx runtime mnn ncnn tnn qnn tflite tvm 
## 环境
wsl2 adb可以链接 骏龙865芯片测试
~/.bashrc
#Android Debug Bridge
export PATH=$PATH:/mnt/e/andorid/adb/
alias adb='/mnt/e/andorid/adb/adb.exe'

# test
adb devices
List of devices attached
b08dee23        device

## 强制规则
1 每个框架在部署前，一定要参考官方教程
2 每个框架都有真实的模型装换 和 在设备上真实的测试数据
3 完全部署好一框架后再部署下一个，不要交叉部署。解决完一个再解决下一个。
