# Shared Models

将跨设备复用的训练产物放在这里。默认共享模型文件名为 `FinalModel.pth`，小车、无人机、机械臂的 `brain_control.py` 都默认读取它，也支持通过 `--model-path` 指定其他路径。
