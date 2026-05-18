# BCI-TELLO

这个仓库按两层组织：

- `bin/`：只放可复用的基础能力，例如 EEG 采集、特征构建、模型推理、键盘读取和 JSON HTTP 传输。
- 设备目录：只放具体设备的控制流程与设备依赖。

## 目录

- `MI-CarControl/`：小车键盘控制与脑环控制。
- `MI-DroneControl/`：Tello 键盘控制、脑控飞行、预测、训练与诊断。
- `MI-DOFBOT/`：机械臂键盘控制与脑控程序。
- `models/`：共享模型产物，默认模型为 `FinalModel.pth`。
- `Spaceinvaders/`：脑环测试游戏。

## 常用入口

```bash
# 小车
python MI-CarControl/keyboard_control.py --host 192.168.149.1 --port 5000
python MI-CarControl/brain_control.py --mindwave-port COM6

# 无人机
python MI-DroneControl/diagnose.py --mindwave-port COM6
python MI-DroneControl/train.py
python MI-DroneControl/predict.py --mindwave-port COM6
python MI-DroneControl/brain_control.py --mindwave-port COM6
python MI-DroneControl/keyboard_control.py

# 机械臂
python MI-DOFBOT/keyboard_control.py --dry-run
python MI-DOFBOT/brain_control.py --mindwave-port COM6 --dry-run
```

## 通用接口

```python
from bin.eeg import BrainSignalReader, EEGSnapshot, build_feature_vector
from bin.keyboard import KeyboardReader
from bin.models import FinalUnifiedModel, ModelPredictor
from bin.transport import JsonHttpClient
```

`bin/` 不再承载小车、无人机、机械臂的具体业务流程；如果要增加新设备，请在新的设备目录里组合这些通用能力。
