# MI-DroneControl

Tello 无人机控制目录。训练不在这里进行；公共训练请使用 `TrainUser/train_user.py`，训练产物会写入 `src/`。

## 文件说明

| 文件 | 用途 |
|---|---|
| `keyboard_control.py` | Tello 键盘控制与视频预览。 |
| `brain_control.py` | 使用共享模型和脑环信号控制 Tello。 |
| `predict.py` | 连续读取脑环窗口并输出模型预测结果。 |
| `diagnose.py` | 脑环信号诊断，查看 attention、meditation、blinkStrength、poorSignal 等。 |
| `drone_hardware.py` | Tello 硬件适配层。 |
| `neuropy.py` | 无人机侧 NeuroSky / MindWave 驱动副本。 |

## 脑环接口

脑环串口统一在根目录 `bci_interface.py` 中配置。更换串口后，这里的诊断、预测和脑控程序都会自动复用。

## 常用命令

```bash
# 检查脑环信号
python MI-DroneControl/diagnose.py

# 连续查看模型预测
python MI-DroneControl/predict.py

# 键盘控制 Tello
python MI-DroneControl/keyboard_control.py

# 脑环控制 Tello
python MI-DroneControl/brain_control.py
```

默认模型路径：

```text
src/models/FinalModel.pth
```

如需临时指定模型：

```bash
python MI-DroneControl/brain_control.py --model-path src/models/FinalModel.pth
```
