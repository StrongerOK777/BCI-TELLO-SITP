# MI-DroneControl

Tello 无人机控制目录。训练不在这里进行；公共训练请使用 `TrainUser/train_user.py`，训练产物会写入 `src/`。

## 文件说明

| 文件 | 用途 |
|---|---|
| `keyboard_control.py` | Tello 键盘控制与视频预览。 |
| `brain_control.py` | 使用脑环信号控制 Tello；转弯模式默认“专注执行、单眨眼换方向”，可选共享模型。 |
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

脑控逻辑：

- 双眨眼在“升降 → 转弯 → 前后”间切换（起飞后才能进入转弯）。
- 升降 / 前后：attention 更高则上升 / 前进，meditation 更高则下降 / 后退。
- 转弯（默认 `--turning-source rule`）：单眨眼在左/右之间切换当前方向（本窗口不动作），窗口内足够多的采样 attention ≥ `--focus-threshold`（默认 55）时向当前方向转一次，专注不足则悬停。
- 转弯（`--turning-source model`）：使用共享模型 `src/models/FinalModel.pth` 预测左 / 右 / 静息，可用 `--model-path` 指定其他模型。
- 没有真机时可加 `--simulated` 使用模拟无人机。

```bash
python MI-DroneControl/brain_control.py --simulated --focus-threshold 60
python MI-DroneControl/brain_control.py --turning-source model --model-path src/models/FinalModel.pth
```
