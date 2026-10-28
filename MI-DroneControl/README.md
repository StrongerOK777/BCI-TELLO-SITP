# MI-DroneControl

Tello 无人机控制目录。训练不在这里进行；公共训练请使用 `TrainUser/train_user.py`，训练产物会写入 `src/`。

## 文件说明

| 文件 | 用途 |
|---|---|
| `keyboard_control.py` | Tello 键盘控制与视频预览。 |
| `brain_control.py` | 脑环控制 Tello 的命令行入口，控制逻辑来自共享的 `bin/control.py`。 |
| `drone_device.py` | 无人机适配器：负责起飞/降落和高度记录，未起飞时拒绝水平和转向动作，急停为降落。 |
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

脑控逻辑与另外两个设备相同（详见根目录 README 的“统一控制方式”）：

- 双眨眼在“升降 → 前后 → 转弯”间循环。
- 升降：在地面时专注更高为起飞，之后每次上升 `--height-step`（默认 10cm，最高 `--max-height` 150cm）；放松更高为下降，回到起飞高度（`--takeoff-height` 100cm）时降落。
- 前后：专注更高前进，放松更高后退。两者都低于 `--min-decision-count` 或相差小于 `--decision-margin` 时悬停。
- **未起飞时只接受起飞**，前后和转弯动作会被拒绝并提示“未起飞”。
- 转弯（默认 `--turning-source rule`）：单眨眼在左/右之间切换当前方向（本窗口不动作），窗口内足够多的采样 attention ≥ `--focus-threshold`（默认 55）时向当前方向转一次，专注不足则悬停。
- 转弯（`--turning-source model`）：使用共享模型 `src/models/FinalModel.pth` 加置信度门控，可用 `--model-path` 指定其他模型。
- 脑环断开（poorSignal=200）或程序退出时自动降落。
- 没有真机时可加 `--simulated` 使用模拟无人机；`--test-brain` 只读取脑环。

也可以用 [统一控制面板](../control_panel/README.md) 在网页上选择本设备，统一显示脑电、模式和急停。

```bash
python MI-DroneControl/brain_control.py --simulated --focus-threshold 60
python MI-DroneControl/brain_control.py --turning-source model --model-path src/models/FinalModel.pth
```
