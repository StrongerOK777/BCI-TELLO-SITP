# TrainUser

`TrainUser/` 是全项目唯一的公共脑环训练入口。它负责采集 NeuroSky / MindWave 脑环数据，训练左手、右手、静息三分类模型，并把训练产物写入根目录 `src/`，供小车、无人机和机械臂共同使用。

## 文件说明

| 文件 | 用途 |
|---|---|
| `train_user.py` | 公共训练主程序：采集数据、训练教师模型、蒸馏最终模型。 |
| `neuropy.py` | NeuroSky / MindWave 设备驱动。 |
| `README.md` | 本说明文件。 |

## 脑环接口配置

不要在训练程序里改串口。统一修改根目录：

```text
bci_interface.py
```

常见配置示例：

```python
MINDWAVE_PORT = "COM5"                    # Windows
MINDWAVE_PORT = "/dev/cu.usbmodem2017_2_251"  # macOS
MINDWAVE_BAUD = 57600
```

修改后，训练程序和三个硬件的 `brain_control.py` 都会自动复用同一配置。

## 运行训练

在项目根目录执行：

```bash
python TrainUser/train_user.py
```

训练过程会显示 pygame 提示窗口，引导用户采集：

- 左手运动想象数据
- 右手运动想象数据
- 静息数据

## 输出位置

```text
src/data/
  actionleft.txt
  actionright.txt
  rest.txt

src/models/
  FinalModel.pth
  model_1_wide_model.pth
  model_2_deep_model.pth
  ...

src/picture/
  training_validation_loss.png
  model_comparison.png
  final_model_training.png
```

其中 `src/models/FinalModel.pth` 是最终共享模型。三个硬件默认读取这个文件。

## 训练后如何使用

```bash
python MI-CarControl/brain_control.py
python MI-DroneControl/brain_control.py
python MI-DOFBOT/brain_control.py --dry-run
```

如果想临时使用别的模型，可以通过 `--model-path` 覆盖：

```bash
python MI-CarControl/brain_control.py --model-path src/models/FinalModel.pth
```
