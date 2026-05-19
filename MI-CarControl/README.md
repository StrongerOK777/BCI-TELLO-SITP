# MI-CarControl

小车控制目录。这里保留小车相关的最小运行闭环：键盘控制、脑环控制和小车侧 NeuroSky 驱动。

## 文件说明

| 文件 | 用途 |
|---|---|
| `keyboard_control.py` | 通过键盘向小车 HTTP 接口发送前进、后退、左转、右转和停止指令。 |
| `brain_control.py` | 通过脑环控制小车；前后模式使用注意力/冥想规则，转向模式使用共享模型。 |
| `neuropy.py` | 小车侧 NeuroSky / MindWave 驱动副本。 |
| `README.md` | 本说明文件。 |

## 脑环接口

脑环串口统一在根目录 `bci_interface.py` 中配置。通常不需要给本目录程序手动传 `--mindwave-port`。

## 运行键盘控制

```bash
python MI-CarControl/keyboard_control.py --host 192.168.149.1 --port 5000
```

按键：

| 按键 | 动作 |
|---|---|
| `i` | 前进 |
| `k` | 后退 |
| `j` | 左转 |
| `l` | 右转 |
| 空格 | 停止 |

## 运行脑环控制

```bash
python MI-CarControl/brain_control.py
```

默认读取共享模型：

```text
src/models/FinalModel.pth
```

脑控逻辑：

- 默认进入前后模式。
- 双眨眼切换“前后模式 / 转向模式”。
- 前后模式：注意力更高则前进，冥想更高则后退。
- 转向模式：模型预测 `left / right / rest`，对应左转、右转、停止。
- 信号质量差或模型不可用时，只报警，不主动发送新动作。
