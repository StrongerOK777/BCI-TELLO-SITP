# MI-DOFBOT

DOFBOT 机械臂控制目录。支持键盘控制和脑环控制。公共训练由 `TrainUser/train_user.py` 完成，最终模型由本目录的 `brain_control.py` 复用。

## 文件说明

| 文件 / 目录 | 用途 |
|---|---|
| `keyboard_control.py` | 机械臂键盘控制入口。 |
| `brain_control.py` | 机械臂脑环控制入口。 |
| `arm_hardware.py` | DOFBOT 机械臂硬件适配。 |
| `Arm_Lib (Windows)/` | DOFBOT 底层串口控制库。 |
| `readme.md` | 本说明文件。 |

## 脑环接口

脑环串口统一在根目录 `bci_interface.py` 中配置。通常不需要在本目录里修改代码。

## 键盘控制

建议先 dry-run 或保持机械臂处于安全位置：

```bash
python MI-DOFBOT/keyboard_control.py --dry-run
```

## 脑环控制

首次建议使用 dry-run：

```bash
python MI-DOFBOT/brain_control.py --dry-run
```

确认信号和逻辑正常后，再连接真实机械臂：

```bash
python MI-DOFBOT/brain_control.py --arm-port COM4
```

默认模型路径：

```text
src/models/FinalModel.pth
```

## 安全建议

- 首次运行一定先使用 `--dry-run`。
- 保持机械臂周围无遮挡。
- 如果动作方向不符合预期，先用 `keyboard_control.py` 校验舵机方向。
