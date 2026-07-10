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

三种脱离真实机械臂的方式：

| 方式 | 命令 | 说明 |
|---|---|---|
| dry-run | `--dry-run` | 走真实适配器但不下发串口指令，最贴近真机。 |
| 仿真臂 | `--simulated` | 用内存中的 `SimulatedArmController` 记录动作，完全不加载任何硬件库。 |
| 仅测脑环 | `--test-brain` | 只读脑环、报告窗口质量，不连接机械臂。 |

## 脑控逻辑（EEG → 控制命令）

程序复用公共核心 `bin/`（`bin.eeg` 采集、`bin.models` 模型、`bin.decoding` 解码），本目录只保留机械臂相关的接线。决策管线：

1. **眨眼是离散事件**：双眨眼在三个模式间循环（升降 → 转弯 → 前后），单眨眼开合夹爪。
2. **模式内产生"原始意图"**：
   - 升降 / 前后：比较窗口内 attention/meditation 计数，带**死区**（两者都低于阈值 → 静止 `rest`）。
   - 转弯：调用共享模型的 `predict_proba`，用 **置信度门控** 得到 `left / right / rest`（softmax 最大值低于阈值 → 归入 `rest`，不乱动）。
3. **多窗投票（滞回）**：原始意图先进入 `VoteWindow`，同一意图在最近若干窗口里达到票数才真正执行——单个噪声窗口无法带动舵机。

## 常用参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--confidence-threshold` | 0.5 | 转弯模式模型 softmax 的静息带阈值。 |
| `--vote-window` / `--vote-min` | 3 / 2 | 多窗投票的窗口长度与所需票数；设为 `1 / 1` 关闭平滑。 |
| `--min-decision-count` | 15 | attention/meditation 死区：两者都低于它则静止。 |
| `--decision-margin` | 3 | attention 与 meditation 计数至少相差此值才选边。 |
| `--angle-step` | 5 | 每次动作的舵机步进角度（1–10）。 |

## 安全建议

- 首次运行一定先使用 `--dry-run`。
- 保持机械臂周围无遮挡。
- 如果动作方向不符合预期，先用 `keyboard_control.py` 校验舵机方向。
