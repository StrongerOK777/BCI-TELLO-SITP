# NiceRice with DOFBOT-SE + NeuroSky

> 同济大学 SITP 项目记录：围绕 DOFBOT-SE 机械臂和 NeuroSky / MindWave 脑电头环，整理机械臂键盘控制、脑电信号控制、底层 Arm_Lib 使用和后续实验代码。

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Hardware](https://img.shields.io/badge/Hardware-DOFBOT--SE-orange)
![EEG](https://img.shields.io/badge/EEG-NeuroSky%20MindWave-6f42c1)
![Status](https://img.shields.io/badge/Status-SITP%20Prototype-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 项目简介

这个仓库用于记录 DOFBOT-SE 机械臂和 NeuroSky 脑电头环的移植、调试和控制程序开发。目前主要包含两条控制路线：

- 键盘控制：用于直接验证机械臂舵机方向、串口连接和基础动作。
- 脑电控制：使用 attention、meditation 和 blinkStrength 等脑电数据，以窗口采样和规则判断的方式控制机械臂。

第一版目标是稳定、安全、可调参，而不是一次性追求复杂控制。脑电信号可稳定区分的命令有限，所以当前脑控程序优先控制核心动作，再逐步扩展。

## 仓库结构

```text
NiceRice-with-DOFBOT-SE-NEUROSKY/
├── brain_arm_control/
│   ├── brain_arm_control.py   # 脑电窗口采样 + 机械臂控制主程序
│   ├── collect_training_data.py # 采集 left/right/rest EEG 训练数据
│   ├── train_eeg_model.py     # 训练可选 FinalModel.pth
│   └── README.md              # 脑控机械臂详细说明
├── keyboard_control/
│   ├── keyboard_control.py    # 实时键盘控制程序
│   └── README.md              # 键盘控制详细说明
├── Arm_Lib (Windows)/         # DOFBOT-SE 机械臂控制库
├── Arm_Lib.egg-info/          # Python 包信息
├── LICENSE
├── readme.md                  # 当前项目首页
└── setup.py
```

## 模块说明

| 目录 | 作用 | 适合场景 |
|---|---|---|
| `brain_arm_control/` | 读取脑环数据，并通过规则或可选模型映射为机械臂动作。 | 脑电控制实验、dry-run 调试、模式切换测试。 |
| `keyboard_control/` | 使用键盘实时控制 1、2、3、4、6 号舵机。 | 验证舵机方向、检查串口和机械臂动作。 |
| `Arm_Lib (Windows)/` | 封装机械臂底层串口控制接口。 | 被上层程序调用，不建议随意绕过。 |

## 快速开始

克隆仓库：

```bash
git clone https://github.com/907nicerice/NiceRice-with-DOFBOT-SE-NEUROSKY.git
cd NiceRice-with-DOFBOT-SE-NEUROSKY
```

安装基础依赖：

```bash
pip install pyserial pynput
```

如果只运行规则版脑电控制，不需要安装 PyTorch。只有在你传入 `--model-path FinalModel.pth` 时，程序才会尝试加载 PyTorch 模型。

## 脑电控制机械臂

进入脑控目录：

```bash
cd brain_arm_control
```

建议先 dry-run。这个模式会读取脑环数据，但不会真实控制机械臂：

```bash
python brain_arm_control.py --mindwave-port COM6 --dry-run
```

连接真实机械臂运行规则控制：

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4
```

带可选模型运行：

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4 --model-path FinalModel.pth
```

### 当前脑控范围

当前第一版不是 6 自由度全部脑控，而是先控制最核心、最容易验证的动作：

| 控制方式 | 控制内容 | 默认舵机 |
|---|---|---:|
| mode 0 | 上升 / 下降 | 3 |
| mode 1 | 底座左转 / 右转 | 1 |
| mode 2 | 前伸 / 收回 | 2 |
| 单眨眼 | 切换夹爪开合状态 | 6 |

说明：

- 4 号舵机在代码里已经预留 `joint4_decrease()` / `joint4_increase()`，但还没有分配脑控模式。
- 5 号舵机当前只在安全姿态里保持 135 度，暂不参与脑控动作。
- “切换夹爪”指的是切换同一个夹爪的开合状态，不是多个爪子之间切换。

### 控制规则

| 输入信号 | 行为 |
|---|---|
| attention 明显高于 meditation | 执行当前模式下的正向动作。 |
| meditation 明显高于 attention | 执行当前模式下的反向动作。 |
| 单眨眼 | 打开或闭合夹爪。 |
| 双眨眼 | 切换到下一个 mode。 |
| poorSignal 过高 | 不执行动作，保持安全。 |
| poorSignal 等于 200 | 认为信号极差或断开，机械臂回 home 并等待恢复。 |

### 关于 `.pth` 模型文件

`.pth` 是可选的 PyTorch 模型文件，不是必须项。当前程序里它只用于 mode 1 旋转模式，辅助判断 `left / right / rest`。

如果不传 `--model-path`，程序会直接使用规则控制：

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4
```

如果传入模型但加载失败，程序也不会崩溃，会自动降级回规则控制。


### 训练可选左右模型

如果要使用 `.pth` 模型，先采集三类训练数据：

```bash
cd brain_arm_control
python collect_training_data.py --mindwave-port COM6
```

再训练模型：

```bash
python train_eeg_model.py --data-dir data --output model/FinalModel.pth
```

当前模型只用于 mode 1 左右旋转，输出含义固定为：`0=left`、`1=right`、`2=rest`。它不会控制上/下、前/后或夹爪。
完整说明见：[`brain_arm_control/README.md`](brain_arm_control/README.md)

## 键盘控制机械臂

进入键盘控制目录：

```bash
cd keyboard_control
```

运行：

```bash
python keyboard_control.py
```

默认映射：

| 输入 | 舵机 | 动作 |
|---|---:|---|
| `A` / `D` | 1 | 底座左转 / 右转 |
| `W` / `S` | 2 | 前伸 / 收回 |
| `↑` / `↓` | 3 | 上升 / 下降 |
| `Q` / `E` | 4 | 辅助关节减小 / 增加 |
| `Space` / `X` | 6 | 末端夹爪方向控制 |
| `R` | 全部 | 回到安全中位姿态 |

完整说明见：[`keyboard_control/README.md`](keyboard_control/README.md)

## 安全提示

- 第一次运行脑控程序时，务必先使用 `--dry-run`。
- 机械臂启动和退出时，默认回到安全姿态 `[90, 90, 90, 90, 135, 90]`。
- 脑控程序每次动作默认只改变 5 度，且最大不超过 10 度。
- poorSignal 保护优先级高于所有动作，信号差时不会继续控制机械臂。
- 不要绕过 `Arm_Lib.py` 直接手写底层串口帧，除非你正在调试底层协议。
- 如果实际动作方向和预期相反，先用 `keyboard_control.py` 验证舵机方向，再调整上层动作映射。

## 后续计划

- 做不同使用者的 attention / meditation 阈值校准。
- 增强眨眼检测和防抖逻辑。
- 增加离线 EEG 数据记录，用于后续调参和训练。
- 持续优化 EEG 数据采集、阈值调参和模型训练流程。
- 逐步扩展 4 号、5 号舵机的脑控模式。
- 补充接线图、实物照片和演示视频。

## 项目说明

维护者：`907nicerice`

这个仓库既是 SITP 项目的实验记录，也是一个面向脑机接口和机械臂控制的工程笔记。当前版本还处在原型阶段，欢迎继续迭代、测试和整理。
