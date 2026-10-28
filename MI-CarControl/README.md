# MI-CarControl

小车控制目录。这里保留小车相关的最小运行闭环：键盘控制、脑环控制和小车侧 NeuroSky 驱动。

## 文件说明

| 文件 | 用途 |
|---|---|
| `keyboard_control.py` | 通过键盘向小车 HTTP 接口发送前进、后退、左转、右转和停止指令。 |
| `brain_control.py` | 脑环控制小车的命令行入口，控制逻辑来自共享的 `bin/control.py`。 |
| `car_device.py` | 小车适配器：意图映射为 前进/后退/左转/右转，静止和急停都发“停止”；`CarSignalClient` HTTP 客户端也在这里。 |
| `test_brain.py` | 独立脑环信号测试脚本，不连接小车，用于排查串口和信号质量问题。 |
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

转向模式默认不使用模型。如需改用共享模型 `src/models/FinalModel.pth`：

```bash
python MI-CarControl/brain_control.py --turning-source model
```

脑控逻辑与另外两个设备相同（详见根目录 README 的“统一控制方式”）：

- 小车只有两个模式：前后 → 转弯，默认进入前后模式；双眨眼切换。
- 前后模式：attention 与 meditation 计数都低于 `--min-decision-count`（默认 20）或相差小于 `--decision-margin`（默认 3）时停止，否则专注更高前进、放松更高后退。
- 转弯模式（默认 `--turning-source rule`）：单眨眼在左/右之间切换当前方向（本窗口不动作），窗口内足够多的采样 attention ≥ `--focus-threshold`（默认 55）时向当前方向转一次，专注不足则停止。
- 转弯模式（`--turning-source model`）：模型预测 `left / right / rest` 并做置信度门控（`--confidence-threshold`，默认 0.5）；启动时会打印模型的分组测试准确率和显著性阈值。
- 小车没有附加动作，其他模式下的单眨眼只提示“重新测量”。
- `--test-brain` 会额外打印每个窗口中 attention ≥ 专注阈值的采样数和 attention 中位数，用来为自己校准 `--focus-threshold`。
- 信号质量差时只报警；脑环断开（poorSignal=200）时发送“停止”并回到前后模式；模型不可用时按静止处理。

也可以用 [统一控制面板](../control_panel/README.md) 在网页上选择本设备，统一显示脑电、模式和急停。

## 脑环诊断

如果出现 `attention=0`、`meditation=0` 或 `poorSignal=200`，先单独测试脑环：

```bash
python MI-CarControl/test_brain.py --duration 10
```

也可以使用脑控程序内置的测试模式：

```bash
python MI-CarControl/brain_control.py --test-brain
```

如果只想测试脑环和控制逻辑、不连接真实小车：

```bash
python MI-CarControl/brain_control.py --dry-run
```
