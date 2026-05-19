# 脑电信号控制机械臂

本目录是 DOFBOT-SE + NeuroSky / MindWave 的脑控机械臂部分，包含三类程序：

| 文件 | 作用 |
|---|---|
| `brain_arm_control.py` | 主控制程序，读取脑环数据并控制机械臂。 |
| `collect_training_data.py` | 采集 `left / right / rest` 三类 EEG 训练数据。 |
| `train_eeg_model.py` | 训练可选模型并导出 `model/FinalModel.pth`。 |

第一版的重点是稳定、安全、可调参。模型只用于 **mode 1 左右旋转**，不控制 6 个自由度。

## 当前脑控范围

| 控制方式 | 控制内容 | 默认舵机 |
|---|---|---:|
| mode 0 | 上升 / 下降 | 3 |
| mode 1 | 底座左转 / 右转 | 1 |
| mode 2 | 前伸 / 收回 | 2 |
| 单眨眼 | 同一个夹爪打开 / 闭合切换 | 6 |

说明：

- 4 号舵机在代码中保留了 `joint4_decrease()` / `joint4_increase()`，但还没有分配脑控模式。
- 5 号舵机当前只在安全姿态里保持 135 度。
- “切换夹爪”指的是同一个夹爪在打开和闭合之间切换，不是多个夹爪之间切换。

## 交互式端口填写

如果运行时不传串口参数，程序会在启动前询问。

```bash
python brain_arm_control.py
```

会提示：

```text
请输入脑环端口，例如 COM6:
请输入机械臂端口，例如 COM4，直接回车使用 COM4:
```

dry-run 模式只需要填写脑环端口：

```bash
python brain_arm_control.py --dry-run
```

如果已经通过命令行传入端口，程序不会重复询问：

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4
```

## Dry-run 测试

没有连接机械臂时，先用 dry-run 测脑环信号：

```bash
python brain_arm_control.py --mindwave-port COM6 --dry-run
```

dry-run 模式下：

- 会读取脑环数据。
- 不会真实控制机械臂。
- 会打印每个窗口的统计结果和将要执行的动作。
- Ctrl+C 后会正常清理退出。

## 真实机械臂运行

无模型规则控制：

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4
```

带可选模型：

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4 --model-path model/FinalModel.pth
```

如果模型加载失败，程序会自动降级为规则控制，不会因为模型不可用而退出。

## 训练数据采集

采集脚本只连接脑环，不连接机械臂。

```bash
python collect_training_data.py
```

如果不传端口，会提示：

```text
请输入脑环端口，例如 COM6:
```

也可以直接传参：

```bash
python collect_training_data.py --mindwave-port COM6
```

默认行为：

- 采集三类：`left`、`right`、`rest`。
- 每类默认 10 轮。
- 每轮准备 3 秒。
- 每轮采集 30 个有效样本。
- 每个样本间隔 0.1 秒。
- `poorSignal > 20` 或 attention / meditation 为 0 时跳过样本。
- 默认输出到 `data/`：
  - `data/actionleft.txt`
  - `data/actionright.txt`
  - `data/rest.txt`

常用参数：

```bash
python collect_training_data.py ^
  --mindwave-port COM6 ^
  --output-dir data ^
  --rounds-per-label 10 ^
  --samples-per-round 30 ^
  --poor-signal-threshold 20
```

默认会清空旧的三个数据文件。如果想继续追加数据，加：

```bash
python collect_training_data.py --mindwave-port COM6 --append
```

## 模型训练

训练脚本读取 `data/` 中的三类 txt 文件，生成 `model/FinalModel.pth`。

```bash
python train_eeg_model.py --data-dir data --output model/FinalModel.pth
```

模型输入方式：

- 每条 EEG 原始数据构造 14 维特征。
- 一个窗口默认 30 条数据。
- 对窗口求 mean 和 std。
- 最终输入为 28 维。

输出标签：

```text
0 -> left
1 -> right
2 -> rest
```

因此模型只用于 mode 1：

```text
left  -> base_left()
right -> base_right()
rest  -> 不动
```

常用训练参数：

```bash
python train_eeg_model.py ^
  --data-dir data ^
  --output model/FinalModel.pth ^
  --window-size 30 ^
  --stride 1 ^
  --epochs 150 ^
  --patience 25 ^
  --batch-size 32
```

训练需要安装：

```bash
pip install numpy torch
```

## 控制规则

mode 0：升降模式

- attention 明显高于 meditation：`arm_up()`
- meditation 明显高于 attention：`arm_down()`
- 单眨眼：夹爪打开 / 闭合切换。
- 双眨眼：切换到 mode 1。

mode 1：旋转模式

- 有模型且模型加载成功：
  - `left`：`base_left()`
  - `right`：`base_right()`
  - `rest`：不动
- 无模型或模型预测失败：
  - attention 明显高于 meditation：`base_right()`
  - meditation 明显高于 attention：`base_left()`
- 单眨眼：夹爪打开 / 闭合切换。
- 双眨眼：切换到 mode 2。

mode 2：前后模式

- attention 明显高于 meditation：`arm_forward()`
- meditation 明显高于 attention：`arm_backward()`
- 单眨眼：夹爪打开 / 闭合切换。
- 双眨眼：切换回 mode 0。

## 安全策略

- 程序启动后先执行 home。
- 默认安全姿态为 `[90, 90, 90, 90, 135, 90]`。
- 每次动作默认只改变 5 度，最大不超过 10 度。
- 所有舵机角度写入前都会限制到安全范围：
  - 1/2/3/4/6 号：0 到 180。
  - 5 号：0 到 270。
- `poorSignal >= 100` 时不执行动作，只保持当前位置。
- `poorSignal == 200` 时认为信号极差或断开，机械臂回 home 并等待信号恢复。
- Ctrl+C 或异常退出时，会停止脑环读取、机械臂回 home，并关闭机械臂串口。

## 依赖路径

程序默认会尝试从当前项目路径导入：

- `D:\XX\SITP脑机\总体文件\drone\neuropy.py`
- `D:\XX\SITP脑机\机械臂控制\0.py_install\Arm_Lib (Windows)\Arm_Lib.py`

如果文件位置变化，可以通过命令行参数指定：

```bash
--neuropy-dir "D:\path\to\neuropy_folder" --arm-lib-dir "D:\path\to\Arm_Lib_folder"
```
