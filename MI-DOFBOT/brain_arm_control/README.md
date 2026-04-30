# 脑电信号控制机械臂

本目录包含第一版脑电信号控制机械臂程序：

- `brain_arm_control.py`

程序通过 `neuropy.py` 读取 NeuroSky / MindWave 脑环数据，通过 `Arm_Lib.py` 控制机械臂，并参考原键盘控制程序的舵机动作映射。

## 功能概览

- 使用 30 次采样窗口，每次间隔 0.1 秒，约 3 秒做一次决策。
- 使用 `attention` / `meditation` 阈值计数控制机械臂动作。
- 使用 `blinkStrength` 检测眨眼：
  - 单次眨眼：切换夹爪开合。
  - 两次或更多眨眼：切换控制模式。
- 实现三个控制模式：
  - mode 0：升降模式，控制 3 号舵机。
  - mode 1：旋转模式，控制 1 号舵机。
  - mode 2：前后模式，控制 2 号舵机。
- 支持 `--dry-run`，不连接真实机械臂，只输出将执行的动作。
- 支持可选模型预测，只有传入 `--model-path` 时才尝试加载 PyTorch 模型。
- poorSignal 过高时不执行动作，poorSignal 等于 200 时机械臂回安全姿态并等待信号恢复。

## 依赖文件

程序默认会尝试从当前项目路径导入：

- `D:\XX\SITP脑机\总体文件\drone\neuropy.py`
- `D:\XX\SITP脑机\机械臂控制\0.py_install\Arm_Lib (Windows)\Arm_Lib.py`

如果你的文件位置变化，可以通过命令行参数指定：

```bash
--neuropy-dir "D:\path\to\neuropy_folder" --arm-lib-dir "D:\path\to\Arm_Lib_folder"
```

## 机械臂动作映射

程序优先参考 `keyboard_control.py` 中已经验证过的映射：

- 1 号舵机：左右旋转。
  - 减小角度：左转。
  - 增加角度：右转。
- 2 号舵机：前后伸缩。
  - 增加角度：前伸。
  - 减小角度：收回。
- 3 号舵机：上下。
  - 增加角度：上升。
  - 减小角度：下降。
- 4 号舵机：辅助关节。
- 6 号舵机：默认作为夹爪控制舵机。

默认安全姿态：

```python
[90, 90, 90, 90, 135, 90]
```

## Dry-run 测试

没有连接机械臂时，先用 dry-run 测试脑环信号：

```bash
python brain_arm_control.py --mindwave-port COM6 --dry-run
```

dry-run 模式下：

- 会读取脑环数据。
- 不会真实控制机械臂。
- 会打印窗口统计和将要执行的动作。
- Ctrl+C 后会正常清理退出。

## 真实机械臂运行

无模型规则控制：

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4
```

带可选模型：

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4 --model-path FinalModel.pth
```

如果模型加载失败，程序会自动降级为规则控制，不会因为模型不可用而退出。

## 常用参数

```bash
python brain_arm_control.py ^
  --mindwave-port COM6 ^
  --mindwave-baud 57600 ^
  --arm-port COM4 ^
  --angle-step 5 ^
  --move-time-ms 200 ^
  --attention-threshold 45 ^
  --meditation-threshold 55 ^
  --blink-threshold 100 ^
  --poor-signal-threshold 100 ^
  --gripper-servo-id 6 ^
  --gripper-open-angle 120 ^
  --gripper-close-angle 60
```

说明：

- `--angle-step` 默认 5，程序会自动限制最大不超过 10。
- `--move-time-ms` 默认 200。
- `--poor-signal-threshold` 默认 100。
- `--gripper-servo-id` 默认 6。如果实际夹爪不是 6 号，只需要改这个参数和开合角度。

## 控制规则

mode 0：升降模式

- attention 明显高于 meditation：`arm_up()`
- meditation 明显高于 attention：`arm_down()`
- 单眨眼：切换夹爪。
- 双眨眼：切换到 mode 1。

mode 1：旋转模式

- 有模型且模型加载成功：
  - `left`：`base_left()`
  - `right`：`base_right()`
  - `rest`：不动
- 无模型或模型预测失败：
  - attention 明显高于 meditation：`base_right()`
  - meditation 明显高于 attention：`base_left()`
- 单眨眼：切换夹爪。
- 双眨眼：切换到 mode 2。

mode 2：前后模式

- attention 明显高于 meditation：`arm_forward()`
- meditation 明显高于 attention：`arm_backward()`
- 单眨眼：切换夹爪。
- 双眨眼：切换回 mode 0。

## 安全策略

- 程序启动后先执行 home。
- 所有舵机角度写入前都会 clamp 到安全范围：
  - 1/2/3/4/6 号：0 到 180。
  - 5 号：0 到 270。
- `poorSignal >= 100` 时不执行动作，只保持当前位置。
- `poorSignal == 200` 时认为信号极差或断开，机械臂回 home 并等待信号恢复。
- Ctrl+C 或异常退出时，会停止脑环读取、机械臂回 home，并关闭机械臂串口。

## 注意事项

- 第一次运行建议务必加 `--dry-run`。
- 如果机械臂方向与预期相反，先不要改底层串口帧，应优先检查 `keyboard_control.py` 中同一舵机的实际方向，再调整本程序动作封装。
- 夹爪默认使用 6 号舵机，因为原键盘程序只提供了 6 号电机的 Space / X 控制映射，没有提供 5 号舵机夹爪映射。
