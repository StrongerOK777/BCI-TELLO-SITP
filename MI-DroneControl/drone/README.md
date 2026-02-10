# BCI Training (Smart Ward inspired)

这个脚本用于采集 NeuroSky EEG 数据并训练一个三分类模型（左/右/休息）。

## 运行

在项目根目录执行：

```bash
/Users/xiesheng/.pyenv/versions/3.11.9/bin/python drone/train_user.py
```

## 数据目录

采集数据将存放在 `data/`：
- `actionleft.txt`
- `actionright.txt`
- `rest.txt`

## 说明

- 需要连接 NeuroSky 设备并确保串口可用。
- 训练过程会在采集结束后自动开始。
- 模型保存为 `user_model.pth`。

## drone_control 使用指南

在项目根目录执行：

```bash
/Users/xiesheng/.pyenv/versions/3.11.9/bin/python drone/drone_control.py
```

运行前准备：
- 连接 Tello Wi-Fi。
- 确保 `user_model.pth` 位于项目根目录。
- 如需修改脑电串口，可设置环境变量 `MINDWAVE_PORT`。

```bash
初始化无人机并连接
初始化 NeuroSky 并开始读取

状态 = 垂直模式
方向模式 = 左右转
高度 = 0

循环:
    读取 attention / meditation / poorSignal / blinkStrength
    如果 poorSignal 过高: 提示信号弱，继续

    如果 状态 == 垂直模式:
        如果 attention 高: 起飞/上升
        如果 meditation 高: 下降/降落
        如果 blinkStrength 高: 切换到 水平模式

    如果 状态 == 水平模式:
        如果 方向模式 == 左右转:
            连续采样 5 次 -> 模型预测 left/right/rest
            多数 left -> 左转
            多数 right -> 右转
            多数 rest -> 不动
            如果 blinkStrength = 1 次: 切换到 前后模式
            如果 blinkStrength > 1 次: 切回 垂直模式
        否则(前后模式):
            attention 高 -> 前进
            meditation 高 -> 后退
            如果 blinkStrength 高: 切回 左右转模式
```