# 脑机接口项目（BCI-TELLO无人机部分）

这是一个同济大学本科生创新训练项目(SITP)。

本文包含脑环基础环境配置、脑环接口配置、公共训练流程，以及小车、无人机、机械臂三个硬件的脑环控制程序说明。

# 目录（Contents）

- [环境配置](#env)
- [测试游戏使用](#spaceinvader)
- [脑环接口配置](#bci-interface)
- [公共训练与模型产物](#train-user)
- [硬件控制使用指南](#hardware-control)
- [仓库结构](#repo-structure)
- [后续计划](#future)

<a name="env"></a>

# 你需要的环境配置（Environment）：

- VSCode，[下载链接](https://code.visualstudio.com)
- VSCode插件：Python，Code Runner
- pyenv进行python版本管理

  ```bash
  brew install pyenv
  ```
- pip工具下载python所需的库

  ```bash
  brew install pip
  ```
- python 2.7.15，python 3.11.9（我将会分别提供两个版本以适配不同版本的python）

  下面列举你需要的库（部分存在依赖关系，不需要全部手动安装）：

  <details>
    <summary>点击展开/折叠库目录</summary>
    <pre><code class="language-bash">
  Package               Version
  --------------------- -----------
  av                    16.1.0
  certifi               2026.1.4
  charset-normalizer    3.4.4
  contourpy             1.3.3
  cycler                0.12.1
  DateTime              6.0
  djitellopy            2.5.0
  filelock              3.20.3
  fonttools             4.61.1
  fsspec                2026.1.0
  idna                  3.11
  Jinja2                3.1.6
  joblib                1.5.3
  kiwisolver            1.4.9
  MarkupSafe            3.0.3
  matplotlib            3.10.8
  mpmath                1.3.0
  networkx              3.6.1
  NeuroPy               0.1
  numpy                 1.26.4
  opencv-contrib-python 4.8.1.78
  opencv-python         4.8.1.78
  packaging             25.0
  pandas                2.3.3
  pillow                12.1.0
  pip                   26.0
  pygame                2.6.1
  pyparsing             3.3.1
  pyserial              3.5
  python-dateutil       2.9.0.post0
  pytz                  2025.2
  requests              2.32.5
  scikit-learn          1.8.0
  scipy                 1.17.0
  setuptools            65.5.0
  six                   1.17.0
  sympy                 1.14.0
  tellopy               0.6.0
  thread                2.0.6
  threadpoolctl         3.6.0
  torch                 2.2.2
  typing_extensions     4.15.0
  tzdata                2025.3
  urllib3               2.6.3
  zope.interface        8.2
  </code></pre> </details>

  ### 安装命令：


  ```bash
  pip install numpy pandas matplotlib
  pip install opencv-python opencv-contrib-python pillow
  pip install torch pygame
  pip install requests networkx sympy
  pip install av djitellopy tellopy NeuroPy
  pip install scikit-learn
  ```
- 部分库如果下载不下来可以尝试使用代理或者使用清华镜像。

  对于使用Mac的同学我推荐**使用homebrew**优先**安装pyenv**，进行方便的python版本下载和管理。（使用Mac的同学应该使用过homebrew，不知道上网搜索即可）

  对于之前已经使用过homebrew安装python的同学，使用pyenv安装python之后记得修改系统的编译路径，二者是完全隔离的，之前下载过的大部分库不能再次使用。

<a name="spaceinvader"></a>

# Github上已有小游戏的基本测试（Spaceinvaders）

## 使用指南:

### 1.修改 **spaceinvaders.py** 程序中的

    ``Python     PORT1="COM3"     ``

    对于Linux和Macos用户可以使用以下命令，查看自己的USB连接端口，使用其中你觉得像串口的替换上面的COM3：

    ``bash     ls /dev/cu.*     ``

### 2.在vscode终端使用python查看自己的python版本之后运行。

    当然，3.11版本运行的时候会调用**本文件夹**中的Neuropy.py程序，请务必**不要删除！** 但是2.7.版本中没有这方面考虑，请各位自己研究其中的原因:)。

<a name="bci-interface"></a>

# 脑环接口配置（BCI Interface）

电脑和脑环之间的串口连接统一写在根目录的 [bci_interface.py](./bci_interface.py) 中。更换电脑、USB 口或串口号时，只需要修改这个文件，训练程序和三个硬件的 `brain_control.py` 都会自动复用。

## 文件用途：

- `MINDWAVE_PORT`：脑环串口，例如 Windows 下的 `COM5`，或 macOS 下的 `/dev/cu.usbmodem2017_2_251`。
- `MINDWAVE_BAUD`：脑环波特率，默认 `57600`。
- `NEUROPY_DIR`：项目中 `neuropy.py` 驱动所在目录，默认指向 `TrainUser/`。
- `get_mindwave_interface()`：统一返回串口、波特率和驱动目录，供其他程序调用。

示例：

```python
MINDWAVE_PORT = "COM5"                    # Windows 示例
MINDWAVE_PORT = "/dev/cu.usbmodem2017_2_251"  # macOS 示例
MINDWAVE_BAUD = 57600
```

<a name="train-user"></a>

# 公共训练与模型产物（TrainUser / src）

现在脑环训练是全项目通用的。只需要训练一次，生成的模型即可被小车、无人机和机械臂共同使用。

## 文件概况：

- [TrainUser/train_user.py](./TrainUser/train_user.py)：公共训练主程序，负责采集左手、右手、静息三类数据，并训练最终模型。
- [TrainUser/neuropy.py](./TrainUser/neuropy.py)：NeuroSky / MindWave 脑环驱动。
- [TrainUser/README.md](./TrainUser/README.md)：训练目录说明。
- [src/data/](./src/data/)：训练得到的原始数据文件。
- [src/models/](./src/models/)：训练得到的模型文件，其中 `FinalModel.pth` 是三个硬件默认使用的最终模型。
- [src/picture/](./src/picture/)：训练过程产生的图表。

## 使用指南：

### 1. 修改脑环接口

先在 [bci_interface.py](./bci_interface.py) 中确认串口配置正确。

### 2. 开始训练

在项目根目录运行：

```bash
python TrainUser/train_user.py
```

训练完成后会生成：

```text
src/data/actionleft.txt
src/data/actionright.txt
src/data/rest.txt
src/models/FinalModel.pth
src/picture/final_model_training.png
```

### 3. 复用模型

三个硬件的脑控程序默认都会读取：

```text
src/models/FinalModel.pth
```

如果需要临时指定其他模型，可以使用：

```bash
--model-path <模型路径>
```

<a name="hardware-control"></a>

# 硬件控制使用指南（Hardware Control）

<a name="MI-Control"></a>

## 1. 小车控制（MI-CarControl）

## 文件概况：

- [keyboard_control.py](./MI-CarControl/keyboard_control.py)：键盘控制小车，通过 HTTP 向小车发送前进、后退、左转、右转和停止指令。
- [brain_control.py](./MI-CarControl/brain_control.py)：脑环控制小车，默认使用 `src/models/FinalModel.pth`。
- [neuropy.py](./MI-CarControl/neuropy.py)：小车目录保留的脑环驱动副本。
- [README.md](./MI-CarControl/README.md)：小车目录说明。

## 使用指南：

键盘控制：

```bash
python MI-CarControl/keyboard_control.py --host 192.168.149.1 --port 5000
```

脑环控制：

```bash
python MI-CarControl/brain_control.py
```

小车脑控逻辑：默认进入前后模式；双眨眼切换前后模式和转向模式；前后模式通过注意力/冥想判断前进或后退，转向模式通过模型预测 `left / right / rest` 判断左转、右转或停止。

## 2. TELLO 无人机控制（MI-DroneControl）

## 文件概况：

- [keyboard_control.py](./MI-DroneControl/keyboard_control.py)：键盘控制 Tello，并显示摄像头画面。
- [brain_control.py](./MI-DroneControl/brain_control.py)：脑环控制 Tello，无人机动作由规则信号和共享模型共同决定。
- [predict.py](./MI-DroneControl/predict.py)：连续读取脑环窗口，并输出模型预测结果。
- [diagnose.py](./MI-DroneControl/diagnose.py)：脑环信号诊断程序，用于查看 `attention`、`meditation`、`blinkStrength`、`poorSignal` 等状态。
- [drone_hardware.py](./MI-DroneControl/drone_hardware.py)：Tello 硬件适配层。
- [neuropy.py](./MI-DroneControl/neuropy.py)：无人机目录保留的脑环驱动副本。
- [README.md](./MI-DroneControl/README.md)：无人机目录说明。

## 使用指南：

诊断脑环信号：

```bash
python MI-DroneControl/diagnose.py
```

查看模型预测：

```bash
python MI-DroneControl/predict.py
```

键盘控制无人机：

```bash
python MI-DroneControl/keyboard_control.py
```

脑环控制无人机：

```bash
python MI-DroneControl/brain_control.py
```

## 3. 机械臂控制（MI-DOFBOT）

## 文件概况：

- [keyboard_control.py](./MI-DOFBOT/keyboard_control.py)：键盘控制 DOFBOT 机械臂。
- [brain_control.py](./MI-DOFBOT/brain_control.py)：脑环控制 DOFBOT 机械臂，默认使用共享模型。
- [arm_hardware.py](./MI-DOFBOT/arm_hardware.py)：DOFBOT 硬件适配层。
- [Arm_Lib (Windows)/](./MI-DOFBOT/Arm_Lib%20(Windows)/)：机械臂底层串口库。
- [readme.md](./MI-DOFBOT/readme.md)：机械臂目录说明。

## 使用指南：

键盘控制建议先 dry-run：

```bash
python MI-DOFBOT/keyboard_control.py --dry-run
```

脑环控制首次也建议 dry-run：

```bash
python MI-DOFBOT/brain_control.py --dry-run
```

确认信号和动作逻辑正常后，再连接真实机械臂并指定串口：

```bash
python MI-DOFBOT/brain_control.py --arm-port COM4
```

<a name="future"></a>

# 后续计划（Future Plan）：

- 保持 `bci_interface.py` 作为统一脑环接口，避免串口配置散落在各个程序中。
- 继续优化 `TrainUser/` 的训练流程，让一次训练得到的模型可以更稳定地服务三类硬件。
- 后续可以在现有架构上继续开发 SSVEP 控制方式。

<a name="repo-structure"></a>

# 仓库结构（Repository Structure）

```text
BCI-TELLO/
├── README.md                         # 项目总说明
├── bci_interface.py                  # 电脑与脑环通信接口配置
├── TrainUser/                        # 公共脑环训练程序
│   ├── train_user.py                 # 采集数据并训练共享模型
│   ├── neuropy.py                    # NeuroSky / MindWave 驱动
│   └── README.md
├── src/                              # 公共训练产物
│   ├── data/                         # 训练数据
│   ├── models/                       # 共享模型，默认 FinalModel.pth
│   ├── picture/                      # 训练图表
│   └── README.md
├── MI-CarControl/                    # 小车控制
│   ├── keyboard_control.py           # 小车键盘控制
│   ├── brain_control.py              # 小车脑环控制
│   ├── neuropy.py
│   └── README.md
├── MI-DroneControl/                  # Tello 无人机控制
│   ├── keyboard_control.py           # 无人机键盘控制
│   ├── brain_control.py              # 无人机脑环控制
│   ├── predict.py                    # 模型预测测试
│   ├── diagnose.py                   # 脑环信号诊断
│   ├── drone_hardware.py             # Tello 硬件适配
│   ├── neuropy.py
│   └── README.md
├── MI-DOFBOT/                        # DOFBOT 机械臂控制
│   ├── keyboard_control.py           # 机械臂键盘控制
│   ├── brain_control.py              # 机械臂脑环控制
│   ├── arm_hardware.py               # 机械臂硬件适配
│   ├── Arm_Lib (Windows)/            # 底层机械臂库
│   └── readme.md
├── bin/                              # 可迁移基础模块
│   ├── eeg.py                        # EEG 采集和特征构建
│   ├── models.py                     # 模型结构和预测工具
│   ├── training.py                   # 可复用训练工具
│   ├── keyboard.py                   # 通用键盘读取
│   └── transport.py                  # 通用 JSON HTTP 传输
├── Spaceinvaders/                    # 脑环测试游戏和历史实验代码
└── tests/                            # 自动化测试
```
