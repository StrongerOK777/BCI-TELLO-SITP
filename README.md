# 脑机接口项目（BCI-TELLO无人机部分）

这是一个同济大学本科生创新训练项目(SITP)。

本文包含了对于脑环的基础环境配置，测试连接，无人机的连接，使用脑环控制无人机三个部分。

# 目录（Contents）

- [环境配置](#env)
- [测试游戏使用](#spaceinvader)
- [仓库文件结构](#repo-structure)
- [基于运动想象控制TELLO无人机](#MI-Control)

<a name="env"></a>

## 你需要的环境配置：

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

  #### 安装命令：


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

## Github上已有小游戏的基本测试（Spaceinvaders）

### 使用指南:

#### 1.修改 **spaceinvaders.py** 程序中的

    ``Python     PORT1="COM3"     ``

    对于Linux和Macos用户可以使用以下命令，查看自己的USB连接端口，使用其中你觉得像串口的替换上面的COM3：

    ``bash     ls /dev/cu.*     ``

#### 2.在vscode终端使用python查看自己的python版本之后运行。

    当然，3.11版本运行的时候会调用**本文件夹**中的Neuropy.py程序，请务必**不要删除！** 但是2.7.版本中没有这方面考虑，请各位自己研究其中的原因:)。

<a name="MI-Control"></a>

## 基于运动想象控制TELLO无人机

### 文件概况：

  [KeyboardControl.py](./MI-DroneControl/KeyboardControl.py)：本使用键盘直接控制无人机移动的程序KeyboardControl.py文件。建议先使用这个程序测试成功无人机飞行状态无误后进行下一步操作。

  [单通道训练参考文献](./MI-DroneControl/Smart_Ward_Control_Based_on_a_Wearable_Multimodal_BrainComputer_Interface_Mouse.pdf)：较新期刊，[train_user.py](./MI-DroneControl/drone/train_user.py)的参考文献。

  [drone/](./MI-DroneControl/drone/)：这是整体无人机程序的所有使用的python文件

  data/：使用[train_user.py](./MI-DroneControl/drone/train_user.py)文件后生成的目录，用以存取用户的个人本次生成的原始数据文件。

  picture/：使用[train_user.py](./MI-DroneControl/drone/train_user.py)文件训练了用户的个性化模型之后生成的图形化图表目录。

  model/：使用[train_user.py](./MI-DroneControl/drone/train_user.py)文件训练的五种不同条件下的以及经过五种模型蒸馏得到的 **FinalModel.pth** 用户个性化模型。

### 使用指南：

  - #### 1.预处理：
  
    首先配置好所有文件的串口（这里没有做统一文件修改，很遗憾），使用[blinktest.py](./MI-DroneControl/drone/blinktest.py)文件测试连接是否正常，当稳定出现poorsignal==0的时候说明连接完美。
    
    尝试使用上眼睑眨眼，观察终端窗口变化的blinkstrength值是否发生变化，如果一直没有变化请继续尝试，知道出现明显的值为止（一般非自然眨眼大约为blinkstrength==140，仅作参考），关闭终端，进行下一步程序。

    打开[train_user.py](./MI-DroneControl/drone/train_user.py)文件，根据pygame敞口提示保持稳定静止状态或者挥动自己的相应侧肢体，通过不同侧运动实现**运动“想象”**。
    每次数据读取3秒，一共30组数据，这是为了实现信号的加权，中间的信号加权更大。读取过程中请时刻观察终端窗口是否报错poorsignal too high，如果报错将一直读取，程序不会自动停止。

    读取数据阶段大约耗时210秒，若pygme窗口自动关闭，请不要关闭终端窗口，程序将在后台进行模型的建立，逐渐生成个人模型，待终端提示 ll tasks completed successfully! 即可关闭，进行测试。
  
  - #### 2.模型自我测试：
    
    使用[predict_loop.py](./MI-DroneControl/drone/predict_loop.py)程序测试模型准确率，程序将会一直侦测当前状态，给出预测结果。算上安静状态，准确率大约在70%到80%之间。

  - #### 3.无人机的飞行：
  
    对于没有无人机或者想要虚拟测试的同学，可以使用[predic.py](./MI-DroneControl/drone/predic.py)程序进行测试，终端将会显示当前无人机行为，除了连接真实无人机部分，其他部分和[mergedrone.py](./MI-DroneControl/drone/mergedrone.py)程序一模一样，不用担心，连接无人机即可使用脑环给电脑发信号，电脑控制无人机。





<a name="repo-structure"></a>

- [README.md](./README.md)
- [MI-DroneControl/](./MI-DroneControl/)
  - [KeyPressModule.py](./MI-DroneControl/KeyPressModule.py)
  - [KeyboardControl.py](./MI-DroneControl/KeyboardControl.py)
  - [Smart_Ward_Control_Based_on_a_Wearable_Multimodal_BrainComputer_Interface_Mouse.pdf](./MI-DroneControl/Smart_Ward_Control_Based_on_a_Wearable_Multimodal_BrainComputer_Interface_Mouse.pdf)
  - [drone/](./MI-DroneControl/drone/)
    - [README.md](./MI-DroneControl/drone/README.md)
    - [blinktest.py](./MI-DroneControl/drone/blinktest.py)
    - [drone_control.py](./MI-DroneControl/drone/drone_control.py)
    - [game_1.py](./MI-DroneControl/drone/game_1.py)
    - [load_final_model.py](./MI-DroneControl/drone/load_final_model.py)
    - [mergedrone.py](./MI-DroneControl/drone/mergedrone.py)
    - [neuropy.py](./MI-DroneControl/drone/neuropy.py)
    - [predic.py](./MI-DroneControl/drone/predic.py)
    - [predict_loop.py](./MI-DroneControl/drone/predict_loop.py)
    - [train_user.py](./MI-DroneControl/drone/train_user.py)
    - [up_and_down.py](./MI-DroneControl/drone/up_and_down.py)
- [Spaceinvaders/](./Spaceinvaders/)
  - [Python2.7ver./](./Spaceinvaders/Python2.7ver./)
    - [README.md](./Spaceinvaders/Python2.7ver./README.md)
    - [spaceinvaders.py](./Spaceinvaders/Python2.7ver./spaceinvaders.py)
    - [test.py](./Spaceinvaders/Python2.7ver./test.py)
  - [Python3.11ver./](./Spaceinvaders/Python3.11ver./)
    - [NeuroPy.py](./Spaceinvaders/Python3.11ver./NeuroPy.py)
    - [diagnose_eeg.py](./Spaceinvaders/Python3.11ver./diagnose_eeg.py)
    - [spaceinvaders.py](./Spaceinvaders/Python3.11ver./spaceinvaders.py)