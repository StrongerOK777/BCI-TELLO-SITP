# 脑机接口项目（BCI-TELLO无人机部分）

这是一个同济大学本科生创新训练项目(SITP)。

本文包含了对于脑环的基础环境配置，测试连接，无人机的连接，使用脑环控制无人机三个部分。

# 目录（Contents）
- [环境配置](#env)
- [测试游戏使用](#spaceinvader)
- [仓库文件结构](#repo-structure)

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
  - 下面列举你需要的库（部分存在依赖关系，不需要全部手动安装）：

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
    pip                   25.3
    pygame                2.6.1
    pyparsing             3.3.1
    pyserial              3.5
    python-dateutil       2.9.0.post0
    pytz                  2025.2
    requests              2.32.5
    setuptools            65.5.0
    six                   1.17.0
    sympy                 1.14.0
    tellopy               0.6.0
    thread                2.0.6
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
    ```
  - 部分库如果下载不下来可以尝试使用代理或者使用清华镜像。

    对于使用Mac的同学我推荐**使用homebrew**优先**安装pyenv**，进行方便的python版本下载和管理。（使用Mac的同学应该使用过homebrew，不知道上网搜索即可）

    对于之前已经使用过homebrew安装python的同学，使用pyenv安装python之后记得修改系统的编译路径，二者是完全隔离的，之前下载过的大部分库不能再次使用。

<a name="spaceinvader"></a>

## Github上已有小游戏的基本测试（Spaceinvaders）

  ### 使用指南:

  - #### 1.修改 **spaceinvaders.py** 程序中的

    ```Python
    PORT1="COM3"
    ```

    对于Linux和Macos用户可以使用以下命令，查看自己的USB连接端口，使用其中你觉得像串口的替换上面的COM3：

    ```bash
    ls /dev/cu.*
    ```

  - #### 在vscode终端使用python查看自己的python版本之后运行。

    当然，3.11版本运行的时候会调用**本文件夹**中的Neuropy.py程序，请务必**不要删除！** 但是2.7.版本中没有这方面考虑，请各位自己研究其中的原因:)。

<a name="repo-structure"></a>
## 仓库文件结构

- [NeuroSky/](./NeuroSky/)
    - [Python2.7ver./](./NeuroSky/Python2.7ver./)
        - [README.md](./NeuroSky/Python2.7ver./README.md)
        - [spaceinvaders.py](./NeuroSky/Python2.7ver./spaceinvaders.py)
        - [test.py](./NeuroSky/Python2.7ver./test.py)
        - [使用说明.docx](./NeuroSky/Python2.7ver./使用说明.docx)
    - [Python3.11ver./](./NeuroSky/Python3.11ver./)
        - [NeuroPy.py](./NeuroSky/Python3.11ver./NeuroPy.py)
        - [spaceinvaders.py](./NeuroSky/Python3.11ver./spaceinvaders.py)
        - [diagnose_eeg.py](./NeuroSky/Python3.11ver./diagnose_eeg.py)
        - [使用说明.docx](./NeuroSky/Python3.11ver./使用说明.docx)
