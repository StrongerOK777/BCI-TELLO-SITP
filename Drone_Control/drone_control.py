import os
import sys
import numpy as np
import torch
import torch.nn as nn#nn.Module来自于pytorch，帮助搭建精神网络模块、管理每层参数以及模型训练
import torch.optim as optim#模型优化器
import time
from neuropy import NeuroSkyPy
from djitellopy import tello#无人机
from time import sleep
me = tello.Tello()
"""
此为【调用训练好的模型】控制无人机
无人机运动流程：把无人机放远后读入数据后（假眨眼）开启无人机，
先进入垂直操作模式，attention>50起飞，眨眼一次后进入水平模式，
水平模式先前后移动。在一次眨眼【此处有改动】进入转弯模式，
转弯模式【此处有改动】眨眼一次就转为前后移动，多次眨眼就准备降落。
只有进入转弯模式再眨眼才能降落。
改动1：完善了提示指引，并且添加了无人机上升下降功能
改动2：把原来输出六维改为了三维：左右和休息（眨眼不需要模型学习）"""
# 在未连接到 Tello Wi-Fi 时直接退出，避免报错导致程序中断
try:
    me.connect()
except Exception as exc:
    print("无人机未响应，请先连接 Tello Wi-Fi 后重试：", exc)
    sys.exit(1)

print("无人机连接成功！")
print(f"无人机目前电量：{me.get_battery()}")#链接无人机打印电量

class ImprovedModel(nn.Module):#子类ImprovedModel对父类nn.Module的提升，定义模型的神经网络
    def __init__(self):
        super(ImprovedModel, self).__init__()#对父类进行初始化，super()在子类中=父类，super().函数就意味着调用父类函数
        #传入ImprovedModel是指定从哪个子类找对应父类，self是指定实例

        #对子类ImprovedModel进行特殊初始化，即往nn.module上搭建神经网络层
        self.fc1 = nn.Linear(12, 256)  # 隐藏层，输入12个特征输出256个神经元，提高隐藏层节点数
        self.bn1 = nn.BatchNorm1d(256)  # 加入BN层，特征标准化
        self.relu = nn.ReLU()#让网络学习非线性关系
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 3)#决定最后输出的是六种特征信号。

    #数据依次进入每个层进行处理，设定数据的流向
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


def print_prediction(input):#调用已训练的模型进行工作
    model = ImprovedModel() # 请确保模型的结构与训练时相同
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用与训练时相同的优化器，parameters传入所有待训练的参数
    #lr为学习率，即每次学习后根据结果调整参数的步长，小则慢，大则可能错过最佳点

    # 指定加载的模型文件路径
    load_path = 'user_model.pth'#指定寻找模型文件的路径
    checkpoint = torch.load(load_path)#加载模型的各项指标，如训练轮数和损失率
    model.load_state_dict(checkpoint['model_state_dict'])#加载模型各项参数

    #把输入数据转化为模型可使用的格式
    input_data = np.array(input, dtype=float)
    print("输入数据：", input_data)
    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # 把数据又转为模型可用的pytorch方量格式
    # 并且Add a batch dimension

    # Set the model to evaluation mode and make predictions
    model.eval()#设置评估模式
    with torch.no_grad():#关闭梯度计算，即不再调整参数
        # Make predictions
        output = model(input_data)
        predicted_class = torch.argmax(output, dim=1)#比较三个结果的得分，把最高分对应项赋值
        labelmap = ['left', 'right', 'rest']#给三个结果赋名

        print(f"预测结果为：{labelmap[predicted_class.item()]}")#即打印得分最高对应的名字即最终判断结果，item（）取索引号。
    return predicted_class.item()#返回判断结果


MINDWAVE_PORT = os.getenv("MINDWAVE_PORT", "/dev/cu.usbmodem2017_2_251")
MINDWAVE_BAUD = int(os.getenv("MINDWAVE_BAUD", "57600"))
neuropy = NeuroSkyPy(MINDWAVE_PORT, MINDWAVE_BAUD)
neuropy.start()#开始接收脑电信号
inputdata = []#建立inputdata空表
pre = []
up_down = True#即先开启上升下降模式
around = False#即水平模式，希望先执行上升下降模式再执行水平模式
height = 0
left_right = True#即左转右转模式开启
try:
    while True:
        # 读取数据
        attention = neuropy.attention
        meditation = neuropy.meditation#冥想度，精神放松程度
        delta = neuropy.delta#深度睡眠时脑波
        theta = neuropy.theta#浅度睡眠，创新力
        lowAlpha = neuropy.lowAlpha
        highAlpha = neuropy.highAlpha#放松度检测
        lowBeta = neuropy.lowBeta
        highBeta = neuropy.highBeta#注意力集中程度
        lowGamma = neuropy.lowGamma
        midGamma = neuropy.midGamma
        poorSignal = neuropy.poorSignal#信号质量，0表示最好
        blinkStrength = neuropy.blinkStrength#眨眼强度
        inputdata = [attention, meditation, delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma,
                     midGamma, poorSignal, blinkStrength]

        if poorSignal <20 and up_down == True:#先做到上升下降，再进入水平模式 #把所有poorsignal信号都改成小于了了。
            print("检测到眨眼，可以通过注意力大小来控制直升机的上升和下降：\n注意力高则起飞，冥想高则降落，眨眼切换水平模式")#并没有眨眼。但是起飞吧。
            time.sleep(1)
            while True:
                attention = neuropy.attention
                poorSignal = neuropy.poorSignal
                meditation = neuropy.meditation
                print(f"{attention}")#f意味着可以直接再字符串里插入变量打印{parametername}
                if attention > 50:
                    if height == 0:
                        height = 100
                        print(f"无人机起飞！{attention}")
                        me.takeoff()
                    elif height > 0 and height < 4:
                        print("无人机上升！")#添加了上升功能
                        me.move_up(30)
                        height += 0.3
                    else:
                        print("无人机已到达最大高度！")
                elif meditation > 50:#阈值太高了改成50
                    if height > 1:#添加了下降功能
                        print("无人机下降！")
                        me.move_down(30)
                        height -= 0.3
                    elif height == 0:
                        print("无人机已降落！")
                    else:
                        print("无人机降落！")
                        height=0#是不是漏了这句？
                        me.land()
                if blinkStrength > 30:#这里是不是也写错了？改成blinkStrengh
                    print("检测到眨眼，已切换为水平模式")
                    up_down = False
                    around = True
                    break
                sleep(1)

        if poorSignal < 20 and around == True:#要求无人机距离远且为水平模式，进入水平模式
            print("检测到眨眼，进入转弯模式，想象左右肢体来控制无人机方向：\n眨眼一次则退出转弯模式进入前进后退模式，眨眼多次则退出水平模式准备降落")
            time.sleep(2)
            leftnum = 0
            rightnum = 0
            restnum = 0
            blinknum = 0
            if left_right == True:#进入左转右转模式
                for i in range(5):

                    attention = neuropy.attention
                    meditation = neuropy.meditation
                    delta = neuropy.delta
                    theta = neuropy.theta
                    lowAlpha = neuropy.lowAlpha
                    highAlpha = neuropy.highAlpha
                    lowBeta = neuropy.lowBeta
                    highBeta = neuropy.highBeta
                    lowGamma = neuropy.lowGamma
                    midGamma = neuropy.midGamma
                    poorSignal = neuropy.poorSignal
                    blinkStrength = neuropy.blinkStrength
                    inputdata = [attention, meditation, delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma,
                                 midGamma, poorSignal, blinkStrength]
                    pre = print_prediction(inputdata)#根据模型给出的判断结果开始进行转弯
                    if pre == 1 :#1对应的是右手右腿
                        rightnum += 1
                        print("右转计数+1")
                    if pre == 0:#0对应左手左腿
                        leftnum += 1
                        print("左转计数+1")
                    if pre == 2:#对应发呆
                        restnum += 1
                        print("休息计数+1")
                    if blinkStrength > 30:#疑似原文写错了，改成了blinkStrength
                        blinknum += 1
                        print("眨眼计数+1")

                    if i == 4:#采样五次再对无人机进行实际操作
                        if blinknum == 1:
                            print("检测到一次眨眼，关闭方向控制重新进入前进后退模式")
                            left_right = False
                            break
                        elif blinknum > 1:
                            print("检测到多次眨眼，切换为控制垂直方向")
                            around = False
                            up_down = True#只有进入垂直方向才有可能降落。。
                            break#新加的
                        elif restnum > 3:#一直不动就会休息
                            print("多数时间是休息状态，rest")
                        elif rightnum > leftnum:
                            print("无人机右转")
                            me.send_rc_control(0, 0, 0, 45)
                        elif rightnum <= leftnum:#改成了elif
                            print("无人机左转")
                            me.send_rc_control(0, 0, 0, -45)
                    time.sleep(1)
            else:#水平模式下不左转右转就进入前进后退模式
                print("进入水平模式，注意力高则前进，冥想高则后退，眨眼则控制方向")
                attention = neuropy.attention
                meditation = neuropy.meditation
                poorSignal = neuropy.poorSignal
                blinkStrength=neuropy.blinkStrength
                if blinkStrength > 30:#原来用poorsignal太怪了，换成眨眼
                    print("检测到一次眨眼，前进后退关闭，控制方向开启！")
                    left_right = True
                elif attention > 40:#原80阈值太高改成了40
                    me.send_rc_control(0, 30, 0, 0)
                    print("前进30cm")
                elif meditation > 40:#原80阈值太高改成了40
                    me.send_rc_control(0, -30, 0, 0)
                    print("后退30cm")
        if poorSignal>20:
            print("请调整脑电装置，信号过弱")#新加的
        time.sleep(1)#每秒读取一次数据




except KeyboardInterrupt:
    pass#不让ctrlc强制退出程序
print("无人机程序已退出")
neuropy.stop()

