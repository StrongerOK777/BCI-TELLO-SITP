import os
import sys
import numpy as np
import torch
import torch.nn as nn#nn.Module来自于pytorch，帮助搭建精神网络模块、管理每层参数以及模型训练
import torch.optim as optim#模型优化器
import time
from neuropy import NeuroSkyPy
from time import sleep
"""
此为【调用训练好的模型】控制无人机
无人机运动流程：把无人机放远后读入数据后（假眨眼）开启无人机，
先进入垂直操作模式，attention>50起飞，眨眼一次后进入水平模式，
水平模式先前后移动。在一次眨眼【此处有改动】进入转弯模式，
转弯模式【此处有改动】眨眼一次就转为前后移动，多次眨眼就准备降落。
只有进入转弯模式再眨眼才能降落。
改动1：完善了提示指引，并且添加了无人机上升下降功能
改动2：把原来输出六维改为了三维：左右和休息（眨眼不需要模型学习）"""


class ImprovedModel(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def build_feature_vector(attention, meditation, delta, theta, low_alpha, high_alpha,
                         low_beta, high_beta, low_gamma, mid_gamma, blink_strength):
    beta = low_beta + high_beta
    alpha = low_alpha + high_alpha
    theta_safe = theta if theta != 0 else 1e-6
    beta_theta_ratio = beta / theta_safe
    alpha_theta_ratio = alpha / theta_safe
    engagement = beta / (alpha + 1e-6)
    return [
        attention, meditation, delta, theta, low_alpha, high_alpha,
        low_beta, high_beta, low_gamma, mid_gamma, blink_strength,
        beta_theta_ratio, alpha_theta_ratio, engagement
    ]


def load_model_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    input_dim = checkpoint.get("input_dim")
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    model = ImprovedModel(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, feature_mean, feature_std


def print_prediction(model, feature_mean, feature_std, window):
    window_data = np.array(window, dtype=np.float32)
    mean_feat = window_data.mean(axis=0)
    std_feat = window_data.std(axis=0)
    combined = np.concatenate([mean_feat, std_feat], axis=0)
    if feature_mean is not None and feature_std is not None:
        combined = (combined - feature_mean) / (feature_std + 1e-6)

    input_data = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_data)
        predicted_class = torch.argmax(output, dim=1)
        labelmap = ['left', 'right', 'rest']
        print(f"预测结果为：{labelmap[predicted_class.item()]}")
    return predicted_class.item()


MINDWAVE_PORT = os.getenv("MINDWAVE_PORT", "/dev/cu.usbmodem2017_2_251")
MINDWAVE_BAUD = int(os.getenv("MINDWAVE_BAUD", "57600"))
neuropy = NeuroSkyPy(MINDWAVE_PORT, MINDWAVE_BAUD)
neuropy.start()#开始接收脑电信号
inputdata = []#建立inputdata空表
pre = []
Mode = 0 # 0为升降，1为水平，2为转弯
height = 0
left_right = True#即左转右转模式开启
model, feature_mean, feature_std = load_model_checkpoint('user_model.pth')
feature_window = []
window_size = 5
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
        feature_vector = build_feature_vector(
            attention, meditation, delta, theta, lowAlpha, highAlpha,
            lowBeta, highBeta, lowGamma, midGamma, blinkStrength
        )
        feature_window.append(feature_vector)
        if len(feature_window) > window_size:
            feature_window.pop(0)
        if poorSignal < 20 and Mode == 0:#先做到上升下降，再进入水平模式 #把所有poorsignal信号都改成小于了。
            print("Blink:升降启动，检测到眨眼\n")#并没有眨眼。但是起飞吧。
            time.sleep(1)
            while True:
                attention = neuropy.attention
                poorSignal = neuropy.poorSignal
                meditation = neuropy.meditation
                blinkStrength = neuropy.blinkStrength#眨眼强度
                print(f"当前读取数据：attention={attention} meditation={meditation} poorSignal={poorSignal} blinkStrength={blinkStrength}")
                print(f"{attention}")#f意味着可以直接再字符串里插入变量打印{parametername}
                if attention > 50:
                    if height == 0:
                        height = 100
                        print(f"无人机起飞！{attention}")
                    elif height > 0 and height < 4:
                        print("无人机上升30cm！")#添加了上升功能
                        height += 0.3
                    else:
                        print("无人机已到达最大高度！")
                elif meditation > 50:#阈值太高了改成50
                    if height > 1:#添加了下降功能
                        print("无人机下降30cm！")
                        height -= 0.3
                    elif height == 0:
                        print("无人机已降落！")
                    else:
                        print("无人机降落！")
                        height=0#是不是漏了这句？
                if blinkStrength > 30:#这里是不是也写错了？改成blinkStrengh
                    print("检测到眨眼，已切换为水平模式")
                    Mode = 1
                    break
                sleep(1)
        print("三秒后，进入水平模式")
        sleep(3)
        if poorSignal < 20 and Mode == 1:#要求无人机距离远且为水平模式，进入水平模式
            left_right = True#即左转右转模式开启
            while True:
                #print("检测到眨眼，进入转弯模式，想象左右肢体来控制无人机方向：\n眨眼一次则退出转弯模式进入前进后退模式，眨眼多次则退出水平模式准备降落")
                if left_right == True:#进入左转右转模式leftnum = 0
                    rightnum = 0
                    restnum = 0
                    blinknum = 0
                    for i in range(50):
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
                        if len(feature_window) >= window_size:
                            pre = print_prediction(model, feature_mean, feature_std, feature_window)
                        else:
                            pre = 2
                        if pre == 1 :#1对应的是右手右腿
                            rightnum += 1
                            print("右转计数+1")
                        if pre == 0:#0对应左手左腿
                            leftnum += 1
                            print("左转计数+1")
                        if pre == 2:#对应发呆
                            restnum += 1
                            print("休息计数+1")
                        if blinkStrength > 130:#疑似原文写错了，改成了blinkStrength
                            blinknum += 1
                            print("眨眼计数+1")
                        time.sleep(0.1)
                        
                    if blinknum == 2:
                        print("两次眨眼，进入前后模式")
                        left_right = False
                    elif blinknum > 2:
                        print("三次及以上眨眼，切换为升降模式")
                        Mode = 0#只有进入垂直方向才有可能降落。
                        break
                    elif restnum > 30:#一直不动就会休息
                        print("多数时间是休息状态，rest")
                    elif rightnum > leftnum:
                        print("无人机右转")
                    elif rightnum <= leftnum:#改成了elif
                        print("无人机左转")
                        time.sleep(1)

                else:#水平模式下不左转右转就进入前进后退模式
                    print("前后模式，注意力高则前进，冥想高则后退，眨眼则控制方向")
                    print(f"当前读取数据：attention={attention} meditation={meditation} poorSignal={poorSignal} blinkStrength={blinkStrength}")
                    attention = neuropy.attention
                    meditation = neuropy.meditation
                    poorSignal = neuropy.poorSignal
                    blinkStrength=neuropy.blinkStrength
                    if blinkStrength > 150:#原来用poorsignal太怪了，换成眨眼
                        print("检测到一次眨眼，前进后退关闭，控制方向开启！")
                        left_right = True
                    elif attention > 40:#原80阈值太高改成了40
                        print("前进3cm")
                    elif meditation > 40:#原80阈值太高改成了40
                        print("后退3cm")
                time.sleep(0.1);
        time.sleep(0.5)#每秒读取一次数据

except KeyboardInterrupt:
    pass#不让ctrlc强制退出程序
neuropy.stop()

