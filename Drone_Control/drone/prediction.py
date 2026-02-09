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
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(ImprovedModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

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
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict")

    fc1_weight = state_dict.get("fc1.weight")
    fc2_weight = state_dict.get("fc2.weight")
    fc3_weight = state_dict.get("fc3.weight")
    if fc1_weight is None or fc2_weight is None or fc3_weight is None:
        raise ValueError("Checkpoint missing fc1/fc2/fc3 weights")

    input_dim = checkpoint.get("input_dim") or fc1_weight.shape[1]
    hidden_dim1 = fc1_weight.shape[0]
    hidden_dim2 = fc2_weight.shape[0]
    num_classes = fc3_weight.shape[0]

    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")

    model = ImprovedModel(input_dim, hidden_dim1, hidden_dim2, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model, feature_mean, feature_std, num_classes


def print_prediction(model, feature_mean, feature_std, window, labelmap):
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
model_path = os.getenv("MODEL_PATH", "user_model.pth")
if not os.path.exists(model_path):
    fallback_path = "model_1_wide_model.pth"
    if os.path.exists(fallback_path):
        print(f"未找到 {model_path}，使用 {fallback_path} 进行预测")
        model_path = fallback_path
    else:
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

model, feature_mean, feature_std, num_classes = load_model_checkpoint(model_path)
if num_classes != 3:
    raise ValueError(f"模型输出类别数为 {num_classes}，与 left/right/rest 不一致")
feature_window = []
labelmap = ["left", "right", "rest"]
window_size = 30
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
                        print("无人机上升3cm！")#添加了上升功能
                        height += 0.3
                    else:
                        print("无人机已到达最大高度！")
                elif meditation > 50:#阈值太高了改成50
                    if height > 1:#添加了下降功能
                        print("无人机下降3cm！")
                        height -= 0.3
                    elif height == 0:
                        print("无人机已降落！")
                    else:
                        print("无人机降落！")
                        height=0#是不是漏了这句？
                if blinkStrength > 30:#这里是不是也写错了？改成blinkStrengh
                    print("检测到眨眼，已切换为水平模式")
                    time.sleep(3)
                    Mode = 1
                    break
                time.sleep(0.1)

        print("三秒后，进入水平模式")
        time.sleep(3)

        if poorSignal < 20 and Mode == 1:#要求无人机距离远且为水平模式，进入水平模式
            left_right = True#即左转右转模式开启
            while True:
                #print("检测到眨眼，进入转弯模式，想象左右肢体来控制无人机方向：\n眨眼一次则退出转弯模式进入前进后退模式，眨眼多次则退出水平模式准备降落")
                if left_right == True:#进入左转右转模式leftnum = 0
                    leftnum = 0
                    rightnum = 0
                    restnum = 0
                    blinknum = 0
                    feature_window = []




                    #############
                    # for i in range(30):
                    #     attention = neuropy.attention
                    #     meditation = neuropy.meditation
                    #     delta = neuropy.delta
                    #     theta = neuropy.theta
                    #     lowAlpha = neuropy.lowAlpha
                    #     highAlpha = neuropy.highAlpha
                    #     lowBeta = neuropy.lowBeta
                    #     highBeta = neuropy.highBeta
                    #     lowGamma = neuropy.lowGamma
                    #     midGamma = neuropy.midGamma
                    #     poorSignal = neuropy.poorSignal
                    #     blinkStrength = neuropy.blinkStrength
                    #     feature_vector = build_feature_vector(
                    #         attention, meditation, delta, 
                    #         theta, lowAlpha, highAlpha,lowBeta, highBeta, 
                    #         lowGamma, midGamma, blinkStrength
                    #     )
                    #     if poorSignal >= 20:
                    #         print(f"PoorSignal too high: {poorSignal}, skipping...")
                    #         time.sleep(0.1)
                    #         continue
                    #     feature_window.append(feature_vector)
                    #     if len(feature_window) > window_size:
                    #         feature_window.pop(0)
                    #     if len(feature_window) >= window_size:
                    #         pre = print_prediction(model, feature_mean, feature_std, feature_window)
                    #     else:
                    #         pre = 2
                    #     if pre == 1 :#1对应的是右手右腿
                    #         rightnum += 1
                    #         print("右转计数+1")
                    #     if pre == 0:#0对应左手左腿
                    #         leftnum += 1
                    #         print("左转计数+1")
                    #     if pre == 2:#对应发呆
                    #         restnum += 1
                    #         print("休息计数+1")
                    #     if blinkStrength > 100:
                    #         blinknum += 1
                    #         print("眨眼计数+1")
                    #     time.sleep(0.1)
                    ################
                    start=time.time()
                    for i in range(30):
                        attention = neuropy.attention or 0
                        meditation = neuropy.meditation or 0
                        delta = neuropy.delta or 0
                        theta = neuropy.theta or 0
                        low_alpha = neuropy.lowAlpha or 0
                        high_alpha = neuropy.highAlpha or 0
                        low_beta = neuropy.lowBeta or 0
                        high_beta = neuropy.highBeta or 0
                        low_gamma = neuropy.lowGamma or 0
                        mid_gamma = neuropy.midGamma or 0
                        blink_strength = neuropy.blinkStrength or 0
                        poorSignal = neuropy.poorSignal or 0
                        if(poorSignal >= 20 or attention ==0 or meditation == 0):
                            print(f"PoorSignal too high: {poorSignal}, reset window...")
                            feature_window.clear()
                            start = time.time()
                            time.sleep(1)
                            continue
                        if(blink_strength > 100):
                            blinknum += 1
                            print("眨眼计数+1")
                        feature_vector = build_feature_vector(
                            attention, meditation, delta, theta, low_alpha, high_alpha,
                            low_beta, high_beta, low_gamma, mid_gamma, blink_strength
                        )
                        feature_window.append(feature_vector)
                        print(f"当前读取数据：attention={attention} meditation={meditation} blinkStrength={neuropy.blinkStrength}")
                        target_time = start + (i + 1) * 0.1
                        sleep_time = target_time - time.time()
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                    window_data = np.array(feature_window, dtype=np.float32)
                    mean_feat = window_data.mean(axis=0)
                    std_feat = window_data.std(axis=0)
                    combined = np.concatenate([mean_feat, std_feat], axis=0)

                    if feature_mean is not None and feature_std is not None:
                        combined = (combined - feature_mean) / (feature_std + 1e-6)

                    input_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        predicted = torch.argmax(outputs, dim=1).item()

                    print(f"预测结果：{labelmap[predicted]} | 1秒/10组完成")
                    if blinknum == 1:
                        print("一次眨眼，进入前后模式")
                        time.sleep(2.3)
                        left_right = False
                    elif blinknum > 1:
                        print("两次及以上眨眼，切换为升降模式")
                        time.sleep(2.3)
                        Mode = 0#只有进入垂直方向才有可能降落。
                        break
                    elif restnum > 16:#一直不动就会休息
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
                    if poorSignal >= 20:
                        print(f"PoorSignal too high: {poorSignal}, skipping...")
                        time.sleep(0.1)
                        continue
                    if blinkStrength > 100:#原来用poorsignal太怪了，换成眨眼
                        print("检测到一次眨眼，前进后退关闭，控制方向开启！")
                        time.sleep(2.3)
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

