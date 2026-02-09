import os
import sys
import numpy as np
import torch
import torch.nn as nn#nn.Module来自于pytorch，帮助搭建精神网络模块、管理每层参数以及模型训练
import torch.optim as optim#模型优化器
import torch.nn.functional as F
import time
from neuropy import NeuroSkyPy
from time import sleep
# from djitellopy import tello#无人机
# me = tello.Tello()

class FinalUnifiedModel(nn.Module):
    def __init__(self, input_dim):
        super(FinalUnifiedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 384)
        self.bn1 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(0.12)
        self.fc2 = nn.Linear(384, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.15)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.18)
        self.fc5 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        identity = x
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = x + identity
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
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


def load_model(checkpoint_path="FinalModel.pth"):
    if not os.path.isabs(checkpoint_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print (f"Loading model from: {checkpoint_path} (resolved to {os.path.join(base_dir, '..', 'model', checkpoint_path)})")
        model_dir = os.path.join(base_dir, "..", "model")
        checkpoint_path = os.path.join(model_dir, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FinalUnifiedModel(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    print(f"Model loaded successfully!")
    print(f"Test Accuracy (strict): {checkpoint['test_accuracy_strict']*100:.2f}%")
    print(f"Test Accuracy (relaxed): {checkpoint['test_accuracy_relaxed']*100:.2f}%")
    return model, feature_mean, feature_std

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


def print_prediction(model, feature_mean, feature_std, window, labelmap):
    if len(window) == 0:
        print("未采集到有效数据，默认预测为 rest")
        return 2
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
# 0 -> left; 1 -> right; 2 -> rest


def get_Prediction(mode):
    global height, Mode
    global model, feature_mean, feature_std
    blinknum = 0
    feature_window = []
    start = time.time()
    while len(feature_window) < 30:
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
        if poorSignal == 200:
            drone_control("land")
            height = 0
            print(f"PoorSignal too high: {poorSignal}, landing for safety...")
            Mode = 0
            return -1, 2
        if(poorSignal >= 20 or attention == 0 or meditation == 0):
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
        print(f"{mode}:当前读取数据：attention={attention} meditation={meditation} blinkStrength={neuropy.blinkStrength}")
        target_time = start + len(feature_window) * 0.1
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
    predicted = print_prediction(model, feature_mean, feature_std, feature_window, labelmap)
    return blinknum , predicted


MINDWAVE_PORT = os.getenv("MINDWAVE_PORT", "/dev/cu.usbmodem2017_2_251")
MINDWAVE_BAUD = int(os.getenv("MINDWAVE_BAUD", "57600"))
neuropy = NeuroSkyPy(MINDWAVE_PORT, MINDWAVE_BAUD)
neuropy.start()
labelmap = ["left", "right", "rest"]
Mode = 0 # 0为升降，1为转弯，2为前后
height = 0
model, feature_mean, feature_std = load_model()
def get_Comparision(mode):
    attnum,mednum ,blinknum = 0,0,0
    global height, Mode
    while True:
        for i in range(30):
            attention = neuropy.attention or 0
            meditation = neuropy.meditation or 0
            blink_strength = neuropy.blinkStrength or 0
            poorSignal = neuropy.poorSignal or 0
            if poorSignal == 200:
                print(f"PoorSignal too high: {poorSignal}, landing for safety...")
                return 0,0,-1
            if(poorSignal >= 20 or attention ==0 or meditation == 0):
                print(f"PoorSignal too high: {poorSignal}, reset window...")
                attnum, mednum, blinknum = 0, 0, 0
                time.sleep(1)
                break
            if(blink_strength > 100):
                blinknum += 1
                print("眨眼计数+1")
            if(attention > 30):
                attnum += 1
            if(meditation > 50):
                mednum += 1
            print(f"{mode}:当前读取数据：attention={attention} meditation={meditation} blinkStrength={neuropy.blinkStrength}")
            time.sleep(0.1)
            if i == 29:
                return attnum, mednum, blinknum

def drone_control(operation):
    # if(operation == "takeoff"):
    #     me.takeoff()
    # elif(operation == "land"):
    #     me.land()
    # elif(operation == "up"):
    #     me.send_rc_control(0, 0, 10, 0)
    # elif(operation == "down"):  
    #     me.send_rc_control(0, 0, -10, 0)
    # elif(operation == "forward"):
    #     me.send_rc_control(0, 10, 0, 0)
    # elif(operation == "backward"):
    #     me.send_rc_control(0, -10, 0, 0)
    # elif(operation == "left"):
    #     me.send_rc_control(0, 0, 0, -30)
    # elif(operation == "right"):
    #     me.send_rc_control(0, 0, 0, 30)
    time.sleep(3)



# try:
#     me.connect(wait_for_state=False)
# except Exception as exc:
#     print("无人机未响应，请先连接 Tello Wi-Fi 后重试：", exc)
#     sys.exit(1)

print("无人机连接成功！")
# print(f"无人机目前电量：{me.get_battery()}")#链接无人机打印电量

try:
    while True:
        # 读取数据
        poorSignal = neuropy.poorSignal#信号质量，0表示最好
        if poorSignal >= 20:
            print(f"PoorSignal too high: {poorSignal}, skipping...")
            time.sleep(0.5)
            continue
        
        if Mode == 0:#先做到上升下降，再进入水平模式 #把所有poorsignal信号都改成小于了。
            print("一秒后，进入升降模式，多次眨眼则进入转弯模式\n注意力高则起飞，冥想高则降落")
            time.sleep(1)
            while True:
                print("升降：")
                attnum, mednum, blinknum = get_Comparision("升降")
                if(blinknum == -1):
                    height = 0
                    Mode = 0
                    if height > 0:
                        drone_control("land")
                    break
                if blinknum > 1:
                    if height > 1:
                        print("检测到眨眼，进入水平模式")
                        Mode = 1
                        break
                    else:
                        print("检测到眨眼，但无人机未起飞，继续升降模式")
                        time.sleep(1)
                        continue
                if attnum < 20 and mednum < 20:
                    print("注意力和冥想过低，原地不动")
                    time.sleep(1)
                    continue
                if attnum >= mednum:
                    if height == 0:
                        height = 100
                        print("Attention更高，无人机起飞！")
                        drone_control("takeoff")
                        continue
                    if height < 150:
                        print("Attention更高，无人机上升10cm！")
                        height += 10
                        drone_control("up")
                    else:
                        print("无人机已到达最大高度！")
                        time.sleep(1)
                else:
                    if height == 100:
                        height = 0
                        print("Meditation更高，无人机降落！")
                        drone_control("land")
                        continue
                    if height > 100:
                        print("Meditation更高，无人机下降10cm！")
                        height -= 10
                        drone_control("down")
                    else:
                        print("无人机已到达最低高度！")
                        time.sleep(1)

        if Mode == 1:#要求无人机距离远且为转弯模式，进入转弯模式
            print("一秒后，进入转弯模式，两次眨眼则进入前后模式")
            time.sleep(1)
            while True:
                #print("检测到眨眼，进入转弯模式，想象左右肢体来控制无人机方向：\n眨眼一次则退出转弯模式进入前进后退模式，眨眼多次则退出水平模式准备降落")
                print("转弯：")
                blinknum , predicted = get_Prediction("转弯")
                if blinknum == -1:
                    height, Mode= 0,0
                    print(f"PoorSignal too high: 200, landing for safety...")
                    drone_control("land")
                    break
                if blinknum >= 2:
                    print("两次眨眼，进入前后模式")
                    Mode = 2
                    break
                print(f"预测结果：{labelmap[predicted]} | 3秒/30组完成")
                if predicted == 2:#一直不动就会休息
                    print("休息状态，rest")
                    time.sleep(3)
                elif predicted == 1:#1对应的是右手右腿
                    print("无人机右转")
                    drone_control("right")
                elif predicted == 0:#0对应的是左手左腿
                    print("无人机左转")
                    drone_control("left")

        if Mode == 2:#水平模式下不左转右转就进入前进后退模式
            print("一秒后，进入前后模式\n注意力高则前进，冥想高则后退，眨眼则进入升降模式")
            time.sleep(1)
            while True:
                attnum, mednum, blinknum = get_Comparision("前后")
                if blinknum == -1:
                    height, Mode = 0, 0
                    print(f"PoorSignal too high: 200, landing for safety...")
                    drone_control("land")
                    break

                if blinknum >= 2:
                    print("检测到眨眼两次，进入升降模式")
                    Mode = 0
                    break
                if blinknum == 1:
                    print("检测到眨眼，重新测量")
                if attnum < 20 or mednum < 20:
                    print("注意力或冥想过低，原地不动")
                    time.sleep(1)
                    continue
                if attnum >= mednum:
                    print("Attention更高，无人机前进10cm！")
                    drone_control("forward")
                else:
                    print("Meditation更高，无人机后退10cm！")
                    drone_control("backward")
                #time.sleep(1)

except KeyboardInterrupt:
    pass#只让ctrlc强制退出程序
drone_control("land")
print(f"\n{'='*60}")
print("所有进程已退出")
print(f"{'='*60}")
neuropy.stop()