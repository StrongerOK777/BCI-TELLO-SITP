import pygame
import numpy as np
import time
from neuropy import NeuroSkyPy
import threading
import random
from pathlib import Path
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
loss_values = [] # 存储训练过程中的损失值
action = 3        # 当前动作状态：0-左，1-右，2-休息，3-倒计时/准备
displaytime = 30  # 训练循环次数（轮回次数）
stop_event = threading.Event()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 定义 PyGame 窗口和全局显示对象（修复 macOS 线程问题，需在主线程初始化）
pygame.init()
window_size = (400, 400)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("脑机接口训练展示")

def show_array(diraction, current_time, total_remaining=None):
    # 定义颜色
    white = (255, 255, 255)
    black = (0, 0, 0)

    # 清空屏幕
    screen.fill(white)

    # 绘制圆形
    circle_radius = 50
    circle_color = (177, 186, 152)
    circle_center = (window_size[0] // 2, window_size[1] // 2)#window_size[0]为窗口宽度，【1】为高度
    pygame.draw.circle(screen, circle_color, circle_center, circle_radius)

    # 绘制箭头
    arrow_color = (0, 0, 255)  # 蓝色
    # 左箭头顶点坐标
    left_arrow = [(circle_center[0] - circle_radius - 100, circle_center[1]),
                  (circle_center[0] - circle_radius - 20, circle_center[1] - 40),
                  (circle_center[0] - circle_radius - 20, circle_center[1] + 40)]#circlecenter【0】为圆心横坐标，【1】为纵坐标
    #三个参数对应的是三个顶点
    if(diraction == 0): # 要求做左手动作
        pygame.draw.polygon(screen, circle_color, left_arrow)
    else: # 非左手状态显示空心箭头
        pygame.draw.polygon(screen, white, left_arrow)  # 绘制白色内部
        pygame.draw.polygon(screen, black, left_arrow, 1)  # 绘制黑色边框
    # 右箭头顶点坐标
    right_arrow =  [(circle_center[0] + circle_radius + 100, circle_center[1]),
                      (circle_center[0] + circle_radius + 20, circle_center[1] - 40),
                      (circle_center[0] + circle_radius + 20, circle_center[1] + 40)]
    if(diraction == 1): # 要求做右手动作
        pygame.draw.polygon(screen, circle_color, right_arrow)
    else: # 非右手状态显示空心箭头
        pygame.draw.polygon(screen, white, right_arrow)  # 绘制白色内部
        pygame.draw.polygon(screen, black, right_arrow, 1)  # 绘制黑色边框

    font = pygame.font.Font(None, 36)#设置字体，none则为使用系统默认字体
    text = font.render(str(current_time), True, black)#给字符串设置颜色，true为是否需要抗锯齿（即平滑效果）
    text_rect = text.get_rect(center=circle_center)#设定文字的位置：即center

    if total_remaining is not None:
        total_text = font.render(f"REST TIME: {int(total_remaining)}s", True, black)
        total_rect = total_text.get_rect(center=(circle_center[0], circle_center[1] + circle_radius + 40))

    # 更新屏幕
    screen.blit(text, text_rect)
    if total_remaining is not None:
        screen.blit(total_text, total_rect)
    pygame.display.flip()#之前的绘制都在缓冲区，现在释放打印在屏幕上
    pygame.event.pump() # 处理事件，防止串口读取时 UI 卡死
    if action == 3:
        time.sleep(1)
    else:
        time.sleep(3)


def show_finish_countdown(seconds):
    # 采集结束倒计时提示
    for i in range(seconds, 0, -1):
        show_array(3, i, i)


def writedata():
    global action
    # 串口配置（如果是 macOS，请确保端口路径正确）
    import sys
    port = "COM5" if sys.platform == "win32" else "/dev/cu.usbmodem2017_2_251" 
    neuropy = NeuroSkyPy(port, 57600)
    # 打开串口连接
    neuropy.start()
    try:
        files = {
            0: open(DATA_DIR / "actionleft.txt", "a"),
            1: open(DATA_DIR / "actionright.txt", "a"),
            2: open(DATA_DIR / "rest.txt", "a"),
        }
        session_start = time.time()
        while True:
            if stop_event.is_set():
                break
            # 从 NeuroSky 设备实时读取 12 个维度的脑电特征数据
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
            # 倒计时/准备阶段不写数据
            if action not in (0, 1, 2):
                time.sleep(0.1)
                continue
            # 跳过无效数据
            if any(v is None for v in [attention, meditation, delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, midGamma, poorSignal, blinkStrength]):
                time.sleep(0.1)
                continue
            # 信号质量差时跳过
            if poorSignal > 50:
                time.sleep(0.1)
                continue
            # action代表着要求人做出的行为分类，三个==0/1/2除了存储数据的文件路径不同其他都一样，但是我不会化简。
            #action==0为左手，==1为右手，==2为rest
            if(action == 0):
                current_time = time.time() - session_start
                current_time_str = "{:.2f}".format(current_time)
                data_line = f"{current_time_str},{attention}|{meditation}|{delta}|{theta}|{lowAlpha}|{highAlpha}|{lowBeta}|{highBeta}|{lowGamma}|{midGamma}|{poorSignal}|{blinkStrength}\n"
                files[0].write(data_line)
                files[0].flush()
                # 打印数据到控制台
                print(f"Recording LEFT: {data_line.strip()}")
                time.sleep(1)
            elif(action == 1):
                current_time = time.time() - session_start
                current_time_str = "{:.2f}".format(current_time)
                data_line = f"{current_time_str},{attention}|{meditation}|{delta}|{theta}|{lowAlpha}|{highAlpha}|{lowBeta}|{highBeta}|{lowGamma}|{midGamma}|{poorSignal}|{blinkStrength}\n"
                files[1].write(data_line)
                files[1].flush()
                # 打印数据到控制台
                print(f"Recording RIGHT: {data_line.strip()}")
                time.sleep(1)
            elif (action == 2):
                current_time = time.time() - session_start
                current_time_str = "{:.2f}".format(current_time)
                data_line = f"{current_time_str},{attention}|{meditation}|{delta}|{theta}|{lowAlpha}|{highAlpha}|{lowBeta}|{highBeta}|{lowGamma}|{midGamma}|{poorSignal}|{blinkStrength}\n"
                files[2].write(data_line)
                files[2].flush()
                # 打印数据到控制台
                print(f"Recording REST: {data_line.strip()}")
                time.sleep(1)
            if displaytime <= 0:
                stop_event.set()
                break

    except Exception as e:
        print(f"Recording stopped or error: {e}")
        pass#当到时间就关闭

    # 关闭文件和串口连接
    for f in [files.get(0), files.get(1), files.get(2)]:
        if f:
            f.close()
    neuropy.stop()


def my_windows():
    global action
    global displaytime
    total_per_round = 7
    while 1:
        if(action == 3):
            # 3秒倒计时准备
            for i in range(3):
                total_remaining = displaytime * total_per_round - (3 - i)
                show_array(3, 3-i, total_remaining)#每次开始时的倒计时，时间每变一次就要重新打印一次，3-i为倒计时
            
            # 核心业务逻辑：随机生成训练指令
            random_num = random.randint(0, 2)#随机获取0~2之间的整数
            action = random_num # 更新全局 action，通知数据采集线程切换文件
            total_remaining = displaytime * total_per_round - 3
            show_array(action, 0, total_remaining) # 切换到动作显示界面，持续 3 秒
            
            action = 3 # 动作结束，恢复到倒计时状态
            time.sleep(1)
            displaytime = displaytime - 1#即轮回三十次后停止

        if displaytime <= 0:
            stop_event.set()
            break

# 开启多线程：数据采集在后台运行，UI 界面在主线程运行（macOS 兼容方案）
thread2 = threading.Thread(target=writedata, daemon=True)

# 启动采集线程
thread2.start()

# UI 循环必须在主线程运行，否则 macOS 下会崩溃
my_windows()

# 采集结束倒计时（UI 可见）
show_finish_countdown(5)

# 等待采集线程执行完毕
thread2.join()

print("\n--- Data collection finished. Starting model training ---")
pygame.quit()

def parse_feature_line(line):
    parts = line.strip().split(',')
    if len(parts) != 2:
        return None
    raw = parts[1].split('|')
    if len(raw) != 12:
        return None
    values = list(map(float, raw))
    (attention, meditation, delta, theta, low_alpha, high_alpha,
     low_beta, high_beta, low_gamma, mid_gamma, poor_signal, blink_strength) = values
    if poor_signal > 50:
        return None
    beta = low_beta + high_beta
    alpha = low_alpha + high_alpha
    theta_safe = theta if theta != 0 else 1e-6
    beta_theta_ratio = beta / theta_safe
    alpha_theta_ratio = alpha / theta_safe
    engagement = beta / (alpha + 1e-6)
    features = [
        attention, meditation, delta, theta, low_alpha, high_alpha,
        low_beta, high_beta, low_gamma, mid_gamma, blink_strength,
        beta_theta_ratio, alpha_theta_ratio, engagement
    ]
    return features


def load_windowed_samples(file_path, label, window_size=5, stride=1):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    samples = []
    features_list = [parse_feature_line(line) for line in lines]
    features_list = [f for f in features_list if f is not None]
    if len(features_list) < window_size:
        return samples
    for i in range(0, len(features_list) - window_size + 1, stride):
        window = np.array(features_list[i:i + window_size], dtype=np.float32)
        mean_feat = window.mean(axis=0)
        std_feat = window.std(axis=0)
        combined = np.concatenate([mean_feat, std_feat], axis=0)
        samples.append((combined, label))
    return samples


class FeatureDataset(Dataset):
    def __init__(self, samples, mean=None, std=None):
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        features, label = self.samples[index]
        # 归一化仅使用训练集统计量
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / (self.std + 1e-6)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 文件路径和对应的标签
file_paths = [
    DATA_DIR / 'actionleft.txt',
    DATA_DIR / 'actionright.txt',
    DATA_DIR / 'rest.txt'
]
labels = [0, 1, 2]

all_samples = []
for file_path, label in zip(file_paths, labels):
    all_samples.extend(load_windowed_samples(file_path, label, window_size=5, stride=1))

if len(all_samples) == 0:
    raise RuntimeError("没有可用的训练样本，请先采集数据。")

# 假设您已经定义了您的模型
class ImprovedModel(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x






random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)
random.seed(random_state)

def stratified_split(samples, train_ratio=0.8, val_ratio=0.1):
    # 分层划分，避免类别分布偏移
    by_label = {}
    for features, label in samples:
        by_label.setdefault(label, []).append((features, label))
    train_samples, val_samples, test_samples = [], [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_samples.extend(items[:n_train])
        val_samples.extend(items[n_train:n_train + n_val])
        test_samples.extend(items[n_train + n_val:])
    return train_samples, val_samples, test_samples

train_samples, val_samples, test_samples = stratified_split(all_samples, train_ratio=0.8, val_ratio=0.1)

train_features = np.stack([s[0] for s in train_samples], axis=0)
feature_mean = train_features.mean(axis=0)
feature_std = train_features.std(axis=0)

train_dataset = FeatureDataset(train_samples, mean=feature_mean, std=feature_std)
val_dataset = FeatureDataset(val_samples, mean=feature_mean, std=feature_std)
test_dataset = FeatureDataset(test_samples, mean=feature_mean, std=feature_std)

input_dim = train_features.shape[1]
model = ImprovedModel(input_dim)
# 定义损失函数和优化器
class_counts = np.bincount([label for _, label in train_samples], minlength=3)
class_weights = class_counts.sum() / (class_counts + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
epochs = 150#模型遍历整个训练集的次数
best_val_loss = float("inf")
patience = 15
patience_counter = 0
for epoch in range(epochs):
    model.train()#开启模型训练模式
    for inputs, labels in train_loader:  # 注意这里只有一个输入
        optimizer.zero_grad()#将模型参数的梯度重置为0
        outputs = model(inputs)  # 使用连接后的数据
        loss = criterion(outputs, labels)#损失率其实是失败率
        loss.backward()#根据loss返回调整参数
        optimizer.step()#根据梯度调整模型参数

    loss_values.append(loss.item())

    model.eval()
    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            v_loss = criterion(outputs, labels)
            val_loss += v_loss.item() * len(labels)
            val_count += len(labels)
    val_loss = val_loss / max(val_count, 1)

    # 早停：验证集不再改善则停止
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "input_dim": input_dim
        }
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
#模型训练完成

#开始评估模型
model.eval()#进入模型评估模式
correct = 0
total = 0
strict_correct = 0

with torch.no_grad():
    for inputs, labels in test_loader:  # 注意这里只有一个输入
        outputs = model(inputs)  # 使用连接后的数据
        _, predicted = torch.max(outputs.data, 1)#模型预测的输出为对三个方向预测的评估分数，分数最高的就是模型的答案。取最高分数的
        #dim==0为32个样本，规定epoch批次大小为32，dim==1为三个方向对应分数：即模型规定输出三个方向分数
        #_意思是只取标签不取分数，即获取对应索引的类别

        # 根据要求调整标签判断
        # 修改判断逻辑，符合要求即认为判断正确
        for i in range(len(predicted)):
            if predicted[i] == labels[i]:
                strict_correct += 1
            if predicted[i] == labels[i]:
                correct += 1
            elif predicted[i] == 0 or predicted[i] == 1 :#即如果rest下被误判为右手左手也是可以接受的
                if labels[i] == 2 :
                    correct += 1

        total += len(predicted)

accuracy = correct / total
strict_accuracy = strict_correct / total
print(f'Test Accuracy (strict): {strict_accuracy * 100:.2f}%')
print(f'Test Accuracy (relaxed): {accuracy * 100:.2f}%')

if_save = None
while True:
    if_save = input("是否保存模型？（是/y，否/n)\n").strip().lower()
    if if_save in ("y", "n", "是", "否", "yes", "no"):
        break
    print("请输入 y/yes 或 n/no。")

if if_save in ("y", "是", "yes"):
    # 指定保存路径
    save_path = 'user_model.pth'

    if "best_state" in locals():
        torch.save(best_state, save_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'feature_mean': feature_mean,
            'feature_std': feature_std,
            'input_dim': input_dim
        }, save_path)
    print("save successfully!")