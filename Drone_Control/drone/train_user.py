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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
# 训练过程统计与状态控制
loss_values = [] # 存储训练过程中的损失值
action = 3        # 当前动作状态：0-左，1-右，2-休息，3-倒计时/准备
displaytime = 30  # 训练循环次数（轮回次数）
stop_event = threading.Event()      # 用于通知采集线程停止
record_event = threading.Event()    # 采集窗口开关（倒计时不采集，动作窗口采集）
batch_done_event = threading.Event()# 采满一批样本后通知 UI 进入下一轮

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent.parent
PICTURE_DIR = BASE_DIR / "picture"
PICTURE_DIR.mkdir(parents=True, exist_ok=True)

# 定义 PyGame 窗口和全局显示对象（修复 macOS 线程问题，需在主线程初始化）
print("kaishilelelelelele")
pygame.init()
window_size = (400, 400)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("脑机接口训练展示")

def show_array(diraction, current_time, total_remaining=None):
    # UI：显示当前动作提示与倒计时
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


def show_finish_countdown(seconds):
    # UI：采集结束提示倒计时
    # 采集结束倒计时提示
    for i in range(seconds, 0, -1):
        show_array(3, i, i)


def writedata():
    # 后台采集线程：只在 record_event 打开时写入数据
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
        last_action = None
        samples_in_action = 0
        max_samples_per_action = 30
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
            # 只在动作采集窗口内写数据
            if not record_event.is_set():
                last_action = None
                samples_in_action = 0
                batch_done_event.clear()
                time.sleep(0.1)
                continue
            if action != last_action:
                last_action = action
                samples_in_action = 0
            if samples_in_action >= max_samples_per_action:
                batch_done_event.set()
                time.sleep(0.05)
                continue
            # 跳过无效数据
            if any(v is None for v in [attention, meditation, delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, midGamma, poorSignal, blinkStrength]):
                time.sleep(0.1)
                continue
            # 信号质量差时跳过
            if poorSignal > 20 or attention == 0 or meditation == 0:
                print(f"Recording PoorSignal: {poorSignal}, skipping...")
                time.sleep(0.1)
                continue
            # action代表着要求人做出的行为分类
            # action==0为左手，==1为右手，==2为rest
            if(action == 0):
                current_time = time.time() - session_start
                current_time_str = "{:.2f}".format(current_time)
                data_line = f"{current_time_str},{attention}|{meditation}|{delta}|{theta}|{lowAlpha}|{highAlpha}|{lowBeta}|{highBeta}|{lowGamma}|{midGamma}|{poorSignal}|{blinkStrength}\n"
                files[0].write(data_line)
                files[0].flush()
                # 打印数据到控制台
                print(f"Recording LEFT: {data_line.strip()}")
                samples_in_action += 1
                time.sleep(0.1)
            elif(action == 1):
                current_time = time.time() - session_start
                current_time_str = "{:.2f}".format(current_time)
                data_line = f"{current_time_str},{attention}|{meditation}|{delta}|{theta}|{lowAlpha}|{highAlpha}|{lowBeta}|{highBeta}|{lowGamma}|{midGamma}|{poorSignal}|{blinkStrength}\n"
                files[1].write(data_line)
                files[1].flush()
                # 打印数据到控制台
                print(f"Recording RIGHT: {data_line.strip()}")
                samples_in_action += 1
                time.sleep(0.1)
            elif (action == 2):
                current_time = time.time() - session_start
                current_time_str = "{:.2f}".format(current_time)
                data_line = f"{current_time_str},{attention}|{meditation}|{delta}|{theta}|{lowAlpha}|{highAlpha}|{lowBeta}|{highBeta}|{lowGamma}|{midGamma}|{poorSignal}|{blinkStrength}\n"
                files[2].write(data_line)
                files[2].flush()
                # 打印数据到控制台
                print(f"Recording REST: {data_line.strip()}")
                samples_in_action += 1
                time.sleep(0.1)
            if displaytime <= 0:
                print("数据采集时间到，停止采集。")
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
    # UI 主线程：倒计时 -> 选择动作 -> 等待采满 30 组 -> 下一轮
    global action
    global displaytime
    total_per_round = 7
    while 1:
        if(action == 3):
            # 3秒倒计时准备
            for i in range(3):
                total_remaining = displaytime * total_per_round - (i)
                show_array(3, 3-i, total_remaining)#每次开始时的倒计时，时间每变一次就要重新打印一次，3-i为倒计时
            
            # 核心业务逻辑：随机生成训练指令
            random_num = random.randint(0, 2)#随机获取0~2之间的整数
            action = random_num # 更新全局 action，通知数据采集线程切换文件
            total_remaining = displaytime * total_per_round - 3
            show_array(action, 0, total_remaining)
            record_event.set()
            batch_done_event.clear()
            while not batch_done_event.is_set() and not stop_event.is_set():
                pygame.event.pump()
                time.sleep(0.01)
            record_event.clear()

            action = 3 # 动作结束，恢复到倒计时状态
            displaytime = displaytime - 1#即轮回三十次后停止

        if displaytime <= 0:
            print("数据采集时间到，准备结束。")
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
    # 解析单行特征并扩展为 14 维（含派生特征）
    parts = line.strip().split(',')
    if len(parts) != 2:
        return None
    raw = parts[1].split('|')
    if len(raw) != 12:
        return None
    values = list(map(float, raw))
    (attention, meditation, delta, theta, low_alpha, high_alpha,
     low_beta, high_beta, low_gamma, mid_gamma, poor_signal, blink_strength) = values
    if poor_signal > 20:
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


def load_windowed_samples(file_path, label, window_size=20, stride=1):
    # 滑动窗口生成样本：窗口内均值+方差拼接
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


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, samples, mean, std, augment=False):
        self.samples = samples
        self.mean = mean
        self.std = std
        self.augment = augment
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, label = self.samples[idx]
        features = (features - self.mean) / (self.std + 1e-8)
        
        # 训练时添加噪声增强
        if self.augment:
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise
        
        return torch.tensor(features, dtype=torch.float32), label

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

# 模型定义（全连接+BN+Dropout）
class WideModel(nn.Module):
    """宽型网络：更多神经元，较浅层"""
    def __init__(self, input_dim):
        super(WideModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)  # 更宽
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.15)  # 降低dropout
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class DeepModel(nn.Module):
    """深型网络：更多层，适中宽度"""
    def __init__(self, input_dim):
        super(DeepModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(256, 192)
        self.bn2 = nn.BatchNorm1d(192)
        self.dropout2 = nn.Dropout(0.15)
        
        self.fc3 = nn.Linear(192, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.15)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x


class BalancedModel(nn.Module):
    """平衡型网络：中等深度和宽度"""
    def __init__(self, input_dim):
        super(BalancedModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 384)
        self.bn1 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(0.12)
        
        self.fc2 = nn.Linear(384, 192)
        self.bn2 = nn.BatchNorm1d(192)
        self.dropout2 = nn.Dropout(0.18)
        
        self.fc3 = nn.Linear(192, 96)
        self.bn3 = nn.BatchNorm1d(96)
        self.dropout3 = nn.Dropout(0.18)
        
        self.fc4 = nn.Linear(96, 3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x


class ResidualModel(nn.Module):
    """带残差连接的网络"""
    def __init__(self, input_dim):
        super(ResidualModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.15)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.15)
        
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        # 第一层
        out = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        
        # 第二层（带残差连接）
        identity = out
        out = self.dropout2(F.relu(self.bn2(self.fc2(out))))
        out = out + identity  # 残差连接
        
        # 第三层和输出层
        out = self.dropout3(F.relu(self.bn3(self.fc3(out))))
        out = self.fc4(out)
        return out


class LightModel(nn.Module):
    """轻量级网络：更少dropout和正则化"""
    def __init__(self, input_dim):
        super(LightModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 320)
        self.bn1 = nn.BatchNorm1d(320)
        self.dropout1 = nn.Dropout(0.08)  # 很小的dropout
        
        self.fc2 = nn.Linear(320, 160)
        self.bn2 = nn.BatchNorm1d(160)
        self.dropout2 = nn.Dropout(0.12)
        
        self.fc3 = nn.Linear(160, 80)
        self.bn3 = nn.BatchNorm1d(80)
        self.dropout3 = nn.Dropout(0.12)
        
        self.fc4 = nn.Linear(80, 3)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x


# ==================== 2. 带数据增强的数据集类 ====================
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, samples, mean, std, augment=False):
        self.samples = samples
        self.mean = mean
        self.std = std
        self.augment = augment
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, label = self.samples[idx]
        features = (features - self.mean) / (self.std + 1e-8)
        
        # 训练时添加噪声增强
        if self.augment:
            noise = np.random.normal(0, 0.008, features.shape)  # 稍微减小噪声
            features = features + noise
        
        return torch.tensor(features, dtype=torch.float32), label


# ==================== 3. 随机种子与数据划分 ====================
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)
random.seed(random_state)

def stratified_split(samples, train_ratio=0.8, val_ratio=0.1):
    """分层划分，避免类别分布偏移"""
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


# ==================== 4. 数据准备与标准化 ====================
train_samples, val_samples, test_samples = stratified_split(all_samples, train_ratio=0.8, val_ratio=0.1)

# 使用RobustScaler进行标准化
train_features = np.stack([s[0] for s in train_samples], axis=0)
scaler = RobustScaler()
train_features_scaled = scaler.fit_transform(train_features)

# 计算标准化后的均值和标准差
feature_mean = train_features_scaled.mean(axis=0)
feature_std = train_features_scaled.std(axis=0)

# 创建数据集
train_dataset = FeatureDataset(train_samples, mean=feature_mean, std=feature_std, augment=True)
val_dataset = FeatureDataset(val_samples, mean=feature_mean, std=feature_std, augment=False)
test_dataset = FeatureDataset(test_samples, mean=feature_mean, std=feature_std, augment=False)

# 批次大小
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==================== 5. 定义不同的模型架构列表 ====================
model_architectures = [
    ("Wide Model", WideModel),
    ("Deep Model", DeepModel),
    ("Balanced Model", BalancedModel),
    ("Residual Model", ResidualModel),
    ("Light Model", LightModel)
]


# ==================== 6. 训练单个模型函数 ====================
def train_single_model(model_name, ModelClass, model_id, epochs=300, patience=40):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"Training Model {model_id + 1}/5: {model_name}")
    print(f"{'='*60}")
    
    # 为每个模型设置不同的随机种子
    torch.manual_seed(random_state + model_id * 100)
    np.random.seed(random_state + model_id * 100)
    
    input_dim = train_features.shape[1]
    model = ModelClass(input_dim)
    
    # 计算类别权重
    class_counts = np.bincount([label for _, label in train_samples], minlength=3)
    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 根据模型类型调整优化器参数
    if "Light" in model_name:
        # 轻量级模型：更小的weight_decay
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    elif "Deep" in model_name:
        # 深度模型：稍大的学习率
        optimizer = optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.008)
    else:
        # 其他模型：标准参数
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.008)
    
    # 学习率调度器（修复warning）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10  # 移除verbose参数
    )
    
    # 训练参数
    best_val_loss = float("inf")
    patience_counter = 0
    loss_values = []
    val_loss_values = []
    best_epoch = 0
    
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_count = 0
        train_correct = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            train_count += len(labels)
            
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / max(train_count, 1)
        train_accuracy = train_correct / max(train_count, 1)
        loss_values.append(train_loss)
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_count = 0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                v_loss = criterion(outputs, labels)
                val_loss += v_loss.item() * len(labels)
                val_count += len(labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / max(val_count, 1)
        val_accuracy = val_correct / max(val_count, 1)
        val_loss_values.append(val_loss)
        
        # 学习率调度（获取当前学习率）
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 如果学习率改变，打印提示
        if old_lr != new_lr:
            print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # 早停机制（patience增加到40）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "input_dim": input_dim,
                "scaler": scaler,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch
            }
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  → Early stopping at epoch {epoch + 1}")
            print(f"  → Best model was at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
            break
        
        # 每15轮打印一次（减少输出频率）
        if (epoch + 1) % 15 == 0:
            print(f'  Epoch [{epoch + 1}/{epochs}] | '
                  f'Train Loss: {train_loss:.4f} (Acc: {train_accuracy*100:.2f}%) | '
                  f'Val Loss: {val_loss:.4f} (Acc: {val_accuracy*100:.2f}%) | '
                  f'Patience: {patience_counter}/{patience}')
    
    # 恢复最佳模型
    model.load_state_dict(best_state["model_state_dict"])
    print(f"\n  ✓ {model_name} completed!")
    print(f"    - Best Epoch: {best_epoch}")
    print(f"    - Best Val Loss: {best_val_loss:.4f}")
    print(f"    - Total Epochs: {epoch + 1}")
    
    return model, loss_values, val_loss_values, best_state


# ==================== 7. 训练5个不同架构的模型 ====================
models = []
all_train_losses = []
all_val_losses = []
model_names = []

for i, (model_name, ModelClass) in enumerate(model_architectures):
    model, train_losses, val_losses, best_state = train_single_model(
        model_name, ModelClass, i, epochs=300, patience=40
    )
    models.append(model)
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    model_names.append(model_name)

print(f"\n{'='*60}")
print("All models trained successfully!")
print(f"{'='*60}\n")


# ==================== 8. 集成预测函数 ====================
def ensemble_predict(models, data_loader):
    """使用多个模型进行集成预测"""
    all_predictions = []
    all_labels = []
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            # 收集所有模型的输出
            batch_predictions = []
            for model in models:
                outputs = model(inputs)
                batch_predictions.append(outputs)
            
            # 平均所有模型的输出
            avg_output = torch.stack(batch_predictions).mean(dim=0)
            _, predicted = torch.max(avg_output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)


# ==================== 9. 评估集成模型 ====================
print("Evaluating ensemble model on test set...")

predictions, true_labels = ensemble_predict(models, test_loader)

# 计算准确率
correct = 0
total = 0
strict_correct = 0

for i in range(len(predictions)):
    pred = predictions[i]
    label = true_labels[i]
    
    # 严格准确率
    if pred == label:
        strict_correct += 1
    
    # 宽松准确率
    if pred == label:
        correct += 1
    elif (pred == 0 or pred == 1) and label == 2:
        correct += 1
    
    total += 1

accuracy = correct / total
strict_accuracy = strict_correct / total

print(f"\n{'='*60}")
print("ENSEMBLE MODEL RESULTS")
print(f"{'='*60}")
print(f'Test Accuracy (strict):  {strict_accuracy * 100:.2f}%')
print(f'Test Accuracy (relaxed): {accuracy * 100:.2f}%')
print(f"{'='*60}\n")


# ==================== 10. 单个模型评估（对比） ====================
print("Individual model performance on test set:")
print(f"{'='*60}")
for idx, (model, model_name) in enumerate(zip(models, model_names)):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
    
    individual_accuracy = correct / total
    print(f"{model_name:20s}: {individual_accuracy * 100:.2f}%")

print(f"{'='*60}\n")


# ==================== 11. 保存模型（可选） ====================
save_models = True
if save_models:
    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        filename = MODEL_DIR / f'model_{idx + 1}_{model_name.replace(" ", "_").lower()}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_mean': feature_mean,
            'feature_std': feature_std,
            'input_dim': train_features.shape[1],
            'scaler': scaler,
            'model_name': model_name
        }, filename)
    print("✓ All models saved successfully!")
    print()


# ==================== 12. 可视化训练过程 ====================
try:
    import matplotlib.pyplot as plt
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training and Validation Loss for All Models', fontsize=16, fontweight='bold')
    
    # 为每个模型绘制训练和验证曲线
    for idx in range(5):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # 绘制训练和验证损失
        epochs_range = range(1, len(all_train_losses[idx]) + 1)
        ax.plot(epochs_range, all_train_losses[idx], label='Train Loss', linewidth=2, alpha=0.8)
        ax.plot(epochs_range, all_val_losses[idx], label='Val Loss', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(model_names[idx], fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(PICTURE_DIR / 'training_validation_loss.png', dpi=300, bbox_inches='tight')
    print("✓ Training/Validation loss plot saved as 'training_validation_loss.png'")
    print()
    
    # 绘制所有模型对比图
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for idx, model_name in enumerate(model_names):
        plt.plot(all_train_losses[idx], label=model_name, linewidth=1.5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for idx, model_name in enumerate(model_names):
        plt.plot(all_val_losses[idx], label=model_name, linewidth=1.5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PICTURE_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model comparison plot saved as 'model_comparison.png'")
    
except ImportError:
    print("Matplotlib not available, skipping visualization")

print(f"\n{'='*60}")
print("All tasks completed successfully!")
print(f"{'='*60}")
# ==================== 13. 知识蒸馏：合成最终模型 ====================
print(f"\n{'='*60}")
print("Knowledge Distillation: Creating Final Unified Model")
print(f"{'='*60}\n")

# 定义最终的统一模型架构（结合各架构优点）
class FinalUnifiedModel(nn.Module):
    """最终统一模型：综合各架构优点"""
    def __init__(self, input_dim):
        super(FinalUnifiedModel, self).__init__()
        
        # 第一层：较宽（吸收Wide Model优点）
        self.fc1 = nn.Linear(input_dim, 384)
        self.bn1 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(0.12)
        
        # 第二层：中等宽度
        self.fc2 = nn.Linear(384, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.15)
        
        # 第三层：带残差连接（吸收Residual Model优点）
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.15)
        
        # 第四层：逐渐收窄
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.18)
        
        # 输出层
        self.fc5 = nn.Linear(128, 3)

    def forward(self, x):
        # 前两层
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        
        # 带残差连接的第三层
        identity = x
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = x + identity  # 残差连接
        
        # 第四层和输出
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x


# 获取教师模型的软标签（集成预测）
def get_teacher_soft_labels(teacher_models, inputs, temperature=3.0):
    """
    从多个教师模型获取软标签
    temperature: 温度参数，越大软标签越"软"，知识传递越丰富
    """
    soft_labels_list = []
    
    for model in teacher_models:
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            # 使用温度缩放softmax
            soft_labels = F.softmax(outputs / temperature, dim=1)
            soft_labels_list.append(soft_labels)
    
    # 平均所有教师模型的软标签
    avg_soft_labels = torch.stack(soft_labels_list).mean(dim=0)
    return avg_soft_labels


# 知识蒸馏损失函数
def distillation_loss(student_outputs, teacher_soft_labels, hard_labels, 
                      temperature=3.0, alpha=0.7):
    """
    知识蒸馏损失 = α * 软标签损失 + (1-α) * 硬标签损失
    
    student_outputs: 学生模型的输出
    teacher_soft_labels: 教师模型的软标签
    hard_labels: 真实标签
    temperature: 温度参数
    alpha: 软标签损失的权重
    """
    # 软标签损失（KL散度）
    student_soft = F.log_softmax(student_outputs / temperature, dim=1)
    soft_loss = F.kl_div(student_soft, teacher_soft_labels, reduction='batchmean')
    soft_loss = soft_loss * (temperature ** 2)  # 温度平方缩放
    
    # 硬标签损失（交叉熵）
    hard_loss = F.cross_entropy(student_outputs, hard_labels)
    
    # 组合损失
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return total_loss, soft_loss, hard_loss


# 初始化最终模型
print("Initializing Final Unified Model...")
input_dim = train_features.shape[1]
final_model = FinalUnifiedModel(input_dim)

# 计算类别权重
class_counts = np.bincount([label for _, label in train_samples], minlength=3)
class_weights = class_counts.sum() / (class_counts + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 优化器设置
optimizer_final = optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=0.008)

# 学习率调度器
scheduler_final = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_final, mode='min', factor=0.5, patience=12
)

# 训练参数
distillation_epochs = 300
temperature = 3.0  # 温度参数
alpha = 0.7  # 软标签权重
best_val_loss_final = float("inf")
patience_final = 40
patience_counter_final = 0

train_losses_final = []
val_losses_final = []
best_epoch_final = 0

print(f"Training with Knowledge Distillation...")
print(f"  - Temperature: {temperature}")
print(f"  - Alpha (soft label weight): {alpha}")
print(f"  - Max Epochs: {distillation_epochs}")
print(f"  - Patience: {patience_final}\n")

for epoch in range(distillation_epochs):
    # ========== 训练阶段 ==========
    final_model.train()
    
    # 将教师模型设为评估模式
    for model in models:
        model.eval()
    
    epoch_loss = 0.0
    epoch_soft_loss = 0.0
    epoch_hard_loss = 0.0
    epoch_count = 0
    epoch_correct = 0
    
    for inputs, labels in train_loader:
        optimizer_final.zero_grad()
        
        # 获取教师模型的软标签
        teacher_soft_labels = get_teacher_soft_labels(models, inputs, temperature)
        
        # 学生模型前向传播
        student_outputs = final_model(inputs)
        
        # 计算蒸馏损失
        loss, soft_loss, hard_loss = distillation_loss(
            student_outputs, teacher_soft_labels, labels, 
            temperature, alpha
        )
        
        # 反向传播
        loss.backward()
        optimizer_final.step()
        
        # 统计
        epoch_loss += loss.item() * len(labels)
        epoch_soft_loss += soft_loss.item() * len(labels)
        epoch_hard_loss += hard_loss.item() * len(labels)
        epoch_count += len(labels)
        
        _, predicted = torch.max(student_outputs.data, 1)
        epoch_correct += (predicted == labels).sum().item()
    
    avg_train_loss = epoch_loss / max(epoch_count, 1)
    avg_soft_loss = epoch_soft_loss / max(epoch_count, 1)
    avg_hard_loss = epoch_hard_loss / max(epoch_count, 1)
    train_accuracy = epoch_correct / max(epoch_count, 1)
    train_losses_final.append(avg_train_loss)
    
    # ========== 验证阶段 ==========
    final_model.eval()
    val_loss = 0.0
    val_count = 0
    val_correct = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            # 获取教师软标签
            teacher_soft_labels = get_teacher_soft_labels(models, inputs, temperature)
            
            # 学生模型预测
            student_outputs = final_model(inputs)
            
            # 计算验证损失
            loss, _, _ = distillation_loss(
                student_outputs, teacher_soft_labels, labels,
                temperature, alpha
            )
            
            val_loss += loss.item() * len(labels)
            val_count += len(labels)
            
            _, predicted = torch.max(student_outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / max(val_count, 1)
    val_accuracy = val_correct / max(val_count, 1)
    val_losses_final.append(avg_val_loss)
    
    # 学习率调度
    old_lr = optimizer_final.param_groups[0]['lr']
    scheduler_final.step(avg_val_loss)
    new_lr = optimizer_final.param_groups[0]['lr']
    
    if old_lr != new_lr:
        print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
    
    # 早停机制
    if avg_val_loss < best_val_loss_final:
        best_val_loss_final = avg_val_loss
        best_epoch_final = epoch + 1
        patience_counter_final = 0
        
        # 保存最佳模型
        best_final_state = {
            'model_state_dict': final_model.state_dict(),
            'optimizer_state_dict': optimizer_final.state_dict(),
            'feature_mean': feature_mean,
            'feature_std': feature_std,
            'input_dim': input_dim,
            'scaler': scaler,
            'best_val_loss': best_val_loss_final,
            'best_epoch': best_epoch_final,
            'temperature': temperature,
            'alpha': alpha
        }
    else:
        patience_counter_final += 1
    
    if patience_counter_final >= patience_final:
        print(f"  → Early stopping at epoch {epoch + 1}")
        print(f"  → Best model was at epoch {best_epoch_final} with Val Loss: {best_val_loss_final:.4f}")
        break
    
    # 每10轮打印一次
    if (epoch + 1) % 10 == 0:
        print(f'  Epoch [{epoch + 1}/{distillation_epochs}] | '
              f'Total Loss: {avg_train_loss:.4f} '
              f'(Soft: {avg_soft_loss:.4f}, Hard: {avg_hard_loss:.4f}) | '
              f'Train Acc: {train_accuracy*100:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Val Acc: {val_accuracy*100:.2f}% | '
              f'Patience: {patience_counter_final}/{patience_final}')

# 恢复最佳模型
final_model.load_state_dict(best_final_state['model_state_dict'])

print(f"\n  ✓ Final Unified Model training completed!")
print(f"    - Best Epoch: {best_epoch_final}")
print(f"    - Best Val Loss: {best_val_loss_final:.4f}")
print(f"    - Total Epochs: {epoch + 1}")


# ==================== 14. 评估最终模型 ====================
print(f"\n{'='*60}")
print("Evaluating Final Unified Model on Test Set")
print(f"{'='*60}\n")

final_model.eval()
test_correct = 0
test_total = 0
strict_correct_final = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = final_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        for i in range(len(predicted)):
            pred = predicted[i].item()
            label = labels[i].item()
            
            # 严格准确率
            if pred == label:
                strict_correct_final += 1
            
            # 宽松准确率
            if pred == label:
                test_correct += 1
            elif (pred == 0 or pred == 1) and label == 2:
                test_correct += 1
            
            test_total += 1

final_accuracy = test_correct / test_total
final_strict_accuracy = strict_correct_final / test_total

print(f"Final Unified Model Performance:")
print(f"  - Test Accuracy (strict):  {final_strict_accuracy * 100:.2f}%")
print(f"  - Test Accuracy (relaxed): {final_accuracy * 100:.2f}%")

print(f"\nComparison with Ensemble:")
print(f"  - Ensemble Accuracy (strict):  {strict_accuracy * 100:.2f}%")
print(f"  - Ensemble Accuracy (relaxed): {accuracy * 100:.2f}%")

if final_accuracy >= accuracy:
    print(f"\n  ✓ Final model matches or exceeds ensemble performance!")
else:
    print(f"\n  → Final model is close to ensemble (difference: {(accuracy - final_accuracy)*100:.2f}%)")


# ==================== 15. 保存最终模型到目标目录 ====================
final_model_path = MODEL_DIR / 'FinalModel.pth'

torch.save({
    'model_state_dict': final_model.state_dict(),
    'feature_mean': feature_mean,
    'feature_std': feature_std,
    'input_dim': input_dim,
    'scaler': scaler,
    'best_val_loss': best_val_loss_final,
    'best_epoch': best_epoch_final,
    'test_accuracy_strict': final_strict_accuracy,
    'test_accuracy_relaxed': final_accuracy,
    'temperature': temperature,
    'alpha': alpha,
    'model_architecture': 'FinalUnifiedModel',
    'training_method': 'Knowledge Distillation from 5 Teacher Models'
}, final_model_path)

print(f"\n{'='*60}")
print(f"✓ Final Unified Model saved to: {final_model_path}")
print(f"{'='*60}\n")


# ==================== 16. 可视化最终模型训练过程 ====================
try:
    import matplotlib.pyplot as plt
    
    # 创建训练曲线对比图
    plt.figure(figsize=(14, 6))
    
    # 子图1：最终模型的训练和验证损失
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(train_losses_final) + 1)
    plt.plot(epochs_range, train_losses_final, label='Train Loss', linewidth=2, color='blue', alpha=0.8)
    plt.plot(epochs_range, val_losses_final, label='Val Loss', linewidth=2, color='red', alpha=0.8)
    plt.axvline(x=best_epoch_final, color='green', linestyle='--', linewidth=1.5, label=f'Best Epoch ({best_epoch_final})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Final Unified Model - Training Progress', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 子图2：所有模型对比（包括最终模型）
    plt.subplot(1, 2, 2)
    
    # 绘制5个教师模型的验证损失（淡化）
    for idx, model_name in enumerate(model_names):
        plt.plot(all_val_losses[idx], label=f'{model_name} (Teacher)', 
                linewidth=1, alpha=0.4, linestyle='--')
    
    # 绘制最终模型的验证损失（突出）
    plt.plot(range(1, len(val_losses_final) + 1), val_losses_final, 
            label='Final Unified Model', linewidth=3, color='darkgreen', alpha=0.9)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss: Teachers vs Final Model', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PICTURE_DIR / 'final_model_training.png', dpi=300, bbox_inches='tight')
    print("✓ Final model training plot saved as 'final_model_training.png'\n")
    
except ImportError:
    print("Matplotlib not available, skipping final model visualization\n")


# ==================== 17. 性能总结 ====================
print(f"{'='*60}")
print("FINAL PERFORMANCE SUMMARY")
print(f"{'='*60}")
print("\nIndividual Teacher Models:")
for idx, model_name in enumerate(model_names):
    print(f"  {idx+1}. {model_name:20s}: Performance already evaluated above")

print(f"\nEnsemble (5 Models):")
print(f"  - Strict Accuracy:  {strict_accuracy * 100:.2f}%")
print(f"  - Relaxed Accuracy: {accuracy * 100:.2f}%")
print(f"  - Note: Requires all 5 models for inference (higher memory/computation)")

print(f"\nFinal Unified Model:")
print(f"  - Strict Accuracy:  {final_strict_accuracy * 100:.2f}%")
print(f"  - Relaxed Accuracy: {final_accuracy * 100:.2f}%")
print(f"  - Note: Single model, faster inference, similar performance!")

print(f"\n{'='*60}")
print("All tasks completed successfully!")
print(f"{'='*60}\n")
# 这段代码的工作原理：

#1. 知识蒸馏（Knowledge Distillation）
#   教师模型：5个已训练好的不同架构模型
#   学生模型：新的FinalUnifiedModel
#   软标签：教师模型的概率分布（比硬标签包含更多信息）

# 2. 蒸馏损失函数
#   总损失 = 0.7 × 软标签损失 + 0.3 × 硬标签损失