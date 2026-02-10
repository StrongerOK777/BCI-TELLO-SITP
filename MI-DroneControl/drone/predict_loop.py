import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuropy import NeuroSkyPy


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FinalUnifiedModel(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    return model, feature_mean, feature_std


def predict_loop():
    model, feature_mean, feature_std = load_model()

    port = os.getenv("MINDWAVE_PORT", "/dev/cu.usbmodem2017_2_251")
    baud = int(os.getenv("MINDWAVE_BAUD", "57600"))
    neuropy = NeuroSkyPy(port, baud)
    neuropy.start()

    labelmap = ["left", "right", "rest"]

    try:
        while True:
            window = []
            start = time.time()
            target_interval = 0.1
            while len(window) < 30:
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
                if poorSignal >= 20:
                    print(f"PoorSignal too high: {poorSignal}, reset window...")
                    window.clear()
                    start = time.time()
                    time.sleep(1)
                    continue
                feature_vector = build_feature_vector(
                    attention, meditation, delta, theta, low_alpha, high_alpha,
                    low_beta, high_beta, low_gamma, mid_gamma, blink_strength
                )
                window.append(feature_vector)
                print(f"当前读取数据：attention={attention} meditation={meditation} blinkStrength={neuropy.blinkStrength}")
                target_time = start + len(window) * target_interval
                sleep_time = target_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if len(window) == 0:
                print("未采集到有效数据，等待下一轮...")
                continue

            window_data = np.array(window, dtype=np.float32)
            mean_feat = window_data.mean(axis=0)
            std_feat = window_data.std(axis=0)
            combined = np.concatenate([mean_feat, std_feat], axis=0)

            if feature_mean is not None and feature_std is not None:
                combined = (combined - feature_mean) / (feature_std + 1e-6)

            input_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                predicted = torch.argmax(outputs, dim=1).item()

            duration = time.time() - start
            print(f"预测结果：{labelmap[predicted]} | {len(window)}/30组, 用时{duration:.2f}s")
    except KeyboardInterrupt:
        pass
    finally:
        neuropy.stop()


if __name__ == "__main__":
    predict_loop()
