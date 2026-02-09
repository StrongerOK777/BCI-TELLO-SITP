# ==================== 加载和使用最终模型的示例代码 ====================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 定义模型架构（需要与训练时一致）
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


# 加载模型
checkpoint = torch.load('FinalModel.pth')

# 重建模型
model = FinalUnifiedModel(checkpoint['input_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 获取标准化参数
feature_mean = checkpoint['feature_mean']
feature_std = checkpoint['feature_std']
scaler = checkpoint['scaler']

print(f"Model loaded successfully!")
print(f"Test Accuracy (strict): {checkpoint['test_accuracy_strict']*100:.2f}%")
print(f"Test Accuracy (relaxed): {checkpoint['test_accuracy_relaxed']*100:.2f}%")

# 使用模型进行预测
def predict(features):
    """
    预测函数
    features: numpy array, shape (n_samples, n_features)
    返回: 预测的类别
    """
    # 标准化
    features_normalized = (features - feature_mean) / (feature_std + 1e-8)
    
    # 转换为tensor
    features_tensor = torch.tensor(features_normalized, dtype=torch.float32)
    
    # 预测
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.numpy()

# 示例使用
# new_data = np.array([...])  # 你的新数据
# predictions = predict(new_data)
# print(f"Predictions: {predictions}")
# 0: 左手, 1: 右手, 2: 休息
