import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from sklearn.preprocessing import RobustScaler

# ==================== 1. 多种架构的模型定义（架构差异化） ====================

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
        filename = f'model_{idx + 1}_{model_name.replace(" ", "_").lower()}.pth'
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
    plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model comparison plot saved as 'model_comparison.png'")
    
except ImportError:
    print("Matplotlib not available, skipping visualization")

print(f"\n{'='*60}")
print("All tasks completed successfully!")
print(f"{'='*60}")