# ==================== 13. 知识蒸馏：合成最终模型 ====================
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
distillation_epochs = 200
temperature = 3.0  # 温度参数
alpha = 0.7  # 软标签权重
best_val_loss_final = float("inf")
patience_final = 30
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


# ==================== 15. 保存最终模型到根目录 ====================
final_model_path = 'FinalModel.pth'

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
    plt.savefig('final_model_training.png', dpi=300, bbox_inches='tight')
    print("✓ Final model training plot saved as 'final_model_training.png'\n")
    
except ImportError:
    print("Matplotlib not available, skipping final model visualization\n")


