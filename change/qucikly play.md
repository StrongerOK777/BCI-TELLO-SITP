# 快速使用指南

## 如何应用这些修改

### 选项 1：手动覆盖文件
```bash
# 将修复的文件复制到相应位置
cp change/eeg.py bin/eeg.py
cp change/brain_control.py MI-CarControl/brain_control.py
cp change/test_brain.py test_brain.py
```

### 选项 2：逐个对比修改
详见 SUMMARY.md 中的具体改动位置

---

## 快速诊断流程

### 步骤 1：验证脑环硬件（2分钟）
```bash
python test_brain.py --port COM5 --duration 10
```

**如果看到：**
- ✅ `✓ 脑环工作正常！` → 继续步骤 2
- ❌ `✗ 脑环未读取到有效数据` → 检查硬件连接

### 步骤 2：验证脑环模式（3分钟）
```bash
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5
```

**按 Ctrl+C 停止**

**如果看到：**
- ✅ `[10] ✓ 有效窗口 | Attention: 5, Meditation: 3` → 继续步骤 3
- ❌ 全是 `✗ 无效窗口` → 尝试降低阈值（见步骤 3B）

### 步骤 3A：完整系统测试（需要小车）
```bash
python MI-CarControl/brain_control.py --mindwave-port COM5 \
    --host 192.168.149.1 --port 5000
```

### 步骤 3B：如果信号质量差，调整参数
```bash
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5 \
    --poor-signal-threshold 100 \
    --attention-threshold 10 \
    --meditation-threshold 10
```

---

## 控制逻辑速查表

### 前后模式（双眨眼前从转向模式切回）

| 注意力值 | 冥想值 | 结果 | 说明 |
|---------|-------|------|------|
| 25 | 20 | 停止 | 两者都低 |
| 25 | 5 | 前进 | 注意力更高 |
| 5 | 25 | 后退 | 冥想更高 |
| 30 | 10 | 前进 | 注意力高 |

### 转向模式（模型预测）

| 预测结果 | 动作 |
|---------|------|
| left | 左转 |
| right | 右转 |
| rest | 停止 |

---

## 常见命令速查

```bash
# 1. 仅测试脑环
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5

# 2. 模拟小车控制（不连接真实小车）
python MI-CarControl/brain_control.py --dry-run --mindwave-port COM5

# 3. 完整控制（需要小车 HTTP 服务器）
python MI-CarControl/brain_control.py --mindwave-port COM5 --host 192.168.149.1

# 4. 降低信号要求
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5 \
    --poor-signal-threshold 100 \
    --attention-threshold 5 \
    --meditation-threshold 5

# 5. 独立脑环硬件测试
python test_brain.py --port COM5 --duration 10
```

---

## 三个模式对比

| 模式 | 命令 | 效果 | 用途 |
|------|------|------|------|
| **Test** | `--test-brain` | 仅读脑环，显示统计 | 诊断脑环 |
| **Dry-run** | `--dry-run` | 读脑环+模拟控制 | 测试逻辑 |
| **正式** | （无参数） | 完整控制系统 | 实际使用 |

---

## 参数调优建议

### 信号质量差的环境
```bash
--poor-signal-threshold 100  # 放宽信号要求
```

### 需要更灵敏的响应
```bash
--attention-threshold 10     # 降低 Attention 阈值
--meditation-threshold 10    # 降低 Meditation 阈值
--min-decision-count 10      # 降低最少样本数
```

### 需要更稳定（减少误触）
```bash
--attention-threshold 50     # 提高 Attention 阈值
--meditation-threshold 70    # 提高 Meditation 阈值
--min-decision-count 25      # 提高最少样本数
```

---

## 关键日志输出

### 正常启动
```
[BrainSignalReader] 初始化脑环设备...
[BrainSignalReader] 端口: COM5, 波特率: 57600
[BrainSignalReader] NeuroSkyPy 加载成功
[BrainSignalReader] 启动脑环设备...
[BrainSignalReader] 脑环设备启动成功
✓ 脑环已连接，开始采集数据...
```

### 有效窗口示例
```
[5] ✓ 有效窗口 | Attention: 8, Meditation: 5, Blinks: 0
    原始数据: A=45, M=32, Signal=0, Blink=0
```

### 无效窗口示例
```
[3] ✗ 无效窗口 | 原因: 脑电波未读取：attention=0 meditation=0
```

---

## 需要帮助？

1. 查看 SUMMARY.md 了解详细改动
2. 尝试不同的参数组合
3. 使用 `--test-brain` 模式诊断
4. 检查 neuropy.py 库是否正确加载
