# BCI 脑环信号修复包

## 📋 包含内容

本目录包含 BCI 小车脑环系统的完整修复方案，用于解决信号接收 `attention=0, meditation=0` 的问题。

```
change/
├── README.md                     # 本文件（使用指南）
├── SUMMARY.md                    # 详细修改总结
├── 快速使用指南.md               # 快速诊断和测试命令
├── 文件修改对照表.md             # 原代码 vs 修复代码对比
├── eeg.py                        # ✅ 修复的 EEG 模块
├── brain_control.py              # ✅ 修复的脑环控制程序
└── test_brain.py                 # ✨ 新增：独立脑环测试脚本
```

## 🚀 快速开始（3 步）

### 第一步：备份原文件
```bash
# 可选，但建议做
cp bin/eeg.py bin/eeg.py.backup
cp MI-CarControl/brain_control.py MI-CarControl/brain_control.py.backup
```

### 第二步：应用修复
```bash
# 将修复的文件复制到相应位置
cp change/eeg.py bin/eeg.py
cp change/brain_control.py MI-CarControl/brain_control.py
cp change/test_brain.py test_brain.py
```

### 第三步：验证修复
```bash
# 测试脑环连接
python test_brain.py --port COM5 --duration 10
```

**预期输出**：
```
✓ 脑环已连接
总采样数: 20
有效样本: 15 (75.0%)
✓ 脑环工作正常！
```

## 📚 文档导航

| 文档 | 内容 | 适用场景 |
|------|------|--------|
| **SUMMARY.md** | 详细的修改说明和原理 | 需要了解改动细节 |
| **快速使用指南.md** | 命令速查、参数说明 | 快速查看常用命令 |
| **文件修改对照表.md** | 代码对比、逐行改动 | 代码审查、理解变化 |

## 🔍 三种诊断模式

### 模式 1：硬件测试
```bash
python test_brain.py --port COM5 --duration 10
```
**用途**：验证脑环硬件是否正常工作  
**输出**：原始数据和统计百分比

### 模式 2：脑环信号测试
```bash
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5
```
**用途**：测试脑环数据读取和窗口有效性  
**输出**：每个采集窗口的统计结果

### 模式 3：完整系统测试（模拟）
```bash
python MI-CarControl/brain_control.py --dry-run --mindwave-port COM5
```
**用途**：测试脑环 + 控制逻辑（不连接真实小车）  
**输出**：模拟的控制命令

## 🔧 关键改动

### ✨ 改动 1：稳定的导入方式
- **文件**：eeg.py
- **问题**：原始 importlib 方式不稳定
- **方案**：改为直接导入，参考 predic.py 成功经验
- **效果**：更可靠的 NeuroSkyPy 加载

### ✨ 改动 2：增强的日志
- **文件**：eeg.py
- **问题**：用户看不到初始化进度
- **方案**：添加详细的初始化日志
- **效果**：快速定位问题

### ✨ 改动 3：清晰的错误提示
- **文件**：eeg.py
- **问题**：无法区分硬件问题 vs 信号质量问题
- **方案**：区分错误原因，显示具体数值
- **效果**：加快故障排查

### ✨ 改动 4：改进的控制逻辑
- **文件**：brain_control.py
- **问题**：前后控制使用 `or` 逻辑，太敏感
- **方案**：改为 `and` 逻辑，两者都低才停止
- **效果**：控制更灵活，与 predic.py 一致

### ✨ 改动 5：新增测试模式
- **文件**：brain_control.py
- **问题**：无法测试脑环而不触发小车错误
- **方案**：添加 `--test-brain` 和 `--dry-run` 参数
- **效果**：独立诊断各组件

## ❓ 常见问题

### Q1：应该先做什么？
**A**：按顺序：
1. 运行 `python test_brain.py` 确认硬件
2. 运行 `python MI-CarControl/brain_control.py --test-brain` 确认读取
3. 运行 `python MI-CarControl/brain_control.py --dry-run` 测试逻辑
4. 运行完整系统

### Q2：看到 `attention=0 meditation=0` 怎么办？
**A**：这是信号未被读取，通常是：
1. 脑环未连接或无电
2. 驱动程序问题
3. COM 端口被占用

使用 `python test_brain.py` 诊断。

### Q3：信号一直是 `poorSignal=200` 什么意思？
**A**：信号质量差，尝试：
```bash
# 降低信号阈值
python MI-CarControl/brain_control.py --test-brain \
    --poor-signal-threshold 100
```

### Q4：能回滚到原始代码吗？
**A**：可以：
```bash
# 使用 git 回滚
git checkout bin/eeg.py MI-CarControl/brain_control.py

# 或用你的备份
cp bin/eeg.py.backup bin/eeg.py
cp MI-CarControl/brain_control.py.backup MI-CarControl/brain_control.py
```

### Q5：需要修改什么参数吗？
**A**：通常不需要。默认参数已经调优过，但如果：
- 信号质量差 → 增加 `--poor-signal-threshold`
- 需要更灵敏 → 降低 `--attention-threshold` 等
- 需要更稳定 → 提高阈值

详见 **快速使用指南.md**。

## 📊 测试清单

在使用之前，建议依次完成：

- [ ] 备份原文件
- [ ] 复制修复文件到对应位置
- [ ] 运行硬件测试（test_brain.py）
- [ ] 运行脑环测试（--test-brain 模式）
- [ ] 运行模拟测试（--dry-run 模式）
- [ ] 根据需要调整参数
- [ ] 运行完整系统测试

## 📞 需要帮助？

1. **查看详细说明** → 打开 SUMMARY.md
2. **快速查找命令** → 打开 快速使用指南.md
3. **代码对比** → 打开 文件修改对照表.md
4. **技术问题** → 参考 bin/eeg.py 和 MI-CarControl/brain_control.py 源代码注释

## ✅ 验证清单

应用修复后，确认以下几点：

```bash
# 1. 日志中能看到初始化过程
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5 2>&1 | grep BrainSignalReader

# 预期输出：
# [BrainSignalReader] 初始化脑环设备...
# [BrainSignalReader] 端口: COM5, 波特率: 57600
# [BrainSignalReader] NeuroSkyPy 加载成功
# [BrainSignalReader] 启动脑环设备...
# [BrainSignalReader] 脑环设备启动成功

# 2. 错误消息清晰
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5 2>&1 | grep "脑电波"

# 预期输出（如果有错误）：
# 脑电波未读取：attention=0 meditation=0

# 3. 可以看到 Attention/Meditation 数值
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5 2>&1 | grep "前后模式"
```

## 🎯 下一步

1. ✅ 应用修复（复制文件）
2. ✅ 验证修复（运行测试）
3. ✅ 调整参数（如需要）
4. ✅ 使用系统（完整控制）

---

**修复包版本**：1.0  
**最后更新**：2024年  
**兼容版本**：Python 3.7+，支持 Windows/Linux
