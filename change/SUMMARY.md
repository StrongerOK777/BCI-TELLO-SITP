# BCI 脑环信号修复总结

## 问题背景

用户的 BCI 小车系统无法接收脑环（MindWave）信号，表现为：
- `attention=0 meditation=0`（脑电波未读取）
- 脑环虽然已连接到 COM5，但系统一直报告"脑环信号异常"

## 根本原因分析

1. **NeuroSkyPy 导入方式不稳定** - 原始代码使用复杂的模块导入逻辑，容易失败
2. **错误信息不清晰** - 用户无法区分是硬件问题还是代码问题
3. **前后控制逻辑不一致** - 与参考的 predic.py 逻辑不同步
4. **缺少测试模式** - 无法独立验证脑环连接

## 修改文件列表

### 1. `bin/eeg.py`（核心EEG模块）

**关键改动：**

#### a) 改进 `import_neurosky()` 函数（第65-100行）
```python
# 原始方式（复杂且容易失败）：使用 importlib.import_module()

# 新方式（直接且稳定）：
# 1. 尝试直接导入 neuropy
# 2. 尝试从指定目录导入
# 3. 尝试从环境变量指向的目录导入
```

**效果**：更稳定地导入 NeuroSkyPy，参考 predic.py 的成功经验

#### b) 增强 `start()` 方法日志（第197-207行）
```python
print(f"[BrainSignalReader] 初始化脑环设备...")
print(f"[BrainSignalReader] 端口: {self.port}, 波特率: {self.baud}")
print(f"[BrainSignalReader] NeuroSkyPy 加载成功")
print(f"[BrainSignalReader] 启动脑环设备...")
print(f"[BrainSignalReader] 脑环设备启动成功")
```

**效果**：用户可以看到初始化进度，快速定位问题

#### c) 改进错误原因提示（第243-248行）
```python
# 区分两种失败原因
if snapshot.attention == 0 or snapshot.meditation == 0:
    result.reason = f"脑电波未读取：attention={snapshot.attention} meditation={snapshot.meditation}"
else:
    result.reason = f"poorSignal={snapshot.poorSignal}"
```

**效果**：清楚地区分是脑环未连接还是信号质量差

### 2. `MI-CarControl/brain_control.py`（小车控制主程序）

**关键改动：**

#### a) 添加命令行参数（第182-186行）
```python
parser.add_argument("--test-brain", action="store_true", 
                    help="Test brain signal reading only (no car control)")
parser.add_argument("--dry-run", action="store_true", 
                    help="Test mode without connecting to car")
```

#### b) 添加 `--test-brain` 测试模式（第229-276行）

功能：
- 仅读取脑环数据，不连接小车
- 显示每个窗口的有效性
- 统计有效窗口比例
- 显示原始数据（Attention、Meditation等）

使用方式：
```bash
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5
```

#### c) 添加 `--dry-run` 模拟模式（第279-308行）

功能：
- 读取脑环数据
- 执行控制逻辑
- 使用模拟小车客户端（不真正连接硬件）

使用方式：
```bash
python MI-CarControl/brain_control.py --dry-run --mindwave-port COM5
```

#### d) 改进 `handle_forward_backward_mode()` 逻辑（第105-125行）

**参考 predic.py 的控制逻辑：**
```python
# 原始逻辑：attention 或 meditation 任一低于阈值就停止
if (result.attention_count < min_count or result.meditation_count < min_count):
    stop()

# 新逻辑：两者都低于阈值才停止（允许局部高）
if (result.attention_count < min_count and result.meditation_count < min_count):
    stop()

# 前后控制
if result.attention_count >= result.meditation_count:
    forward()  # Attention 更高 → 前进
else:
    backward()  # Meditation 更高 → 后退
```

**效果**：前后控制更灵活，与无人机脑控程序逻辑保持一致

### 3. `test_brain.py`（独立脑环测试脚本）

功能：
- 独立验证脑环硬件连接
- 不依赖小车系统
- 显示原始数据和统计信息

使用方式：
```bash
python test_brain.py --port COM5 --duration 10
```

## 使用工作流

### 第一步：验证脑环连接
```bash
# 方式 1：使用独立测试脚本
python test_brain.py --port COM5 --duration 10

# 方式 2：使用 brain_control 测试模式
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5
```

**预期输出**：
```
[BrainSignalReader] 脑环设备启动成功
✓ 脑环已连接，开始采集数据...

[1] ✓ 有效窗口 | Attention: 5, Meditation: 3, Blinks: 0
      原始数据: A=35, M=28, Signal=0, Blink=0
```

### 第二步：调整参数（如果信号质量差）
```bash
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5 \
    --attention-threshold 10 \
    --meditation-threshold 10 \
    --poor-signal-threshold 100
```

### 第三步：运行完整系统
```bash
python MI-CarControl/brain_control.py --mindwave-port COM5 \
    --host 192.168.149.1 --port 5000
```

## 前后控制对比

### 原始逻辑（有问题）
```
if attention_count < 20 or meditation_count < 20:
    停止
if attention_count >= meditation_count:
    前进
else:
    后退
```
**问题**：任一数值低就停止，无法有效控制

### 修复后的逻辑（参考 predic.py）
```
if attention_count < 20 and meditation_count < 20:
    停止（两者都低）
if attention_count >= meditation_count:
    前进
else:
    后退
```
**优势**：
- ✅ 允许单个数值较低（例如冥想值低但注意力高）
- ✅ 只在两个数值都太低时停止
- ✅ 与无人机脑控程序逻辑一致

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--mindwave-port` | COM5 | 脑环设备端口 |
| `--attention-threshold` | 30 | Attention 计数阈值 |
| `--meditation-threshold` | 50 | Meditation 计数阈值 |
| `--poor-signal-threshold` | 20 | 信号质量阈值（0最好） |
| `--window-size` | 30 | 每窗口采样数 |
| `--min-decision-count` | 20 | 最少有效样本数 |

## 故障排查

| 错误 | 原因 | 解决方案 |
|------|------|--------|
| `PermissionError: Access is denied` | COM5 被占用 | 关闭其他占用该端口的程序 |
| `脑电波未读取：attention=0` | 脑环未连接 | 检查硬件、USB连接、驱动 |
| `Could not import NeuroSkyPy` | 库找不到 | 检查 `--neuropy-dir` 参数 |
| 信号一直无效 | 信号质量差或阈值过高 | 使用 `--poor-signal-threshold 100` 等参数 |

## 文件清单

```
change/
├── eeg.py                    # 核心 EEG 模块（bin/eeg.py 的修复版）
├── brain_control.py          # 小车脑环控制（MI-CarControl/brain_control.py 的修复版）
├── test_brain.py             # 独立脑环测试脚本
└── SUMMARY.md                # 本总结文档
```

## 测试建议

1. **先测试硬件**：`python test_brain.py`
2. **再测试脑环模式**：`python MI-CarControl/brain_control.py --test-brain`
3. **然后测试模拟模式**：`python MI-CarControl/brain_control.py --dry-run`
4. **最后完整系统**：`python MI-CarControl/brain_control.py`

## 参考资料

- 参考 `predic.py` 的 NeuroSkyPy 使用方式
- 参考 `predic.py` 的前后控制逻辑
- NeuroSkyPy 文档：MI-CarControl/neuropy.py（第43-58行类说明）
