# 文件修改对照表

## 总览

| 文件 | 原位置 | 修复文件 | 主要改动 | 重要等级 |
|------|-------|--------|--------|--------|
| eeg.py | bin/eeg.py | change/eeg.py | 改进导入方式、增强日志、改进错误提示 | ⭐⭐⭐⭐⭐ |
| brain_control.py | MI-CarControl/ | change/brain_control.py | 添加测试模式、改进控制逻辑 | ⭐⭐⭐⭐⭐ |
| test_brain.py | （新增） | change/test_brain.py | 独立脑环测试 | ⭐⭐⭐⭐ |

---

## 详细对比

### 1. eeg.py 修改清单

#### 改动 1：import_neurosky 函数（第65-100行）

**原始代码（有问题）：**
```python
def import_neurosky(neuropy_dir: Optional[str] = None) -> Any:
    add_sys_path(neuropy_dir)
    add_sys_path(os.getenv("NEUROPY_DIR"))
    candidates = ["neuropy"]
    last_error: Optional[BaseException] = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "NeuroSkyPy")
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Could not import NeuroSkyPy...") from last_error
```

**问题**：
- 使用 importlib 方式过于复杂
- 添加路径后才尝试导入，容易失败
- 错误处理不够清晰

**修复后（更稳定）：**
```python
def import_neurosky(neuropy_dir: Optional[str] = None) -> Any:
    # 尝试 1：直接导入
    try:
        import neuropy
        return neuropy.NeuroSkyPy
    except ImportError:
        pass
    
    # 尝试 2：从指定目录
    if neuropy_dir:
        add_sys_path(neuropy_dir)
        try:
            import neuropy
            return neuropy.NeuroSkyPy
        except ImportError:
            pass
    
    # 尝试 3：从环境变量
    neuropy_env = os.getenv("NEUROPY_DIR")
    if neuropy_env:
        add_sys_path(neuropy_env)
        try:
            import neuropy
            return neuropy.NeuroSkyPy
        except ImportError:
            pass
    
    raise RuntimeError("Could not import NeuroSkyPy...")
```

**优势**：
- ✅ 参考 predic.py 的成功经验
- ✅ 直接导入方式更可靠
- ✅ 层级明确的失败处理

#### 改动 2：start 方法（第197-207行）

**原始代码：**
```python
def start(self) -> None:
    if self.port is None:
        raise RuntimeError("...")
    factory = self.device_factory or import_neurosky(self.neuropy_dir)
    self.device = factory(self.port, self.baud)
    self.device.start()
    self.running = True
```

**修复后：**
```python
def start(self) -> None:
    if self.port is None:
        raise RuntimeError("...")
    print(f"[BrainSignalReader] 初始化脑环设备...")
    print(f"[BrainSignalReader] 端口: {self.port}, 波特率: {self.baud}")
    factory = self.device_factory or import_neurosky(self.neuropy_dir)
    print(f"[BrainSignalReader] NeuroSkyPy 加载成功")
    self.device = factory(self.port, self.baud)
    print(f"[BrainSignalReader] 启动脑环设备...")
    self.device.start()
    print(f"[BrainSignalReader] 脑环设备启动成功")
    self.running = True
```

**优势**：
- ✅ 用户可看到初始化进度
- ✅ 快速定位问题点
- ✅ 提高信心

#### 改动 3：错误原因识别（第243-248行）

**原始代码：**
```python
if (snapshot.poorSignal >= self.poor_signal_threshold
    or snapshot.attention == 0
    or snapshot.meditation == 0):
    result.valid = False
    result.reason = f"poorSignal={snapshot.poorSignal}"
    break
```

**问题**：
- 没有区分是脑电波未读取还是信号质量差
- 用户无法判断是硬件问题还是参数问题

**修复后：**
```python
if (snapshot.poorSignal >= self.poor_signal_threshold
    or snapshot.attention == 0
    or snapshot.meditation == 0):
    result.valid = False
    if snapshot.attention == 0 or snapshot.meditation == 0:
        result.reason = f"脑电波未读取：attention={snapshot.attention} meditation={snapshot.meditation}"
    else:
        result.reason = f"poorSignal={snapshot.poorSignal}"
    break
```

**优势**：
- ✅ 区分两种失败原因
- ✅ 帮助用户快速诊断

---

### 2. brain_control.py 修改清单

#### 改动 1：命令行参数（第182-186行）

**添加：**
```python
parser.add_argument("--test-brain", action="store_true", 
                    help="Test brain signal reading only (no car control)")
parser.add_argument("--dry-run", action="store_true", 
                    help="Test mode without connecting to car")
```

**优势**：
- ✅ 三种运行模式可选
- ✅ 便于诊断和测试

#### 改动 2：main 函数（第210-228行）

**原始代码：**
```python
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        build_controller(config_from_args(args), ...).run()
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
```

**修复后：**
```python
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    
    if args.test_brain:
        return test_brain_signal_mode(args)
    
    if args.dry_run:
        return test_dry_run_mode(args)
    
    try:
        build_controller(config_from_args(args), ...).run()
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
```

**优势**：
- ✅ 支持三种运行模式
- ✅ 提前检测脑环问题

#### 改动 3：handle_forward_backward_mode 方法（第105-125行）

**原始代码：**
```python
def handle_forward_backward_mode(self, result: WindowResult) -> None:
    if result.blink_count >= self.config.mode_switch_blinks:
        self.mode = MODE_TURNING
        print("双眨眼：切换到转向模式")
        return
    if (
        result.attention_count < self.config.min_decision_count
        or result.meditation_count < self.config.min_decision_count
    ):
        self._send("停止")
        return
    self._send("前进" if result.attention_count >= result.meditation_count else "后退")
```

**问题**：
- 任一数值低就停止（`or` 条件）
- 不够灵活，容易误触

**修复后：**
```python
def handle_forward_backward_mode(self, result: WindowResult) -> None:
    if result.blink_count >= self.config.mode_switch_blinks:
        self.mode = MODE_TURNING
        print("双眨眼：切换到转向模式")
        return
    
    print(f"前后模式 | Attention: {result.attention_count}/30, Meditation: {result.meditation_count}/30")
    
    if (
        result.attention_count < self.config.min_decision_count
        and result.meditation_count < self.config.min_decision_count
    ):
        print(f"注意力和冥想都过低（各<{self.config.min_decision_count}），停止")
        self._send("停止")
        return
    
    if result.attention_count >= result.meditation_count:
        self._send("前进")
    else:
        self._send("后退")
```

**改进点**：
- ✅ 条件从 `or` 改为 `and`（两个都低才停止）
- ✅ 添加详细日志（显示 Attention/Meditation 数值）
- ✅ 参考 predic.py 的逻辑

#### 改动 4：新增测试模式函数（第229-276行 & 279-308行）

**新增函数1：test_brain_signal_mode**
```python
def test_brain_signal_mode(args: argparse.Namespace) -> int:
    """Test brain signal reading without car control."""
    # 显示标题和参数
    # 创建读取器
    # 采集并显示每个窗口的数据
    # 统计有效窗口比例
```

**功能**：
- ✅ 仅读脑环，显示统计
- ✅ 用于诊断脑环连接

**新增函数2：test_dry_run_mode**
```python
def test_dry_run_mode(args: argparse.Namespace) -> int:
    """Test mode without car connection."""
    # 创建虚拟小车客户端
    # 执行完整控制逻辑但不连接真实硬件
```

**功能**：
- ✅ 读脑环 + 模拟控制
- ✅ 测试控制逻辑

---

### 3. test_brain.py（新增文件）

**功能**：
- 独立验证脑环硬件
- 无依赖于小车系统
- 输出原始数据和统计信息

**使用**：
```bash
python test_brain.py --port COM5 --duration 10
```

---

## 修改影响范围

### 必需修改（核心功能）
1. ⭐⭐⭐⭐⭐ `bin/eeg.py` - 导入和日志
2. ⭐⭐⭐⭐⭐ `MI-CarControl/brain_control.py` - 控制逻辑和测试模式

### 可选修改（辅助诊断）
3. ⭐⭐⭐⭐ `test_brain.py` - 独立测试脚本

### 建议
- ✅ 必须应用改动 1 和 2
- ✅ 强烈建议添加改动 3 用于诊断

---

## 回滚方法

如需回滚到原始版本，只需还原对应文件：

```bash
git checkout bin/eeg.py
git checkout MI-CarControl/brain_control.py
rm test_brain.py
```

---

## 验证修改

修改后验证：

```bash
# 1. 测试脑环连接
python MI-CarControl/brain_control.py --test-brain --mindwave-port COM5

# 2. 检查日志输出
# 应该看到 [BrainSignalReader] 的初始化过程

# 3. 检查错误消息
# 应该清楚地区分问题原因
```
