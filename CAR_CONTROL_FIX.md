# 🚗 BCI 小车控制 - 修复说明

## 修复内容

本次修复解决了以下问题，使 `car_control` 可以在 Windows 上直接点击运行：

### 1. **跨平台键盘输入支持** ✅
   - 问题：`MI-CarControl/KeyPressModule.py` 使用 Unix/Linux 专用的 `tty`/`termios` 模块
   - 修复：添加了 Windows 支持，在 Windows 上使用 `msvcrt` 模块
   - 文件：`MI-CarControl/KeyPressModule.py`

### 2. **改进错误处理** ✅
   - 问题：缺少依赖检查和错误处理
   - 修复：
     - 添加了 `requests` 库依赖检查
     - 添加了具体的网络错误提示
     - 改进了用户友好的错误信息
   - 文件：`bin/car_control.py`

### 3. **修复相对导入问题** ✅
   - 问题：`bin/hardware.py` 的相对导入在直接运行时失败
   - 修复：添加了 fallback 以支持直接脚本运行
   - 文件：`bin/hardware.py`

### 4. **增强用户界面** ✅
   - 添加了更清晰的启动信息
   - 添加了控制说明和故障诊断信息
   - 文件：`bin/car_control.py`

## 使用方法

### 方式 1：直接双击批处理文件（推荐）⭐
```
双击：car_control.bat
```
或从命令行运行：
```powershell
cd "d:\A SITP\BCI-TELLO-SITP-main\BCI-TELLO-SITP-main"
car_control.bat
```

### 方式 2：使用 PowerShell 脚本
```powershell
cd "d:\A SITP\BCI-TELLO-SITP-main\BCI-TELLO-SITP-main"
powershell -ExecutionPolicy Bypass -File car_control.ps1
```

### 方式 3：从命令行直接运行
```bash
cd "d:\A SITP\BCI-TELLO-SITP-main\BCI-TELLO-SITP-main"
python bin/car_control.py
```

使用自定义参数：
```bash
python bin/car_control.py --host 192.168.149.1 --port 5000 --speed 50
```

## 控制说明

| 按键 | 功能 |
|------|------|
| **i** | 前进 |
| **k** | 后退 |
| **j** | 左转 |
| **l** | 右转 |
| **SPACE** | 停止 |
| **Ctrl+C** | 退出程序 |

## 依赖要求

- Python 3.8 或更高版本
- `requests` 库（用于 HTTP 通信）

若未安装依赖，请运行：
```bash
pip install requests
```

## 常见问题排查

### 问题：找不到树莓派（连接错误）
```
✗ 连接错误: 无法连接到 192.168.149.1:5000
```
**解决方案：**
1. 检查树莓派是否在线
2. 检查网络连接是否正常
3. 确认树莓派上的服务正在运行
4. 使用自定义主机和端口：
   ```bash
   python bin/car_control.py --host <树莓派IP> --port <端口>
   ```

### 问题：Python 未安装
```
错误：未检测到 Python 环境
```
**解决方案：**
1. 下载并安装 Python 3.8+ 从 https://www.python.org
2. 确保在安装时勾选 "Add Python to PATH"
3. 重新启动命令行窗口

### 问题：缺少 requests 库
```
✗ 错误：缺少 requests 库
```
**解决方案：**
```bash
pip install requests
```

## 测试脚本

可以使用提供的测试脚本验证所有依赖是否正确安装：
```bash
python bin/test_car_control.py
```

## 技术细节

### KeyPressModule 跨平台实现
- **Windows**: 使用 `msvcrt.kbhit()` 和 `msvcrt.getch()`
- **Unix/Linux**: 使用 `tty.setraw()` 和 `termios`
- **优点**: 非阻塞式键盘输入，不影响程序响应性

### 错误处理
- 网络连接错误会显示具体的错误信息和建议
- 缺少依赖时会给出安装提示
- 程序崩溃时会显示完整的错误堆栈跟踪

## 修改的文件列表

1. ✅ `MI-CarControl/KeyPressModule.py` - 添加 Windows 支持
2. ✅ `bin/car_control.py` - 改进错误处理和用户界面
3. ✅ `bin/hardware.py` - 修复相对导入问题
4. ✅ `bin/test_car_control.py` - 新增依赖检查脚本
5. ✅ `car_control.bat` - 新增 Windows 批处理启动脚本
6. ✅ `car_control.ps1` - 新增 PowerShell 启动脚本

## 下一步

现在你可以：
1. ✅ 直接双击 `car_control.bat` 启动程序
2. ✅ 或使用 `python bin/car_control.py` 从命令行启动
3. ✅ 按照控制说明用键盘控制小车

祝使用愉快！🎮
