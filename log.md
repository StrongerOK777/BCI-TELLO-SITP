# 更新日志（fanta 分支）

## 2026-05-17｜提交 658f102（"Add files via upload"）

### 概览
本次提交主要围绕 **小车（car_control）键盘控制在 Windows 下的可用性** 做了完整修复，同时新增图形化控制界面与多硬件“超级控制中心”入口，并补充了使用说明与快速启动脚本。

### 关键改动
- **跨平台键盘输入修复**：`MI-CarControl/KeyPressModule.py` 增加 Windows 版本（`msvcrt`），并加入按键缓冲机制，避免多次 `getKey()` 造成按键丢失。
- **键盘控制逻辑修复**：`bin/car_control.py` 的按键检测逻辑改为与 `test.py` 一致（SPACE 独立检查、j/l 与 i/k 各自成对），并改为“每次都发送”以避免信号丢失。
- **改进错误提示与依赖检查**：`bin/car_control.py` 增加 `requests` 依赖检查、网络错误提示，并提升启动信息可读性。
- **相对导入兼容**：`bin/hardware.py`、`bin/car_control.py` 增加脚本直接运行时的 fallback 导入。
- **新增 GUI 控制界面**：
  - `bin/car_control_ui.py`：Tkinter 图形界面版本的键盘/鼠标控制。
  - `bin/supercontrol.py`：pygame 实现的多硬件“超级控制中心”。
- **新增脚本与说明**：
  - 启动脚本：`car_control.bat`、`car_control.ps1`
  - 测试/验证脚本：`bin/test_car_control.py`、`bin/verify_fix.py`
  - 说明文档：`CAR_CONTROL_FIX.md`、`KEYBOARD_FIX.md`
  - 快速指南：`快速启动.txt`、`快速对比.txt`

### 文件变更清单
**新增**
- `CAR_CONTROL_FIX.md`
- `KEYBOARD_FIX.md`
- `bin/car_control_ui.py`
- `bin/supercontrol.py`
- `bin/test_car_control.py`
- `bin/verify_fix.py`
- `car_control.bat`
- `car_control.ps1`
- `快速启动.txt`
- `快速对比.txt`

**修改**
- `MI-CarControl/KeyPressModule.py`
- `bin/car_control.py`
- `bin/hardware.py`

**新增（编译缓存）**
- `MI-CarControl/__pycache__/KeyPressModule.cpython-313.pyc`
- `bin/__pycache__/__init__.cpython-313.pyc`
- `bin/__pycache__/car_control.cpython-313.pyc`
- `bin/__pycache__/car_control_ui.cpython-313.pyc`
- `bin/__pycache__/eeg.cpython-313.pyc`
- `bin/__pycache__/hardware.cpython-313.pyc`
- `bin/__pycache__/supercontrol.cpython-313.pyc`

### 备注
- 若希望仓库保持干净，可考虑将 `__pycache__/` 与 `*.pyc` 加入 `.gitignore`，并从版本库移除已提交的缓存文件。
