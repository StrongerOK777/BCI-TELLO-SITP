@echo off
REM car_control.bat - Windows 批处理文件用于启动小车控制
REM 可以直接双击运行

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ========================================
echo   BCI 小车控制程序
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到 Python 环境
    echo 请先安装 Python 3.8 或更高版本
    pause
    exit /b 1
)

REM 显示启动信息
echo [*] 启动参数：
echo     主机: 192.168.149.1
echo     端口: 5000
echo     速度: 50
echo.
echo [*] 按键控制说明：
echo     i 键：前进
echo     k 键：后退
echo     j 键：左转
echo     l 键：右转
echo     空格：停止
echo     Ctrl+C：退出程序
echo.
echo ========================================
echo.

python bin\car_control.py --host 192.168.149.1 --port 5000 --speed 50

if errorlevel 1 (
    echo.
    echo 程序出错，按任意键继续...
    pause
    exit /b 1
)

pause
