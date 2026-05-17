# car_control.ps1 - PowerShell 脚本用于启动小车控制

# 获取当前脚本目录
$scriptPath = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Set-Location $scriptPath

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BCI 小车控制程序" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[✓] Python 已安装: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] 错误：未检测到 Python 环境" -ForegroundColor Red
    Write-Host "    请先安装 Python 3.8 或更高版本" -ForegroundColor Red
    Read-Host "按 Enter 键退出"
    exit 1
}

Write-Host ""
Write-Host "[*] 启动参数：" -ForegroundColor Yellow
Write-Host "    主机: 192.168.149.1"
Write-Host "    端口: 5000"
Write-Host "    速度: 50"
Write-Host ""
Write-Host "[*] 按键控制说明：" -ForegroundColor Yellow
Write-Host "    i 键：前进"
Write-Host "    k 键：后退"
Write-Host "    j 键：左转"
Write-Host "    l 键：右转"
Write-Host "    空格：停止"
Write-Host "    Ctrl+C：退出程序"
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 运行程序
python bin\car_control.py --host 192.168.149.1 --port 5000 --speed 50

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[✗] 程序出错" -ForegroundColor Red
    Read-Host "按 Enter 键退出"
    exit 1
}

Read-Host "按 Enter 键退出"
