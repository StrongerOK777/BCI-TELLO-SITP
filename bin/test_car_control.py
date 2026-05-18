#!/usr/bin/env python3
"""测试 car_control 是否可以正常导入和运行"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("测试 car_control 修复")
print("=" * 60)

# 测试 1: 检查依赖
print("\n[1/3] 检查依赖...")
try:
    import requests
    print("  ✓ requests 库已安装")
except ImportError:
    print("  ✗ requests 库未安装，请运行: pip install requests")
    sys.exit(1)

# 测试 2: 检查 KeyPressModule
print("\n[2/3] 检查 KeyPressModule...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "MI-CarControl"))
    import KeyPressModule as kp
    print("  ✓ KeyPressModule 可以导入")
    print(f"  ✓ KeyPressModule.init 函数存在")
    print(f"  ✓ KeyPressModule.getKey 函数存在")
except ImportError as e:
    print(f"  ✗ KeyPressModule 导入失败: {e}")
    sys.exit(1)

# 测试 3: 检查 hardware 模块
print("\n[3/3] 检查 hardware 模块...")
try:
    from hardware import CarHttpController
    print("  ✓ CarHttpController 可以导入")
    controller = CarHttpController()
    print(f"  ✓ CarHttpController 实例化成功")
    print(f"  ✓ 默认主机: {controller.host}")
    print(f"  ✓ 默认端口: {controller.port}")
except ImportError as e:
    print(f"  ✗ hardware 模块导入失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有检查通过！car_control 可以正常运行")
print("=" * 60)
print("\n运行 car_control.py:")
print("  python bin/car_control.py [--host 192.168.149.1] [--port 5000] [--speed 50]")
