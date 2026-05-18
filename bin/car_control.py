"""Keyboard-to-HTTP car control helpers and UI entrypoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

try:
    from .hardware import CarHttpController
except ImportError:
    # 支持直接脚本运行
    sys.path.insert(0, str(Path(__file__).parent))
    from hardware import CarHttpController

try:
    from .supercontrol import SupercontrolUI
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from supercontrol import SupercontrolUI


def key_to_signal(key_module) -> str:
    """按照 test.py 的方式检测按键"""
    signal = "停止"  # 默认值
    
    try:
        # SPACE 独立检查（不使用 elif）
        if key_module.getKey("SPACE"):
            signal = "停止"
        
        # j/l 是 if-elif 对（j 优先）
        if key_module.getKey("j"):
            signal = "左转"
        elif key_module.getKey("l"):
            signal = "右转"
        
        # i/k 是 if-elif 对（i 优先）
        if key_module.getKey("i"):
            signal = "前进"
        elif key_module.getKey("k"):
            signal = "后退"
    except Exception:
        pass
    
    return signal


def run_keyboard_car_control(
    *,
    host: str = "192.168.149.1",
    port: int = 5000,
    speed: int = 50,
    keypress_dir: Optional[str] = None,
    interval: float = 0.05,
) -> None:
    """运行键盘控制小车"""
    if keypress_dir:
        path = Path(keypress_dir).expanduser()
    else:
        # 自动定位 MI-CarControl 目录
        path = Path(__file__).parent.parent / "MI-CarControl"
    
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    
    try:
        kp = __import__("KeyPressModule")
    except ImportError as e:
        print(f"✗ 错误：无法导入 KeyPressModule: {e}")
        sys.exit(1)

    try:
        import requests
    except ImportError:
        print("✗ 错误：缺少 requests 库")
        print("  请运行: pip install requests")
        sys.exit(1)

    print(f"🚗 小车控制程序已启动")
    print(f"   主机: {host}:{port}")
    print(f"   速度: {speed}")
    print(f"   按键控制:")
    print(f"     i: 前进  k: 后退")
    print(f"     j: 左转  l: 右转")
    print(f"     SPACE: 停止")
    print(f"     Ctrl+C: 退出")
    print()
    
    controller = CarHttpController(host=host, port=port, speed=speed)
    kp.init()
    
    try:
        while True:
            signal = key_to_signal(kp)
            
            # 每次都发送信号（和 test.py 一样）
            try:
                result = controller.send_signal(signal)
                print(f"✓ 发送成功: {signal:6s} (速度: {speed}) -> {result}")
            except requests.exceptions.ConnectionError:
                print(f"✗ 连接错误: 无法连接到 {host}:{port}")
                print(f"  请检查树莓派是否在线且服务正在运行")
            except requests.exceptions.Timeout:
                print(f"✗ 超时: 连接到 {host}:{port} 超时")
            except requests.exceptions.RequestException as e:
                print(f"✗ 请求错误: {e}")
            except Exception as exc:
                print(f"✗ 发送出错: {exc}")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n🛑 已停止")
        try:
            controller.send_signal("停止")
        except Exception:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Car control UI and keyboard CLI entrypoint.")
    parser.add_argument("--cli", action="store_true", help="启动旧版命令行键盘控制")
    parser.add_argument("--host", default="192.168.149.1", help="树莓派主机地址")
    parser.add_argument("--port", type=int, default=5000, help="树莓派端口")
    parser.add_argument("--speed", type=int, default=50, help="小车速度 (0-100)")
    parser.add_argument("--keypress-dir", default=None, help="KeyPressModule 所在目录")
    parser.add_argument("--interval", type=float, default=0.05, help="按键检查间隔（秒）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.cli:
            run_keyboard_car_control(
                host=args.host,
                port=args.port,
                speed=args.speed,
                keypress_dir=args.keypress_dir,
                interval=args.interval,
            )
        else:
            app = SupercontrolUI()
            app.run()
        return 0
    except Exception as e:
        print(f"\n✗ 程序出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

