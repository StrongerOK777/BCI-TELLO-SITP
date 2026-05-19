#!/usr/bin/env python3
"""Test script to verify brain signal reception from MindWave."""

import sys
import time
import os
from pathlib import Path

# Add MI-CarControl to path for neuropy
sys.path.insert(0, str(Path(__file__).resolve().parent / "MI-CarControl"))

from neuropy import NeuroSkyPy

def test_brain_signal(port: str = "COM5", baud: int = 57600, duration: int = 10):
    """Test brain signal reception."""
    print(f"测试脑环连接...")
    print(f"端口: {port}, 波特率: {baud}")
    print(f"测试时长: {duration} 秒\n")
    
    try:
        # 初始化脑环
        mindwave = NeuroSkyPy(port, baud)
        mindwave.start()
        print("✓ 脑环已连接\n")
        
        # 收集数据
        start_time = time.time()
        count = 0
        good_samples = 0
        
        print("采集中... (Ctrl+C 停止)")
        print("-" * 80)
        print(f"{'时间':<8} {'Attention':<12} {'Meditation':<12} {'PoorSignal':<12} {'BlinkStrength':<12}")
        print("-" * 80)
        
        while time.time() - start_time < duration:
            attention = mindwave.attention or 0
            meditation = mindwave.meditation or 0
            poor_signal = mindwave.poorSignal or 0
            blink_strength = mindwave.blinkStrength or 0
            
            elapsed = time.time() - start_time
            print(f"{elapsed:6.1f}s  {attention:<10}  {meditation:<10}  {poor_signal:<10}  {blink_strength:<10}")
            
            count += 1
            if attention > 0 and meditation > 0:
                good_samples += 1
            
            time.sleep(0.5)
        
        print("-" * 80)
        
        # 统计结果
        print(f"\n✓ 测试完成!")
        print(f"总采样数: {count}")
        print(f"有效样本: {good_samples} ({100*good_samples/count:.1f}%)")
        
        if good_samples > 0:
            print("✓ 脑环工作正常！")
        else:
            print("✗ 脑环未读取到有效数据")
        
        mindwave.stop()
        return True
        
    except Exception as exc:
        print(f"✗ 错误: {exc}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test brain signal reception")
    parser.add_argument("--port", default="COM5", help="MindWave port (default: COM5)")
    parser.add_argument("--baud", type=int, default=57600, help="Baud rate (default: 57600)")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds (default: 10)")
    
    args = parser.parse_args()
    
    success = test_brain_signal(
        port=args.port,
        baud=args.baud,
        duration=args.duration
    )
    
    sys.exit(0 if success else 1)
