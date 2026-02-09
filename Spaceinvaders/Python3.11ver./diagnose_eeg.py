from NeuroPy import NeuroPy
import time
import sys

# 使用用户代码中的端口
PORT = '/dev/cu.usbmodem2017_2_251'
BAUD = 57600

def diagnose():
    print(f"尝试连接到 {PORT}...")
    neuropy = NeuroPy(PORT, BAUD)
    
    try:
        neuropy.start()
    except Exception as e:
        print(f"连接失败: {e}")
        return

    print("已启动。正在读取数据（请确保已佩戴设备）...")
    print("时间 | 信号质量 | 注意力 | 冥想度 | 眨眼强度 | Raw电压")
    print("-" * 70)
    
    start_time = time.time()
    try:
        while time.time() - start_time < 60: # 运行60秒
            sig = neuropy.poorSignal
            att = neuropy.attention
            med = neuropy.meditation
            blink = neuropy.blinkStrength
            raw = neuropy.rawValue
            
            status = "PERFECT" if sig == 0 else "NOISY"
            if sig == 200: status = "OFF-HEAD"
            
            # 这里的输出可以帮你判断：如果你眨眼，blink 有没有值？
            sys.stdout.write(f"\r{int(time.time()-start_time)}s | {sig:3} ({status:7}) | {att:3} | {med:3} | {blink:3} | {raw:5}        ")
            sys.stdout.flush()
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        neuropy.stop()
        print("\n诊断结束。")

if __name__ == "__main__":
    diagnose()
