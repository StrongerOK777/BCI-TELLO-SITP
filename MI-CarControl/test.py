import joblib
import time
import numpy as np
import requests
import time
import KeyPressModule as kp 
# 3. 主要发送函数
def send_to_raspberry(speed = 50):
    """
    发送信号到树莓派
    signal: 你的模型输出的信号，比如 "前进"
    speed: 速度，默认50
    """

    # 树莓派的IP地址（改成你的树莓派IP）
    PI_IP = "192.168.149.1"

    PI_PORT = 5000

    # 发送的地址
    url = f"http://{PI_IP}:{PI_PORT}/signal"

    key_pressed = 0

    if kp.getKey("SPACE"):
        key_pressed = "停止"
    if kp.getKey("j"):
        key_pressed = "左转"
    elif kp.getKey("l"):
        key_pressed = "右转"
    if kp.getKey("i"):
        key_pressed = "前进"
    elif kp.getKey("k"):
        key_pressed = "后退"

    data = {
        "signal": key_pressed,
        "speed": speed
    }

    try:
        # 发送POST请求
        response = requests.post(url, json=data)

        # 打印结果
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 发送成功: {key_pressed} {speed}")
            print(f"  树莓派返回: {result}")
        else:
            print(f"✗ 发送失败，错误码: {response.status_code}")

    except Exception as e:
        print(f"✗ 发送出错: {e}")
        print("  请检查：")
        print("  1. 树莓派IP是否正确")
        print("  2. 树莓派服务器是否在运行")
        print("  3. 电脑和树莓派是否在同一网络")


# 4. 主程序
def main():
    print("🚀 开始发送脑电信号...")
    print("💡 按 Ctrl+C 停止")
    print("-" * 40)
    
    # 初始化 pygame（键盘检测必需）
    kp.init()

    try:
        while True:
            
            send_to_raspberry()

    except KeyboardInterrupt:
        print("\n👋 停止发送")
    except Exception as e:
        print(f"❌ 程序出错: {e}")


# 5. 运行程序
if __name__ == "__main__":
    # 先安装requests库（如果没有的话）
    try:
        import requests
    except ImportError:
        print("正在安装requests库...")
        import subprocess

        subprocess.check_call(["pip", "install", "requests"])
        print("安装完成，请重新运行程序")
        exit()

    main()