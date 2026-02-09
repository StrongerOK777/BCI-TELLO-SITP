import logging
import time
import cv2
from djitellopy import tello
import KeyPressModule as kp  # 用于获取键盘按键
from time import sleep

def getKeyboardInput(drone, speed, image):
    lr, fb, ud, yv = 0, 0, 0, 0
    key_pressed = 0
    if kp.getKey("e"):
        cv2.imwrite('snap-{}.jpg'.format(time.strftime("%H%M%S", time.localtime())), image)
    if kp.getKey("UP"):
        drone.takeoff()
    elif kp.getKey("DOWN"):
        drone.land()

    if kp.getKey("j"):
        key_pressed = 1
        lr = -speed
    elif kp.getKey("l"):
        key_pressed = 1
        lr = speed

    if kp.getKey("i"):
        key_pressed = 1
        fb = speed
    elif kp.getKey("k"):
        key_pressed = 1
        fb = -speed

    if kp.getKey("w"):
        key_pressed = 1
        ud = speed
    elif kp.getKey("s"):
        key_pressed = 1
        ud = -speed

    if kp.getKey("a"):
        key_pressed = 1
        yv = -speed
    elif kp.getKey("d"):
        key_pressed = 1
        yv = speed
    InfoText = "battery : {0}% height: {1}cm   time: {2}".format(drone.get_battery(), drone.get_height(), time.strftime("%H:%M:%S",time.localtime()))
    cv2.putText(image, InfoText, (10, 20), font, fontScale, (0, 0, 255), lineThickness)
    if key_pressed == 1:
        InfoText = "Command : lr:{0}% fb:{1} ud:{2} yv:{3}".format(lr, fb, ud, yv)
        cv2.putText(image, InfoText, (10, 40), font, fontScale, (0, 0, 255), lineThickness)

    drone.send_rc_control(lr, fb, ud, yv)

# 主程序
# 摄像头设置 
Camera_Width = 720
Camera_Height = 480
DetectRange = [6000, 11000]  # DetectRange[0] 是保持静止的检测人脸面积阈值下限，DetectRange[0] 是保持静止的检测人脸面积阈值上限
PID_Parameter = [0.5, 0.0004, 0.4]
pErrorRotate, pErrorUp = 0, 0

# 字体设置
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 0, 0)
lineThickness = 1

# Tello初始化设置
Drone = tello.Tello()  # 创建飞行器对象
Drone.connect(wait_for_state=False)  # 连接到飞行器，不等待状态包
Drone.streamon()  # 开启视频传输
Drone.LOGGER.setLevel(logging.ERROR)  # 只显示错误信息
sleep(5)  #  等待视频初始化
kp.init()  # 初始化按键处理模块


while True:
    OriginalImage = Drone.get_frame_read().frame
    Image = cv2.resize(OriginalImage, (Camera_Width, Camera_Height))
    getKeyboardInput(drone=Drone, speed=70, image=Image)  # 按键控制
    cv2.imshow("Drone Control Centre", Image)
    cv2.waitKey(1)