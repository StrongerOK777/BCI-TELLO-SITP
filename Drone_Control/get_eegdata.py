import time
from neuropy import NeuroSkyPy

# 创建一个NeuroPy对象
neuropy = NeuroSkyPy("COM5", 57600)

# 打开串口连接
neuropy.start()

# 打开或创建一个文本文件来存储数据
output_file = open("sky_1.txt", "a")#a意味着在该文档后面追加，如果没有这个文件则新建
current_time0 = time.time()
try:
    while True:
        # 读取数据
        attention = neuropy.attention
        meditation = neuropy.meditation
        delta = neuropy.delta
        theta = neuropy.theta
        lowAlpha = neuropy.lowAlpha
        highAlpha = neuropy.highAlpha
        lowBeta = neuropy.lowBeta
        highBeta = neuropy.highBeta
        lowGamma = neuropy.lowGamma
        midGamma = neuropy.midGamma
        poorSignal = neuropy.poorSignal
        blinkStrength = neuropy.blinkStrength

        # 获取当前时间
        current_time = time.time() - current_time0

        # 将时间戳格式化为字符串，精确到小数点后两位
        current_time_str = "{:.2f}".format(current_time)

        # 将数据写入文本文件
        data_line = f"{current_time_str},{attention}|{meditation}|{delta}|{theta}|{lowAlpha}|{highAlpha}|{lowBeta}|{highBeta}|{lowGamma}|{midGamma}|{poorSignal}|{blinkStrength}\n"
        output_file.write(data_line)

        # 打印数据
        print(data_line)

        # 每秒读取20次，等待50毫秒
        time.sleep(1)

except KeyboardInterrupt:
    pass

# 关闭文件和串口连接
output_file.close()
neuropy.stop()