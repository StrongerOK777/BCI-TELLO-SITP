# MI-DroneControl

- `keyboard_control.py`：Tello 键盘控制与视频预览。
- `brain_control.py`：运动想象脑控飞行主程序。
- `predict.py`：连续模型预测。
- `train.py`：采集训练数据并生成共享模型。
- `diagnose.py`：脑环信号诊断。
- `neuropy.py`：NeuroSky / MindWave 设备驱动。

常用命令：

```bash
python MI-DroneControl/diagnose.py --mindwave-port COM6
python MI-DroneControl/train.py
python MI-DroneControl/predict.py --mindwave-port COM6
python MI-DroneControl/brain_control.py --mindwave-port COM6
python MI-DroneControl/keyboard_control.py
```
