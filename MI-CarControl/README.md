# MI-CarControl

- `keyboard_control.py`：通过键盘向小车 HTTP 接口发送前进、后退、左转、右转和停止指令。
- `brain_control.py`：通过脑环控制小车；前后模式使用注意力/冥想规则，转向模式使用共享模型，双眨眼切换模式。
- `neuropy.py`：小车侧使用的 NeuroSky / MindWave 设备驱动。
- `README.md`：本目录说明。

```bash
python MI-CarControl/keyboard_control.py --host 192.168.149.1 --port 5000
python MI-CarControl/brain_control.py --mindwave-port COM6
```
