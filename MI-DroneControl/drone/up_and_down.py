from neuropy import NeuroSkyPy
from djitellopy import tello
from time import sleep


def main():
	neuropy = NeuroSkyPy("/dev/cu.usbmodem2017_2_251", 57600)
	neuropy.start()

	me = tello.Tello()
	try:
		me.connect()
	except Exception as exc:
		print("无人机未响应，请先连接 Tello Wi-Fi 后重试：", exc)
		neuropy.stop()
		return

	print(me.get_battery())

	height = 0
	attentionnum = 0
	meditationnum = 0
	try:
		while True:
			attention = neuropy.attention
			meditation = neuropy.meditation
			print(f"attention={attention} meditation={meditation}")

			if attention >= 50:
				attentionnum += 1
			else:
				attentionnum = 0

			if meditation >= 50:
				meditationnum += 1
			else:
				meditationnum = 0

			if attentionnum >= 3:
				if height == 0:
					print("ATT>=50 连续3次：无人机起飞！")
					me.takeoff()
					height = 1
				elif height > 0 and height < 4:
					print("ATT>=50 连续3次：无人机上升！")
					me.move_up(30)
					height += 0.3
				attentionnum = 0

			if meditationnum >= 3:
				if height == 1:
					print("MED>=50 连续3次：无人机降落！")
					me.land()
					height = 0
				elif height > 1:
					print("MED>=50 连续3次：无人机下降！")
					me.move_down(30)
					height -= 0.3
				meditationnum = 0

			sleep(1)
	except KeyboardInterrupt:
		pass
	finally:
		neuropy.stop()


if __name__ == "__main__":
	main()
