"""Keyboard control for Tello with camera preview."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


def run_keyboard_drone_control(
    *,
    speed: int = 70,
    camera_width: int = 720,
    camera_height: int = 480,
    keypress_dir: Optional[str] = None,
) -> None:
    import cv2
    from djitellopy import tello

    if keypress_dir:
        path = Path(keypress_dir).expanduser()
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import KeyPressModule as kp

    font = cv2.FONT_HERSHEY_SIMPLEX
    drone = tello.Tello()
    drone.connect(wait_for_state=False)
    drone.streamon()
    drone.LOGGER.setLevel(logging.ERROR)
    time.sleep(5)
    kp.init()

    try:
        while True:
            original_image = drone.get_frame_read().frame
            image = cv2.resize(original_image, (camera_width, camera_height))
            handle_keyboard_input(drone, kp, cv2, image, speed, font)
            cv2.imshow("Drone Control Centre", image)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        drone.streamoff()


def handle_keyboard_input(drone, key_module, cv2, image, speed: int, font) -> None:
    lr, fb, ud, yv = 0, 0, 0, 0
    key_pressed = False
    if key_module.getKey("e"):
        cv2.imwrite("snap-{}.jpg".format(time.strftime("%H%M%S", time.localtime())), image)
    if key_module.getKey("UP"):
        drone.takeoff()
    elif key_module.getKey("DOWN"):
        drone.land()

    if key_module.getKey("j"):
        key_pressed = True
        lr = -speed
    elif key_module.getKey("l"):
        key_pressed = True
        lr = speed
    if key_module.getKey("i"):
        key_pressed = True
        fb = speed
    elif key_module.getKey("k"):
        key_pressed = True
        fb = -speed
    if key_module.getKey("w"):
        key_pressed = True
        ud = speed
    elif key_module.getKey("s"):
        key_pressed = True
        ud = -speed
    if key_module.getKey("a"):
        key_pressed = True
        yv = -speed
    elif key_module.getKey("d"):
        key_pressed = True
        yv = speed

    info_text = "battery : {0}% height: {1}cm   time: {2}".format(
        drone.get_battery(), drone.get_height(), time.strftime("%H:%M:%S", time.localtime())
    )
    cv2.putText(image, info_text, (10, 20), font, 0.5, (0, 0, 255), 1)
    if key_pressed:
        command_text = "Command : lr:{0}% fb:{1} ud:{2} yv:{3}".format(lr, fb, ud, yv)
        cv2.putText(image, command_text, (10, 40), font, 0.5, (0, 0, 255), 1)
    drone.send_rc_control(lr, fb, ud, yv)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard Tello control with video preview.")
    parser.add_argument("--speed", type=int, default=70)
    parser.add_argument("--camera-width", type=int, default=720)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--keypress-dir", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_keyboard_drone_control(
        speed=args.speed,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        keypress_dir=args.keypress_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
