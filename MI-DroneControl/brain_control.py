"""Motor-imagery BCI control state machine for Tello drones."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DEVICE_DIR) not in sys.path:
    sys.path.insert(0, str(DEVICE_DIR))

from bin.eeg import BrainSignalReader, FeatureWindowResult, WindowResult
from bin.models import LABELS, ModelPredictor
from drone_hardware import DroneController, SimulatedDroneController, TelloDroneController


DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "FinalModel.pth"


MODE_VERTICAL = 0
MODE_TURNING = 1
MODE_FORWARD_BACKWARD = 2
MODE_NAMES = {
    MODE_VERTICAL: "升降",
    MODE_TURNING: "转弯",
    MODE_FORWARD_BACKWARD: "前后",
}


@dataclass
class MIDroneConfig:
    mindwave_port: Optional[str] = None
    mindwave_baud: Optional[int] = None
    model_path: Optional[str] = str(DEFAULT_MODEL_PATH)
    neuropy_dir: Optional[str] = str(DEVICE_DIR)
    simulated: bool = False
    window_size: int = 30
    sample_interval: float = 0.1
    attention_threshold: int = 30
    meditation_threshold: int = 50
    blink_threshold: int = 100
    poor_signal_threshold: int = 20
    mode_switch_blinks: int = 2
    min_decision_count: int = 20
    takeoff_height: int = 100
    max_height: int = 150
    height_step: int = 10
    action_pause_sec: float = 1.0


class MIDroneController:
    def __init__(
        self,
        config: MIDroneConfig,
        reader: BrainSignalReader,
        drone: DroneController,
        predictor: ModelPredictor,
    ) -> None:
        self.config = config
        self.reader = reader
        self.drone = drone
        self.predictor = predictor
        self.mode = MODE_VERTICAL
        self.height = 0
        self.running = False

    def run(self, max_windows: Optional[int] = None) -> None:
        self.running = True
        windows_seen = 0
        try:
            self.drone.connect()
            self.reader.start()
            while self.running:
                self.step()
                windows_seen += 1
                if max_windows is not None and windows_seen >= max_windows:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def step(self) -> None:
        mode_name = MODE_NAMES[self.mode]
        print(f"{mode_name}：")
        if self.mode == MODE_TURNING:
            result = self.reader.collect_feature_window(mode_name)
        else:
            result = self.reader.collect_rule_window(mode_name)

        if not result.valid:
            self.handle_invalid_window(result)
            return

        if self.mode == MODE_VERTICAL:
            self.handle_vertical_mode(result)
        elif self.mode == MODE_TURNING:
            self.handle_turning_mode(result)
        else:
            self.handle_forward_backward_mode(result)

    def handle_invalid_window(self, result: WindowResult) -> None:
        print(f"PoorSignal too high: {result.poor_signal}, {result.reason}")
        if result.disconnected:
            self.height = 0
            self.mode = MODE_VERTICAL
            print("PoorSignal=200，安全降落")
            self.drone.land()
        time.sleep(self.config.action_pause_sec)

    def handle_vertical_mode(self, result: WindowResult) -> None:
        if result.blink_count >= self.config.mode_switch_blinks:
            if self.height > 1:
                print("检测到眨眼，进入转弯模式")
                self.mode = MODE_TURNING
            else:
                print("检测到眨眼，但无人机未起飞，继续升降模式")
            return

        if (
            result.attention_count < self.config.min_decision_count
            and result.meditation_count < self.config.min_decision_count
        ):
            print("注意力和冥想过低，原地不动")
            time.sleep(self.config.action_pause_sec)
            return

        if result.attention_count >= result.meditation_count:
            if self.height == 0:
                self.height = self.config.takeoff_height
                print("Attention更高，无人机起飞！")
                self.drone.takeoff()
            elif self.height < self.config.max_height:
                self.height += self.config.height_step
                print(f"Attention更高，无人机上升{self.config.height_step}cm！")
                self.drone.up()
            else:
                print("无人机已到达最大高度！")
                time.sleep(self.config.action_pause_sec)
            return

        if self.height == self.config.takeoff_height:
            self.height = 0
            print("Meditation更高，无人机降落！")
            self.drone.land()
        elif self.height > self.config.takeoff_height:
            self.height -= self.config.height_step
            print(f"Meditation更高，无人机下降{self.config.height_step}cm！")
            self.drone.down()
        else:
            print("无人机已到达最低高度！")
            time.sleep(self.config.action_pause_sec)

    def handle_turning_mode(self, result: WindowResult) -> None:
        if result.blink_count >= self.config.mode_switch_blinks:
            print("两次眨眼，进入前后模式")
            self.mode = MODE_FORWARD_BACKWARD
            return

        predicted = 2
        if isinstance(result, FeatureWindowResult):
            predicted = self.predictor.predict_window(result.feature_window)
        print(f"预测结果：{LABELS[predicted]} | {self.config.window_size}组完成")

        if predicted == 2:
            print("休息状态，rest")
            time.sleep(self.config.action_pause_sec)
        elif predicted == 1:
            print("无人机右转")
            self.drone.right()
        else:
            print("无人机左转")
            self.drone.left()

    def handle_forward_backward_mode(self, result: WindowResult) -> None:
        if result.blink_count >= self.config.mode_switch_blinks:
            print("检测到眨眼两次，进入升降模式")
            self.mode = MODE_VERTICAL
            return
        if result.blink_count == 1:
            print("检测到眨眼，重新测量")
            return
        if (
            result.attention_count < self.config.min_decision_count
            or result.meditation_count < self.config.min_decision_count
        ):
            print("注意力或冥想过低，原地不动")
            time.sleep(self.config.action_pause_sec)
            return
        if result.attention_count >= result.meditation_count:
            print("Attention更高，无人机前进10cm！")
            self.drone.forward()
        else:
            print("Meditation更高，无人机后退10cm！")
            self.drone.backward()

    def shutdown(self) -> None:
        self.running = False
        try:
            if self.height > 0:
                self.drone.land()
        except Exception as exc:
            print(f"无人机降落失败：{exc}")
        try:
            self.reader.stop()
        finally:
            self.drone.close()
        print("\n" + "=" * 60)
        print("所有进程已退出")
        print("=" * 60)


def build_controller(
    config: Optional[MIDroneConfig] = None,
    *,
    reader: Optional[BrainSignalReader] = None,
    drone: Optional[DroneController] = None,
    predictor: Optional[ModelPredictor] = None,
) -> MIDroneController:
    config = config or MIDroneConfig(
        model_path=str(DEFAULT_MODEL_PATH),
        neuropy_dir=str(DEVICE_DIR),
    )
    reader = reader or BrainSignalReader(
        port=config.mindwave_port,
        baud=config.mindwave_baud,
        neuropy_dir=config.neuropy_dir,
        window_size=config.window_size,
        sample_interval=config.sample_interval,
        attention_threshold=config.attention_threshold,
        meditation_threshold=config.meditation_threshold,
        blink_threshold=config.blink_threshold,
        poor_signal_threshold=config.poor_signal_threshold,
    )
    drone = drone or (SimulatedDroneController() if config.simulated else TelloDroneController())
    predictor = predictor or ModelPredictor(config.model_path, autoload=False)
    return MIDroneController(config, reader, drone, predictor)


def run_mi_drone_control(config: Optional[MIDroneConfig] = None) -> MIDroneController:
    controller = build_controller(config)
    controller.run()
    return controller


def run_prediction_loop(config: Optional[MIDroneConfig] = None) -> None:
    config = config or MIDroneConfig(
        model_path=str(DEFAULT_MODEL_PATH),
        neuropy_dir=str(DEVICE_DIR),
    )
    reader = BrainSignalReader(
        port=config.mindwave_port,
        baud=config.mindwave_baud,
        neuropy_dir=config.neuropy_dir,
        window_size=config.window_size,
        sample_interval=config.sample_interval,
        attention_threshold=config.attention_threshold,
        meditation_threshold=config.meditation_threshold,
        blink_threshold=config.blink_threshold,
        poor_signal_threshold=config.poor_signal_threshold,
    )
    predictor = ModelPredictor(config.model_path)
    try:
        reader.start()
        while True:
            result = reader.collect_feature_window("预测")
            if not result.valid:
                print(f"PoorSignal too high: {result.poor_signal}, reset window...")
                time.sleep(config.action_pause_sec)
                continue
            predicted = predictor.predict_window(result.feature_window)
            print(f"预测结果：{LABELS[predicted]} | {config.window_size}组完成")
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control Tello with MindWave motor-imagery signals.")
    parser.add_argument("--mindwave-port", default=None)
    parser.add_argument("--mindwave-baud", type=int, default=None)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--neuropy-dir", default=str(DEVICE_DIR))
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--attention-threshold", type=int, default=30)
    parser.add_argument("--meditation-threshold", type=int, default=50)
    parser.add_argument("--blink-threshold", type=int, default=100)
    parser.add_argument("--poor-signal-threshold", type=int, default=20)
    return parser


def config_from_args(args: argparse.Namespace) -> MIDroneConfig:
    return MIDroneConfig(
        mindwave_port=args.mindwave_port,
        mindwave_baud=args.mindwave_baud,
        model_path=args.model_path,
        neuropy_dir=args.neuropy_dir,
        simulated=args.simulated,
        window_size=args.window_size,
        sample_interval=args.sample_interval,
        attention_threshold=args.attention_threshold,
        meditation_threshold=args.meditation_threshold,
        blink_threshold=args.blink_threshold,
        poor_signal_threshold=args.poor_signal_threshold,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    try:
        if args.predict_only:
            run_prediction_loop(config)
        else:
            run_mi_drone_control(config)
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
