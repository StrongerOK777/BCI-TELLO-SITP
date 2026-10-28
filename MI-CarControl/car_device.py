"""Car adapter for the unified brain-control session (``bin.control``)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.control import DeviceBase  # noqa: E402
from bin.transport import JsonHttpClient  # noqa: E402

CAR_SIGNALS = {"forward": "前进", "backward": "后退", "left": "左转", "right": "右转"}
STOP_SIGNAL = "停止"


@dataclass
class CarSignalClient:
    """Sends a signal to the car's HTTP server (POST /signal)."""

    host: str = "192.168.149.1"
    port: int = 5000
    speed: int = 50
    transport: Optional[JsonHttpClient] = None

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = JsonHttpClient(f"http://{self.host}:{self.port}")

    def send_signal(self, signal: str) -> Any:
        assert self.transport is not None
        return self.transport.post_json("/signal", {"signal": signal, "speed": self.speed})


class RecordingSender:
    """Stand-in for CarSignalClient: records signals, sends nothing."""

    def __init__(self) -> None:
        self.signals: List[str] = []

    def send_signal(self, signal: str) -> str:
        self.signals.append(signal)
        return f"[模拟] {signal}"


class CarDevice(DeviceBase):
    """Maps intents to the car's HTTP signals; rest and emergency both send 停止."""

    name = "car"
    capabilities = frozenset(CAR_SIGNALS)

    def __init__(self, sender: Any) -> None:
        super().__init__()
        self.sender = sender

    def _send(self, signal: str) -> str:
        return f"{signal} -> {self.sender.send_signal(signal)}"

    def _perform(self, intent: str) -> str:
        return self._send(CAR_SIGNALS[intent])

    def _rest(self) -> str:
        return self._send(STOP_SIGNAL)

    def close(self) -> None:
        try:
            self._send(STOP_SIGNAL)
        except Exception as exc:
            print(f"停车指令发送失败：{exc}")
