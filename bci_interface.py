"""Shared computer-to-headset interface configuration.

Edit this file when the NeuroSky / MindWave headset is connected through a
different serial port. All training and brain-control entrypoints read their
default headset connection from here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parent

# 修改这里即可切换脑环串口。
# Windows 常见格式: "COM5" / "COM6"
# macOS 常见格式: "/dev/cu.usbmodem2017_2_251"
MINDWAVE_PORT = "COM5" if sys.platform == "win32" else "/dev/cu.usbmodem2017_2_251"

# NeuroSky / MindWave 默认波特率通常为 57600。
MINDWAVE_BAUD = 57600

# 统一使用 TrainUser 中的 neuropy.py 作为脑环驱动。
NEUROPY_DIR = ROOT_DIR / "TrainUser"


@dataclass(frozen=True)
class MindWaveInterface:
    port: str
    baud: int
    neuropy_dir: str


def get_mindwave_interface(
    *,
    port: Optional[str] = None,
    baud: Optional[int] = None,
    neuropy_dir: Optional[str | Path] = None,
) -> MindWaveInterface:
    """Return the shared headset connection, allowing explicit CLI overrides."""

    return MindWaveInterface(
        port=port or MINDWAVE_PORT,
        baud=int(baud or MINDWAVE_BAUD),
        neuropy_dir=str(Path(neuropy_dir or NEUROPY_DIR).expanduser()),
    )
