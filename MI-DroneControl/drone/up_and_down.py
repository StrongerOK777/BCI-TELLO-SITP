#!/usr/bin/env python3
"""Legacy entrypoint; vertical control is now part of bin.mi_drone_control."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.mi_drone_control import main


if __name__ == "__main__":
    raise SystemExit(main())
