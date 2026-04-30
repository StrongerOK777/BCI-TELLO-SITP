#!/usr/bin/env python3
"""Compatibility entrypoint for continuous model prediction."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.mi_drone_control import main


if __name__ == "__main__":
    argv = ["--predict-only", *sys.argv[1:]]
    raise SystemExit(main(argv))
