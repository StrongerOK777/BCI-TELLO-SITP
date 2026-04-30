#!/usr/bin/env python3
"""Compatibility entrypoint for keyboard car control."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.car_control import main


if __name__ == "__main__":
    argv = ["--keypress-dir", str(Path(__file__).resolve().parent), *sys.argv[1:]]
    raise SystemExit(main(argv))
