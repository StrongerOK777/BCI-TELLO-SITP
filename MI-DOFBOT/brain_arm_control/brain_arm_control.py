#!/usr/bin/env python3
"""Compatibility entrypoint for brain-controlled DOFBOT arm."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.brain_arm_control import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
