#!/usr/bin/env python3
"""Compatibility entrypoint for the interactive training workflow."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


if __name__ == "__main__":
    runpy.run_module("bin.training_legacy", run_name="__main__")
