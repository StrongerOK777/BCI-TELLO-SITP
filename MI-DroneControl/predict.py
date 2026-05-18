#!/usr/bin/env python3
"""Continuous model-prediction entrypoint for MindWave feature windows."""

from __future__ import annotations

import sys

from brain_control import main


if __name__ == "__main__":
    raise SystemExit(main(["--predict-only", *sys.argv[1:]]))
