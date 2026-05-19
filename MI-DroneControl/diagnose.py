#!/usr/bin/env python3
"""MindWave signal diagnostic entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.eeg import run_blink_test
from bci_interface import get_mindwave_interface


DEFAULT_INTERFACE = get_mindwave_interface(neuropy_dir=DEVICE_DIR)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print live MindWave diagnostics.")
    parser.add_argument("--mindwave-port", default=DEFAULT_INTERFACE.port)
    parser.add_argument("--mindwave-baud", type=int, default=DEFAULT_INTERFACE.baud)
    parser.add_argument("--neuropy-dir", default=DEFAULT_INTERFACE.neuropy_dir)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_blink_test(
        port=args.mindwave_port,
        baud=args.mindwave_baud,
        neuropy_dir=args.neuropy_dir,
        sample_interval=args.sample_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
