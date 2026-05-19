"""Portable keyboard input primitives."""

from __future__ import annotations

import os
import select
import sys
from typing import Optional, Protocol


class KeyboardBackend(Protocol):
    def start(self) -> None:
        ...

    def read_key(self) -> Optional[str]:
        ...

    def close(self) -> None:
        ...


class _WindowsBackend:
    def start(self) -> None:
        pass

    def read_key(self) -> Optional[str]:
        import msvcrt

        if not msvcrt.kbhit():
            return None
        value = msvcrt.getwch()
        return value.lower() if value else None

    def close(self) -> None:
        pass


class _PosixBackend:
    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._old_settings = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("KeyboardReader requires an interactive terminal")
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def read_key(self) -> Optional[str]:
        if self._fd is None:
            raise RuntimeError("KeyboardReader.start() must be called first")
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return None
        value = sys.stdin.read(1)
        return value.lower() if value else None

    def close(self) -> None:
        if self._fd is None or self._old_settings is None:
            return
        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self._fd = None
        self._old_settings = None


class KeyboardReader:
    """Small context-managed wrapper around a replaceable keyboard backend."""

    def __init__(self, backend: Optional[KeyboardBackend] = None) -> None:
        self.backend = backend or (_WindowsBackend() if os.name == "nt" else _PosixBackend())
        self.started = False

    def start(self) -> None:
        self.backend.start()
        self.started = True

    def read_key(self) -> Optional[str]:
        if not self.started:
            raise RuntimeError("KeyboardReader.start() must be called first")
        return self.backend.read_key()

    def close(self) -> None:
        self.backend.close()
        self.started = False

    def __enter__(self) -> "KeyboardReader":
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
