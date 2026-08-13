"""Simple tee logger used by command-line runs."""

from __future__ import annotations

from pathlib import Path
from threading import Lock


class RunLogger:
    """Write the same concise progress message to console and a UTF-8 log file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8", newline="\n")
        self._lock = Lock()

    def log(self, message: str) -> None:
        with self._lock:
            print(message, flush=True)
            self._file.write(message + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
