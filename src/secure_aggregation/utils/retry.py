"""Retry and cleanup helpers."""

from __future__ import annotations

import time
from typing import Callable, Iterable, List, Tuple, Type, TypeVar


T = TypeVar("T")


class RetryError(Exception):
    """Raised when retry attempts are exhausted."""


def retry(
    func: Callable[[], T],
    retries: int,
    backoff: float,
    exceptions: Tuple[Type[BaseException], ...],
) -> T:
    """Retry a callable with exponential backoff."""
    attempt = 0
    delay = backoff
    while True:
        try:
            return func()
        except exceptions:
            attempt += 1
            if attempt > retries:
                raise RetryError(f"Failed after {retries} retries")
            time.sleep(delay)
            delay *= 2


class CleanupManager:
    """Registers cleanup callbacks to run in LIFO order."""

    def __init__(self) -> None:
        self._callbacks: List[Callable[[], None]] = []

    def register(self, cb: Callable[[], None]) -> None:
        self._callbacks.append(cb)

    def run(self) -> None:
        while self._callbacks:
            cb = self._callbacks.pop()
            try:
                cb()
            except Exception:
                # Best-effort cleanup; ignore individual failures.
                continue
