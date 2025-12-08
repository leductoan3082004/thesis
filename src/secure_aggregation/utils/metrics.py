"""Minimal metrics and timing utilities for tests and lightweight instrumentation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


Labels = Tuple[Tuple[str, str], ...]


@dataclass
class MetricPoint:
    """Represents a single metric sample."""

    value: float
    labels: Labels


class InMemoryMetrics:
    """In-memory sink for counters, gauges, and timers."""

    def __init__(self) -> None:
        self.counters: Dict[str, List[MetricPoint]] = {}
        self.gauges: Dict[str, List[MetricPoint]] = {}
        self.timers: Dict[str, List[MetricPoint]] = {}

    def _emit(self, store: Dict[str, List[MetricPoint]], name: str, value: float, labels: Labels) -> None:
        store.setdefault(name, []).append(MetricPoint(value=value, labels=labels))

    def emit_counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        self._emit(self.counters, name, value, tuple(labels.items()))

    def emit_gauge(self, name: str, value: float, **labels: str) -> None:
        self._emit(self.gauges, name, value, tuple(labels.items()))

    def emit_timer(self, name: str, value: float, **labels: str) -> None:
        self._emit(self.timers, name, value, tuple(labels.items()))

    def snapshot(self) -> Dict[str, Dict[str, List[MetricPoint]]]:
        return {
            "counters": {k: list(v) for k, v in self.counters.items()},
            "gauges": {k: list(v) for k, v in self.gauges.items()},
            "timers": {k: list(v) for k, v in self.timers.items()},
        }


class Timer:
    """Context manager that records elapsed time to a metrics sink."""

    def __init__(self, sink: InMemoryMetrics, name: str, **labels: str) -> None:
        self.sink = sink
        self.name = name
        self.labels = labels
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._start is None:
            return
        elapsed = time.monotonic() - self._start
        self.sink.emit_timer(self.name, elapsed, **self.labels)


class CompositeMetrics:
    """Fan-out sink that emits to multiple underlying sinks."""

    def __init__(self, sinks: List[InMemoryMetrics]) -> None:
        self.sinks = sinks

    def emit_counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        for s in self.sinks:
            s.emit_counter(name, value, **labels)

    def emit_gauge(self, name: str, value: float, **labels: str) -> None:
        for s in self.sinks:
            s.emit_gauge(name, value, **labels)

    def emit_timer(self, name: str, value: float, **labels: str) -> None:
        for s in self.sinks:
            s.emit_timer(name, value, **labels)


class ConvergenceDetector:
    """Tracks when a metric reaches a target."""

    def __init__(self, target: float, higher_is_better: bool = True) -> None:
        self.target = target
        self.higher_is_better = higher_is_better
        self.best_round: int | None = None

    def observe(self, round_idx: int, value: float) -> int | None:
        if self.higher_is_better:
            if value >= self.target and self.best_round is None:
                self.best_round = round_idx
        else:
            if value <= self.target and self.best_round is None:
                self.best_round = round_idx
        return self.best_round
