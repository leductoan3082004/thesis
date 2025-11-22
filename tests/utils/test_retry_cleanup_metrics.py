import time

import pytest

from secure_aggregation.utils import (
    CleanupManager,
    CompositeMetrics,
    ConvergenceDetector,
    InMemoryMetrics,
    RetryError,
    retry,
)


def test_retry_exponential_backoff_and_success() -> None:
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("fail")
        return "ok"

    result = retry(flaky, retries=5, backoff=0.001, exceptions=(ValueError,))
    assert result == "ok"
    assert attempts["count"] == 3


def test_retry_raises_after_exhaustion() -> None:
    def always_fail():
        raise ValueError("nope")

    with pytest.raises(RetryError):
        retry(always_fail, retries=2, backoff=0.001, exceptions=(ValueError,))


def test_cleanup_manager_runs_callbacks() -> None:
    order = []
    mgr = CleanupManager()
    mgr.register(lambda: order.append("first"))
    mgr.register(lambda: order.append("second"))
    mgr.run()
    assert order == ["second", "first"]


def test_composite_metrics_fanout_and_convergence_detector() -> None:
    sink1 = InMemoryMetrics()
    sink2 = InMemoryMetrics()
    composite = CompositeMetrics([sink1, sink2])
    composite.emit_counter("messages", value=2, node="a")
    assert sink1.counters["messages"][0].labels == (("node", "a"),)
    assert sink2.counters["messages"][0].value == 2

    det = ConvergenceDetector(target=0.9, higher_is_better=True)
    assert det.observe(0, 0.5) is None
    assert det.observe(1, 0.95) == 1
    assert det.observe(2, 1.0) == 1  # stays at first round
