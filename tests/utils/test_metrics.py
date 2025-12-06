from secure_aggregation.utils import InMemoryMetrics, Timer


def test_in_memory_metrics_and_timer() -> None:
    sink = InMemoryMetrics()
    sink.emit_counter("messages", value=2, node="a")
    sink.emit_gauge("queue_depth", value=5, node="a")
    with Timer(sink, "round_duration", role="trainer"):
        pass
    snapshot = sink.snapshot()
    assert snapshot["counters"]["messages"][0].value == 2
    assert snapshot["gauges"]["queue_depth"][0].labels == (("node", "a"),)
    assert snapshot["timers"]["round_duration"][0].labels == (("role", "trainer"),)
