import math
import time

import pytest

from secure_aggregation.config.models import NodeRole
from secure_aggregation.node import GossipCache, ModelSnapshot, NodeEngine, NodeRuntimeConfig, ReliabilityScore


def _engine(node_id: str, uptime: float, bandwidth: float, latency: float, **kwargs) -> NodeEngine:
    return NodeEngine(
        NodeRuntimeConfig(
            node_id=node_id,
            role=kwargs.get("role", NodeRole.HYBRID),
            reliability=ReliabilityScore(uptime=uptime, bandwidth=bandwidth, latency=latency),
            freshness_window=kwargs.get("freshness_window", 10.0),
            decay_tau=kwargs.get("decay_tau", 5.0),
            cache_size=kwargs.get("cache_size", 3),
        )
    )


def test_deterministic_aggregator_rotation() -> None:
    engines = [
        _engine("a", uptime=0.9, bandwidth=0.9, latency=0.1),
        _engine("b", uptime=0.8, bandwidth=0.7, latency=0.2),
        _engine("c", uptime=0.7, bandwidth=0.6, latency=0.3),
    ]
    order = [NodeEngine.select_aggregator(engines, window) for window in range(6)]
    # Should cycle deterministically through sorted-by-score order
    sorted_by_score = [e.config.node_id for e in sorted(engines, key=lambda e: (-e.reliability_score, e.config.node_id))]
    assert order[:3] == sorted_by_score
    assert order[3:] == sorted_by_score


def test_gossip_cache_capacity_and_freshness() -> None:
    cache = GossipCache(capacity=2)
    base = time.time()
    cache.add_snapshot(ModelSnapshot("u1", [1.0], base - 30))
    cache.add_snapshot(ModelSnapshot("u2", [2.0], base - 20))
    cache.add_snapshot(ModelSnapshot("u3", [3.0], base - 10))  # pushes out u1
    snap_ids = [s.source_id for s in cache.fresh_snapshots(base, 100)]
    assert snap_ids == ["u3", "u2"]
    recent = cache.fresh_snapshots(base, freshness_window=15)
    assert [s.source_id for s in recent] == ["u3"]


def test_merge_with_remote_weights_and_fallback() -> None:
    engine = _engine("a", 1, 1, 0, freshness_window=10, decay_tau=2, cache_size=3)
    now = 100.0
    engine.add_remote_snapshot([2.0, 2.0], timestamp=now - 1, source_id="n1")
    engine.add_remote_snapshot([4.0, 4.0], timestamp=now - 3, source_id="n2")
    merged = engine.merge_with_remote([1.0, 1.0], now_ts=now)
    w1 = math.exp(-1 / 2)
    w2 = math.exp(-3 / 2)
    expected0 = (1 + w1 * 2 + w2 * 4) / (1 + w1 + w2)
    assert pytest.approx(merged[0], rel=1e-6) == expected0
    # Fallback when nothing fresh
    stale = engine.merge_with_remote([5.0, 5.0], now_ts=now + 100)
    assert stale == [5.0, 5.0]


def test_orchestrate_window_handles_dropout_and_returns_aggregator() -> None:
    engines = [
        _engine("u1", 1, 1, 0),
        _engine("u2", 0.8, 0.7, 0.1),
        _engine("u3", 0.9, 0.6, 0.2),
    ]
    model_vectors = {"u1": [1, 1], "u2": [2, 2], "u3": [100, 100]}
    aggregator_id, result = NodeEngine.orchestrate_window(engines, model_vectors, threshold=2, window_index=0, dropouts=["u3"])
    assert aggregator_id == NodeEngine.select_aggregator(engines, 0)
    assert result.survivors == ["u1", "u2"]
    assert result.aggregate_sum == [3, 3]
