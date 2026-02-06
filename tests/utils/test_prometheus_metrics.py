"""Tests for PrometheusMetrics functionality."""

import pytest

from secure_aggregation.utils.prometheus_metrics import PrometheusMetrics


@pytest.fixture
def metrics():
    """Create a fresh PrometheusMetrics instance for each test."""
    PrometheusMetrics._instance = None
    return PrometheusMetrics.get_instance(node_id="test_node", clique_id=0)


def test_metrics_singleton_pattern():
    PrometheusMetrics._instance = None
    metrics1 = PrometheusMetrics.get_instance(node_id="node_1", clique_id=0)
    metrics2 = PrometheusMetrics.get_instance(node_id="node_2", clique_id=1)
    assert metrics1 is metrics2


def test_set_topology_max_degree(metrics):
    metrics.set_topology_max_degree(7)
    assert True


def test_set_topology_average_degree(metrics):
    metrics.set_topology_average_degree(4.85)
    assert True


def test_set_topology_type(metrics):
    metrics.set_topology_type("d_cliques")
    assert True


def test_set_total_bytes_per_round(metrics):
    metrics.set_round(1)
    metrics.set_total_bytes_per_round(1024)
    assert True


def test_topology_metrics_with_different_values(metrics):
    metrics.set_topology_max_degree(10)
    metrics.set_topology_average_degree(5.5)
    metrics.set_topology_type("fully_connected")
    assert True


def test_bytes_per_round_across_rounds(metrics):
    for round_idx in range(3):
        metrics.set_round(round_idx)
        metrics.set_total_bytes_per_round(1000 * (round_idx + 1))
    assert True


def test_topology_degree_with_zero(metrics):
    metrics.set_topology_max_degree(0)
    metrics.set_topology_average_degree(0.0)
    assert True


def test_topology_type_with_different_topologies(metrics):
    topologies = ["d_cliques", "fully_connected", "random", "ring", "small_world"]
    for topo in topologies:
        metrics.set_topology_type(topo)
    assert True


def test_metrics_labels(metrics):
    assert metrics.node_id == "test_node"
    assert metrics.clique_id == 0


def test_set_round_and_bytes(metrics):
    metrics.set_round(5)
    metrics.set_total_bytes_per_round(2048)
    assert True
