"""Targeted tests for node service helpers."""

import pytest

pytest.importorskip("google.protobuf")
pytest.importorskip("torch")

from secure_aggregation.communication.node_service import NodeService
from secure_aggregation.convergence.central_checker import CentralChecker


def test_central_checker_records_signal_without_bridge() -> None:
    node = NodeService.__new__(NodeService)  # type: ignore[misc]
    node.node_id = "node_4"
    node.central_neighbor_addresses = {"node_4": "node_4:52052"}
    node.is_bridge_node = False
    node.bridge_client = None
    node.central_checker = CentralChecker(blockchain=None, total_cliques=2, cluster_ids=["cluster_0", "cluster_1"])
    node.central_metadata = None
    node._latest_cluster_converged = True
    node._latest_delta_norm = 0.0
    node._broadcast_central_signal(1, True, 0.0)

    assert node.central_checker._round_signals[1]["cluster_0"] is True  # type: ignore[attr-defined]
