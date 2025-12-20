"""Integration tests for D-Cliques topology with secure aggregation services."""

import pytest
from typing import Dict, List

from secure_aggregation.communication.ttp_service import TTPServicer, TopologyConfig, TopologyState
from secure_aggregation.topology import (
    build_full_topology,
    compute_clique_threshold,
    compute_node_labels_from_partition,
    elect_clique_aggregator,
    find_node_clique,
)
from secure_aggregation.data import dirichlet_partition


def create_test_labels(num_samples: int = 1000, num_classes: int = 10) -> Dict[int, int]:
    """Create synthetic MNIST-like labels for testing."""
    return {i: i % num_classes for i in range(num_samples)}


class TestTTPTopologyIntegration:
    """Test TTP service topology computation."""

    def test_ttp_builds_topology_at_startup(self):
        """Verify TTP builds topology when given config and labels."""
        labels = create_test_labels(1000, 10)
        config = TopologyConfig(
            num_clients=10,
            clique_size=5,
            alpha=0.5,
            seed=42,
        )

        servicer = TTPServicer(topology_config=config, labels=labels)

        assert len(servicer.topology.cliques) == 2  # 10 clients / 5 per clique
        assert len(servicer.topology.node_to_clique) == 10
        assert len(servicer.topology.partition) == 10
        assert len(servicer.topology.thresholds) == 2

    def test_ttp_computes_correct_thresholds(self):
        """Verify thresholds are 2/3 majority per clique."""
        labels = create_test_labels(1000, 10)
        config = TopologyConfig(
            num_clients=15,
            clique_size=5,
            alpha=0.5,
            seed=42,
        )

        servicer = TTPServicer(topology_config=config, labels=labels)

        # Each clique of 5 should have threshold = ceil(0.6667 * 5) = 4
        for clique_id, threshold in servicer.topology.thresholds.items():
            clique_size = len(servicer.topology.cliques[clique_id])
            expected = compute_clique_threshold(clique_size)
            assert threshold == expected

    def test_ttp_assigns_data_to_all_clients(self):
        """Verify all clients get data indices."""
        labels = create_test_labels(1000, 10)
        config = TopologyConfig(
            num_clients=10,
            clique_size=5,
            alpha=0.5,
            seed=42,
        )

        servicer = TTPServicer(topology_config=config, labels=labels)

        total_indices = 0
        for client_id, indices in servicer.topology.partition.items():
            assert len(indices) > 0, f"Client {client_id} has no data"
            total_indices += len(indices)

        assert total_indices == 1000  # All data assigned

    def test_ttp_without_topology_config(self):
        """Verify TTP works without topology configuration."""
        servicer = TTPServicer()

        assert servicer.topology.cliques == []
        assert servicer.topology.node_to_clique == {}
        assert servicer.topology.partition == {}


class TestCliqueAssignment:
    """Test clique membership and assignment."""

    def test_all_nodes_in_exactly_one_clique(self):
        """Verify each node belongs to exactly one clique."""
        labels = create_test_labels(1000, 10)
        config = TopologyConfig(
            num_clients=20,
            clique_size=5,
            alpha=0.5,
            seed=42,
        )

        servicer = TTPServicer(topology_config=config, labels=labels)

        # Check node_to_clique mapping
        seen_nodes = set()
        for clique in servicer.topology.cliques:
            for node in clique:
                assert node not in seen_nodes, f"Node {node} in multiple cliques"
                seen_nodes.add(node)

        assert len(seen_nodes) == 20

    def test_clique_members_consistent(self):
        """Verify clique members match node_to_clique mapping."""
        labels = create_test_labels(1000, 10)
        config = TopologyConfig(
            num_clients=15,
            clique_size=5,
            alpha=0.5,
            seed=42,
        )

        servicer = TTPServicer(topology_config=config, labels=labels)

        for node_id, clique_id in servicer.topology.node_to_clique.items():
            assert node_id in servicer.topology.cliques[clique_id]


class TestAggregatorElection:
    """Test per-clique aggregator election."""

    def test_elect_clique_aggregator_round_robin(self):
        """Verify aggregator election is round-robin within clique."""
        clique_members = ["node_0", "node_1", "node_2", "node_3", "node_4"]

        # Over multiple rounds, each member should be elected once
        elected = []
        for round_idx in range(5):
            agg = elect_clique_aggregator(clique_members, round_idx)
            elected.append(agg)

        # Should cycle through all members in sorted order
        assert elected == sorted(clique_members)

    def test_elect_aggregator_wraps_around(self):
        """Verify election wraps around after full cycle."""
        clique_members = ["node_0", "node_1", "node_2"]

        agg_round_0 = elect_clique_aggregator(clique_members, 0)
        agg_round_3 = elect_clique_aggregator(clique_members, 3)
        agg_round_6 = elect_clique_aggregator(clique_members, 6)

        assert agg_round_0 == agg_round_3 == agg_round_6


class TestThresholdCalculation:
    """Test threshold calculation for various clique sizes."""

    @pytest.mark.parametrize("clique_size,expected_threshold", [
        (1, 1),   # ceil(0.6667 * 1) = ceil(0.6667) = 1
        (2, 2),   # ceil(0.6667 * 2) = ceil(1.3334) = 2
        (3, 3),   # ceil(0.6667 * 3) = ceil(2.0001) = 3
        (4, 3),   # ceil(0.6667 * 4) = ceil(2.6668) = 3
        (5, 4),   # ceil(0.6667 * 5) = ceil(3.3335) = 4
        (6, 5),   # ceil(0.6667 * 6) = ceil(4.0002) = 5
        (10, 7),  # ceil(0.6667 * 10) = ceil(6.667) = 7
    ])
    def test_threshold_is_two_thirds_majority(self, clique_size, expected_threshold):
        """Verify threshold calculation matches 2/3 majority."""
        assert compute_clique_threshold(clique_size) == expected_threshold

    def test_threshold_raises_for_invalid_size(self):
        """Verify threshold raises for non-positive clique size."""
        with pytest.raises(ValueError):
            compute_clique_threshold(0)
        with pytest.raises(ValueError):
            compute_clique_threshold(-1)


class TestFindNodeClique:
    """Test finding which clique a node belongs to."""

    def test_find_node_in_clique(self):
        """Verify finding node's clique."""
        cliques = [{"node_0", "node_1"}, {"node_2", "node_3"}]

        idx, members = find_node_clique("node_0", cliques)
        assert idx == 0
        assert members == {"node_0", "node_1"}

        idx, members = find_node_clique("node_2", cliques)
        assert idx == 1
        assert members == {"node_2", "node_3"}

    def test_find_node_not_in_any_clique(self):
        """Verify error when node not found."""
        cliques = [{"node_0", "node_1"}, {"node_2", "node_3"}]

        with pytest.raises(ValueError, match="not found"):
            find_node_clique("node_99", cliques)


class TestEndToEndTopology:
    """End-to-end tests for topology construction and assignment."""

    def test_full_topology_flow(self):
        """Test complete topology construction flow."""
        # Step 1: Create labels
        labels = create_test_labels(1000, 10)

        # Step 2: Create partition
        partition = dirichlet_partition(
            dataset=list(range(1000)),
            labels=labels,
            num_clients=10,
            alpha=0.5,
            seed=42,
        )

        # Step 3: Compute node labels from partition
        node_labels = compute_node_labels_from_partition(partition, labels)

        # Step 4: Build full topology
        cliques, intra_edges, inter_edges, edge_counts = build_full_topology(
            node_labels=node_labels,
            clique_size=5,
            iterations=100,
            edge_mode="small_world",
            seed=42,
        )

        # Verify results
        assert len(cliques) == 2  # 10 clients / 5 per clique
        assert len(intra_edges) > 0  # Should have intra-clique edges
        assert len(edge_counts) == 10  # All nodes have edge counts

        # Verify all nodes assigned
        all_nodes = set()
        for clique in cliques:
            all_nodes.update(clique)
        assert len(all_nodes) == 10

    def test_topology_with_uneven_clients(self):
        """Test topology when clients don't divide evenly into cliques."""
        labels = create_test_labels(1000, 10)
        config = TopologyConfig(
            num_clients=7,  # Doesn't divide evenly by clique_size
            clique_size=3,
            alpha=0.5,
            seed=42,
        )

        servicer = TTPServicer(topology_config=config, labels=labels)

        # All clients assigned and each clique has at least 2 members
        total_nodes = sum(len(c) for c in servicer.topology.cliques)
        assert total_nodes == 7
        assert all(len(clique) >= 2 for clique in servicer.topology.cliques)
