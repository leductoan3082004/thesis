"""
Tests for D-Cliques topology using real MNIST dataset.

These tests verify that the topology construction works correctly with
real-world non-IID data distributions.
"""

from collections import Counter
from typing import Dict, List, Set

import pytest

from secure_aggregation.data.partition import dirichlet_partition
from secure_aggregation.topology import (
    assign_node_edges,
    build_d_cliques,
    build_full_topology,
    build_interclique_edges,
    compute_label_distribution,
    compute_skew,
)

try:
    from torchvision import datasets

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def _load_mnist_labels() -> Dict[int, int]:
    """Load MNIST dataset and return index -> label mapping."""
    mnist = datasets.MNIST(root="./data", train=True, download=True)
    return {i: int(mnist.targets[i]) for i in range(len(mnist))}


def _compute_global_distribution(node_distributions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute global label distribution by averaging node distributions."""
    total: Counter[str] = Counter()
    for dist in node_distributions.values():
        for label, prob in dist.items():
            total[label] += prob
    denom = float(sum(total.values()))
    return {k: v / denom for k, v in total.items()}


def _average_skew(
    cliques: List[Set[str]], node_distributions: Dict[str, Dict[str, float]], global_dist: Dict[str, float]
) -> float:
    """Compute average skew across all cliques."""
    return sum(compute_skew(clique, node_distributions, global_dist) for clique in cliques) / len(cliques)


def _get_node_label_counts(partition: Dict[str, List[int]], labels: Dict[int, int]) -> Dict[str, Dict[str, float]]:
    """Convert partition to node label distribution."""
    node_labels: Dict[str, Dict[str, float]] = {}
    for client, indices in partition.items():
        counts: Counter[str] = Counter()
        for idx in indices:
            counts[str(labels[idx])] += 1
        node_labels[client] = dict(counts)
    return node_labels


@pytest.mark.skipif(not TORCHVISION_AVAILABLE, reason="torchvision not installed")
class TestTopologyWithMNIST:
    """Test suite for topology construction with real MNIST data."""

    @pytest.fixture(scope="class")
    def mnist_labels(self) -> Dict[int, int]:
        """Load MNIST labels once for all tests in this class."""
        return _load_mnist_labels()

    @pytest.fixture(scope="class")
    def non_iid_partition(self, mnist_labels: Dict[int, int]) -> Dict[str, List[int]]:
        """Create a non-IID partition of MNIST using Dirichlet distribution."""
        indices = list(mnist_labels.keys())
        return dirichlet_partition(indices, mnist_labels, num_clients=50, alpha=0.5, seed=42)

    @pytest.fixture(scope="class")
    def node_labels(
        self, non_iid_partition: Dict[str, List[int]], mnist_labels: Dict[int, int]
    ) -> Dict[str, Dict[str, float]]:
        """Get label distributions for each node."""
        return _get_node_label_counts(non_iid_partition, mnist_labels)

    def test_greedy_swaps_reduce_skew_on_mnist(self, node_labels: Dict[str, Dict[str, float]]) -> None:
        """Verify that greedy swaps reduce average skew on real MNIST data."""
        node_dist = compute_label_distribution(node_labels)
        global_dist = _compute_global_distribution(node_dist)

        base_cliques = build_d_cliques(node_labels, clique_size=10, iterations=0, seed=42)
        improved_cliques = build_d_cliques(node_labels, clique_size=10, iterations=500, seed=42)

        base_skew = _average_skew(base_cliques, node_dist, global_dist)
        improved_skew = _average_skew(improved_cliques, node_dist, global_dist)

        assert improved_skew <= base_skew, f"Skew should decrease: {improved_skew} > {base_skew}"

    def test_clique_distributions_approach_global(self, node_labels: Dict[str, Dict[str, float]]) -> None:
        """Verify that optimized cliques have distributions closer to global."""
        node_dist = compute_label_distribution(node_labels)
        global_dist = _compute_global_distribution(node_dist)

        cliques = build_d_cliques(node_labels, clique_size=10, iterations=500, seed=42)
        avg_skew = _average_skew(cliques, node_dist, global_dist)

        # A well-optimized topology should have low average skew.
        # Maximum possible skew for 10 labels is 2.0 (completely disjoint distributions).
        assert avg_skew < 1.0, f"Average skew too high: {avg_skew}"

    def test_load_balancing_with_mnist_topology(self, node_labels: Dict[str, Dict[str, float]]) -> None:
        """Verify load balancing works correctly with MNIST-based topology."""
        cliques = build_d_cliques(node_labels, clique_size=10, iterations=100, seed=42)
        interclique_edges = build_interclique_edges(cliques, mode="small_world", small_world_c=3)

        _, edge_counts = assign_node_edges(cliques, interclique_edges)

        for clique in cliques:
            counts = [edge_counts[n] for n in clique]
            max_diff = max(counts) - min(counts)
            assert max_diff <= 2, f"Load imbalance too high in clique: max_diff={max_diff}"

    def test_full_topology_with_mnist(self, node_labels: Dict[str, Dict[str, float]]) -> None:
        """Test complete topology construction with MNIST data."""
        cliques, intra_edges, inter_edges, edge_counts = build_full_topology(
            node_labels, clique_size=10, iterations=200, edge_mode="small_world", small_world_c=2, seed=42
        )

        num_cliques = len(cliques)
        assert num_cliques == 5, f"Expected 5 cliques for 50 nodes with size 10, got {num_cliques}"

        all_nodes = set()
        for clique in cliques:
            all_nodes.update(clique)
        assert len(all_nodes) == 50, f"Expected 50 unique nodes, got {len(all_nodes)}"

        expected_intra = num_cliques * (10 * 9 // 2)  # C(10,2) per clique
        assert len(intra_edges) == expected_intra

        assert len(inter_edges) > 0

        for node in all_nodes:
            assert node in edge_counts
            assert edge_counts[node] >= 9  # At least intra-clique edges

    def test_distribution_balance_across_cliques(self, node_labels: Dict[str, Dict[str, float]]) -> None:
        """Verify that label distributions are balanced across cliques after optimization."""
        node_dist = compute_label_distribution(node_labels)
        global_dist = _compute_global_distribution(node_dist)

        cliques = build_d_cliques(node_labels, clique_size=10, iterations=500, seed=42)

        skews = [compute_skew(clique, node_dist, global_dist) for clique in cliques]
        max_skew = max(skews)
        min_skew = min(skews)

        # Cliques should have relatively similar skew values after optimization.
        skew_variance = max_skew - min_skew
        assert skew_variance < 0.5, f"Skew variance too high: {skew_variance}"

        print(f"\nMNIST topology stats:")
        print(f"  Number of cliques: {len(cliques)}")
        print(f"  Average skew: {sum(skews) / len(skews):.4f}")
        print(f"  Min skew: {min_skew:.4f}")
        print(f"  Max skew: {max_skew:.4f}")
        print(f"  Global distribution: {sorted(global_dist.items())}")
