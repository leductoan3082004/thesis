from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

from secure_aggregation.topology import (
    assign_node_edges,
    build_d_cliques,
    build_full_topology,
    build_interclique_edges,
    compute_average_degree,
    compute_label_distribution,
    compute_max_degree,
    compute_node_degrees,
    compute_skew,
    metropolis_hastings_weights,
)
from secure_aggregation.topology.graph import identify_central_clique


def _global_distribution(node_distributions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    total = Counter()
    for dist in node_distributions.values():
        for label, prob in dist.items():
            total[label] += prob
    denom = float(sum(total.values()))
    return {k: v / denom for k, v in total.items()}


def _average_skew(cliques: List[Set[str]], node_distributions: Dict[str, Dict[str, float]], global_dist: Dict[str, float]) -> float:
    return sum(compute_skew(clique, node_distributions, global_dist) for clique in cliques) / len(cliques)


def _is_connected(num_nodes: int, edges: Iterable[Tuple[int, int]]) -> bool:
    adjacency = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    seen = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node] - seen)
    return len(seen) == num_nodes


def test_greedy_swaps_reduce_average_skew() -> None:
    node_labels = {
        "n1": {"cat": 0.9, "dog": 0.1},
        "n2": {"cat": 0.8, "dog": 0.2},
        "n3": {"dog": 0.9, "cat": 0.1},
        "n4": {"dog": 0.8, "cat": 0.2},
        "n5": {"cat": 0.7, "dog": 0.3},
        "n6": {"dog": 0.7, "cat": 0.3},
    }
    node_dist = compute_label_distribution(node_labels)
    global_dist = _global_distribution(node_dist)
    base_cliques = build_d_cliques(node_labels, clique_size=2, iterations=0, seed=42)
    improved = build_d_cliques(node_labels, clique_size=2, iterations=200, seed=42)
    assert _average_skew(improved, node_dist, global_dist) <= _average_skew(base_cliques, node_dist, global_dist)


def test_interclique_edges_are_connected_and_bounded() -> None:
    cliques = [set([f"n{i*2}", f"n{i*2+1}"]) for i in range(5)]
    edges = build_interclique_edges(cliques, mode="small_world", small_world_c=2)
    assert len(edges) >= len(cliques)  # at least a ring
    assert len(edges) <= len(cliques) * 3  # ring + offsets, with deduplication
    assert _is_connected(len(cliques), edges)


def test_metropolis_hastings_weights_are_row_stochastic() -> None:
    cliques = [set([f"n{i}"]) for i in range(4)]
    edges = build_interclique_edges(cliques, mode="ring")
    weights = metropolis_hastings_weights(len(cliques), edges)
    for row in weights:
        assert all(value >= 0 for value in row)
        assert abs(sum(row) - 1.0) < 1e-9


def test_assign_node_edges_produces_correct_count() -> None:
    cliques = [{"n0", "n1", "n2"}, {"n3", "n4", "n5"}, {"n6", "n7", "n8"}]
    interclique_edges = [(0, 1), (1, 2), (0, 2)]
    node_edges, edge_counts = assign_node_edges(cliques, interclique_edges)

    assert len(node_edges) >= len(interclique_edges)

    node_to_clique = {}
    for idx, clique in enumerate(cliques):
        for node in clique:
            node_to_clique[node] = idx

    realized_pairs = {(min(node_to_clique[a], node_to_clique[b]), max(node_to_clique[a], node_to_clique[b])) for a, b in node_edges}
    for expected_pair in interclique_edges:
        assert tuple(sorted(expected_pair)) in realized_pairs

    # All inter-clique edges should still connect different cliques.
    for node_a, node_b in node_edges:
        clique_a_idx = node_to_clique[node_a]
        clique_b_idx = node_to_clique[node_b]
        assert clique_a_idx != clique_b_idx


def test_assign_node_edges_load_balances() -> None:
    cliques = [{"n0", "n1", "n2"}, {"n3", "n4", "n5"}, {"n6", "n7", "n8"}, {"n9", "n10", "n11"}]
    interclique_edges = build_interclique_edges(cliques, mode="fully_connected")
    _, edge_counts = assign_node_edges(cliques, interclique_edges)
    central_idx, _ = identify_central_clique(cliques, interclique_edges)

    for idx, clique in enumerate(cliques):
        counts_in_clique = [edge_counts[n] for n in clique]
        if central_idx is not None and idx == central_idx:
            continue
        max_diff = max(counts_in_clique) - min(counts_in_clique)
        assert max_diff <= 1, f"Load imbalance in clique: {counts_in_clique}"


def test_build_full_topology_returns_all_components() -> None:
    node_labels = {f"n{i}": {"A": 0.5, "B": 0.5} for i in range(12)}
    cliques, intra_edges, inter_edges, edge_counts = build_full_topology(
        node_labels, clique_size=3, iterations=10, edge_mode="small_world", seed=42
    )

    assert len(cliques) == 4
    assert all(len(c) == 3 for c in cliques)

    expected_intra = 4 * 3  # 4 cliques, each with C(3,2)=3 edges
    assert len(intra_edges) == expected_intra

    assert len(inter_edges) > 0

    all_nodes = set()
    for c in cliques:
        all_nodes.update(c)
    assert len(all_nodes) == 12
    assert all(n in edge_counts for n in all_nodes)


def test_build_full_topology_edges_connect_correct_nodes() -> None:
    node_labels = {f"n{i}": {"A": 0.5, "B": 0.5} for i in range(9)}
    cliques, intra_edges, inter_edges, _ = build_full_topology(
        node_labels, clique_size=3, iterations=10, edge_mode="ring", seed=42
    )

    node_to_clique = {}
    for idx, clique in enumerate(cliques):
        for node in clique:
            node_to_clique[node] = idx

    for n1, n2 in intra_edges:
        assert node_to_clique[n1] == node_to_clique[n2], "Intra-edge connects different cliques"

    for n1, n2 in inter_edges:
        assert node_to_clique[n1] != node_to_clique[n2], "Inter-edge connects same clique"


def test_compute_node_degrees_simple_topology() -> None:
    cliques = [{"n0", "n1", "n2"}, {"n3", "n4", "n5"}]
    inter_edges = [("n0", "n3"), ("n1", "n4")]

    degrees = compute_node_degrees(cliques, inter_edges)

    assert degrees["n0"] == 3
    assert degrees["n1"] == 3
    assert degrees["n2"] == 2
    assert degrees["n3"] == 3
    assert degrees["n4"] == 3
    assert degrees["n5"] == 2


def test_compute_node_degrees_no_inter_edges() -> None:
    cliques = [{"n0", "n1", "n2"}, {"n3", "n4"}]
    inter_edges = []

    degrees = compute_node_degrees(cliques, inter_edges)

    assert degrees["n0"] == 2
    assert degrees["n1"] == 2
    assert degrees["n2"] == 2
    assert degrees["n3"] == 1
    assert degrees["n4"] == 1


def test_compute_max_degree() -> None:
    cliques = [{"n0", "n1", "n2"}, {"n3", "n4", "n5"}]
    inter_edges = [("n0", "n3"), ("n0", "n4"), ("n1", "n3")]

    max_degree = compute_max_degree(cliques, inter_edges)

    assert max_degree == 4


def test_compute_max_degree_single_clique() -> None:
    cliques = [{"n0", "n1", "n2", "n3"}]
    inter_edges = []

    max_degree = compute_max_degree(cliques, inter_edges)

    assert max_degree == 3


def test_compute_average_degree() -> None:
    cliques = [{"n0", "n1", "n2"}, {"n3", "n4", "n5"}]
    inter_edges = [("n0", "n3")]

    avg_degree = compute_average_degree(cliques, inter_edges)

    assert abs(avg_degree - 2.333) < 0.01


def test_compute_average_degree_fully_connected_cliques() -> None:
    cliques = [{"n0", "n1"}, {"n2", "n3"}]
    inter_edges = [("n0", "n2"), ("n0", "n3"), ("n1", "n2"), ("n1", "n3")]

    avg_degree = compute_average_degree(cliques, inter_edges)

    assert avg_degree == 3.0


def test_degree_calculation_with_real_topology() -> None:
    node_labels = {f"n{i}": {"A": 0.5, "B": 0.5} for i in range(12)}
    cliques, _, inter_edges, _ = build_full_topology(
        node_labels, clique_size=3, iterations=10, edge_mode="ring", seed=42
    )

    degrees = compute_node_degrees(cliques, inter_edges)
    max_degree = compute_max_degree(cliques, inter_edges)
    avg_degree = compute_average_degree(cliques, inter_edges)

    assert len(degrees) == 12
    assert max_degree >= 2
    assert 2.0 <= avg_degree <= 11.0
    assert max_degree == max(degrees.values())
