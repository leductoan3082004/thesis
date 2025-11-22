from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

from secure_aggregation.topology import (
    build_d_cliques,
    build_interclique_edges,
    compute_label_distribution,
    compute_skew,
    metropolis_hastings_weights,
)


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
