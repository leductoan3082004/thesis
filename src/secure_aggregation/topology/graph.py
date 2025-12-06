import math
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, MutableSequence, Sequence, Set, Tuple

LabelDist = Dict[str, float]


def _normalize(counts: Mapping[str, float]) -> LabelDist:
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("Label counts must sum to a positive value")
    return {label: value / total for label, value in counts.items()}


def compute_label_distribution(node_labels: Mapping[str, Mapping[str, float] | str]) -> Dict[str, LabelDist]:
    """
    Normalizes label distributions per node and returns a mapping node_id -> distribution.
    Accepts either a mapping of label->count or a single label string.
    """
    normalized: Dict[str, LabelDist] = {}
    for node_id, labels in node_labels.items():
        if isinstance(labels, str):
            counts = Counter({labels: 1.0})
        else:
            counts = Counter({k: float(v) for k, v in labels.items()})
        normalized[node_id] = _normalize(counts)
    return normalized


def _aggregate_clique_distribution(clique: Iterable[str], node_distributions: Mapping[str, LabelDist]) -> LabelDist:
    agg: Dict[str, float] = defaultdict(float)
    for node in clique:
        if node not in node_distributions:
            raise ValueError(f"Node '{node}' missing label distribution")
        for label, prob in node_distributions[node].items():
            agg[label] += prob
    return _normalize(agg)


def compute_skew(clique: Iterable[str], node_distributions: Mapping[str, LabelDist], global_distribution: LabelDist) -> float:
    clique_dist = _aggregate_clique_distribution(clique, node_distributions)
    labels = set(clique_dist) | set(global_distribution)
    return sum(abs(clique_dist.get(label, 0.0) - global_distribution.get(label, 0.0)) for label in labels)


def build_d_cliques(
    node_labels: Mapping[str, Mapping[str, float] | str],
    clique_size: int,
    iterations: int = 1000,
    seed: int | None = None,
) -> List[Set[str]]:
    """
    Build D-cliques using greedy swaps to reduce label skew.
    """
    if clique_size <= 0:
        raise ValueError("clique_size must be positive")
    nodes = list(node_labels.keys())
    if not nodes:
        raise ValueError("node_labels cannot be empty")
    rng = random.Random(seed)
    rng.shuffle(nodes)
    node_distributions = compute_label_distribution(node_labels)
    global_counts: Counter[str] = Counter()
    for dist in node_distributions.values():
        for label, prob in dist.items():
            global_counts[label] += prob
    global_distribution = _normalize(global_counts)
    cliques: List[Set[str]] = []
    for i in range(0, len(nodes), clique_size):
        cliques.append(set(nodes[i : i + clique_size]))
    if len(cliques) == 1:
        return cliques
    for _ in range(iterations):
        idx_a, idx_b = rng.sample(range(len(cliques)), 2)
        clique_a = cliques[idx_a]
        clique_b = cliques[idx_b]
        base_skew = compute_skew(clique_a, node_distributions, global_distribution) + compute_skew(
            clique_b, node_distributions, global_distribution
        )
        improvements: List[Tuple[str, str]] = []
        for a in clique_a:
            for b in clique_b:
                new_a = clique_a.copy()
                new_b = clique_b.copy()
                new_a.remove(a)
                new_a.add(b)
                new_b.remove(b)
                new_b.add(a)
                new_skew = compute_skew(new_a, node_distributions, global_distribution) + compute_skew(
                    new_b, node_distributions, global_distribution
                )
                if new_skew < base_skew:
                    improvements.append((a, b))
        if improvements:
            swap_a, swap_b = rng.choice(improvements)
            cliques[idx_a].remove(swap_a)
            cliques[idx_a].add(swap_b)
            cliques[idx_b].remove(swap_b)
            cliques[idx_b].add(swap_a)
    return cliques


def _add_edge(edges: Set[Tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    edge = (min(a, b), max(a, b))
    edges.add(edge)


def build_interclique_edges(
    cliques: Sequence[Set[str]], mode: str = "small_world", small_world_c: int = 2
) -> List[Tuple[int, int]]:
    """
    Construct inter-clique edges between clique indices.
    """
    num = len(cliques)
    edges: Set[Tuple[int, int]] = set()
    if num <= 1:
        return []
    if mode not in {"ring", "fractal", "small_world", "fully_connected"}:
        raise ValueError(f"Unknown edge mode '{mode}'")
    # Base connectivity keeps the graph connected.
    for i in range(num):
        _add_edge(edges, i, (i + 1) % num)
    if mode == "ring":
        return sorted(edges)
    if mode == "fully_connected":
        for i in range(num):
            for j in range(i + 1, num):
                _add_edge(edges, i, j)
        return sorted(edges)
    if mode == "fractal":
        stride = max(2, num // 2)
        for i in range(num):
            _add_edge(edges, i, (i + stride) % num)
        return sorted(edges)
    # small_world
    if small_world_c <= 0:
        raise ValueError("small_world_c must be positive")
    for k in range(small_world_c):
        offset = 2**k
        for i in range(num):
            _add_edge(edges, i, (i + offset) % num)
    return sorted(edges)


def assign_node_edges(
    cliques: Sequence[Set[str]],
    interclique_edges: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Assign inter-clique edges to specific nodes using load-balanced greedy selection.

    For each inter-clique edge (clique_a, clique_b), picks the node with the lowest
    current edge count from each clique to form the actual node-to-node connection.

    Args:
        cliques: List of node sets, where each set represents a clique.
        interclique_edges: List of (clique_idx_a, clique_idx_b) tuples.

    Returns:
        A tuple of:
        - List of (node_a, node_b) edges between specific nodes.
        - Dict mapping each node to its final edge count.
    """
    node_edge_count: Dict[str, int] = {}

    for clique in cliques:
        clique_size = len(clique)
        for node in clique:
            node_edge_count[node] = clique_size - 1

    node_to_node_edges: List[Tuple[str, str]] = []

    for clique_a_idx, clique_b_idx in interclique_edges:
        clique_a = cliques[clique_a_idx]
        clique_b = cliques[clique_b_idx]

        best_a = min(clique_a, key=lambda n: (node_edge_count[n], n))
        best_b = min(clique_b, key=lambda n: (node_edge_count[n], n))

        node_to_node_edges.append((best_a, best_b))
        node_edge_count[best_a] += 1
        node_edge_count[best_b] += 1

    return node_to_node_edges, node_edge_count


def build_full_topology(
    node_labels: Mapping[str, Mapping[str, float] | str],
    clique_size: int,
    iterations: int = 1000,
    edge_mode: str = "small_world",
    small_world_c: int = 2,
    seed: int | None = None,
) -> Tuple[List[Set[str]], List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, int]]:
    """
    Build the complete D-Cliques topology with node-level edges.

    This is a convenience function that combines all topology construction steps:
    1. Build D-cliques with greedy swaps
    2. Build inter-clique edges
    3. Assign node-to-node edges with load balancing

    Args:
        node_labels: Mapping of node_id -> label distribution or single label.
        clique_size: Maximum size of each clique.
        iterations: Number of greedy swap iterations.
        edge_mode: Inter-clique edge mode ("ring", "small_world", "fractal", "fully_connected").
        small_world_c: Parameter for small_world mode (number of power-of-2 offsets).
        seed: Random seed for reproducibility.

    Returns:
        A tuple of:
        - cliques: List of node sets.
        - intra_edges: List of (node_a, node_b) edges within cliques.
        - inter_edges: List of (node_a, node_b) edges between cliques.
        - node_edge_count: Dict mapping each node to its final edge count.
    """
    cliques = build_d_cliques(node_labels, clique_size, iterations, seed)
    interclique_edges = build_interclique_edges(cliques, mode=edge_mode, small_world_c=small_world_c)

    intra_edges: List[Tuple[str, str]] = []
    for clique in cliques:
        nodes = sorted(clique)
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                intra_edges.append((n1, n2))

    inter_edges, node_edge_count = assign_node_edges(cliques, interclique_edges)

    return cliques, intra_edges, inter_edges, node_edge_count


def metropolis_hastings_weights(num_nodes: int, edges: Iterable[Tuple[int, int]]) -> List[List[float]]:
    """
    Compute row-stochastic Metropolis-Hastings weights for an undirected graph.
    """
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    weights: List[List[float]] = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in neighbors[i]:
            weights[i][j] = 1.0 / (1.0 + max(len(neighbors[i]), len(neighbors[j])))
        row_sum = sum(weights[i])
        weights[i][i] = max(0.0, 1.0 - row_sum)
    return weights


def compute_node_labels_from_partition(
    partition: Mapping[str, Sequence[int]],
    labels: Mapping[int, int],
) -> Dict[str, Dict[str, int]]:
    """
    Convert a data partition to node label distributions.

    Args:
        partition: Mapping of node_id -> list of data indices assigned to that node.
        labels: Mapping of data_index -> label (e.g., {0: 5, 1: 3, ...} for MNIST).

    Returns:
        Mapping of node_id -> {label: count} for each node.
    """
    node_labels: Dict[str, Dict[str, int]] = {}
    for node_id, indices in partition.items():
        counts: Counter[str] = Counter()
        for idx in indices:
            counts[str(labels[idx])] += 1
        node_labels[node_id] = dict(counts)
    return node_labels


def compute_clique_threshold(clique_size: int) -> int:
    """
    Compute threshold as 2/3 majority of clique size.

    Args:
        clique_size: Number of nodes in the clique.

    Returns:
        Minimum number of survivors needed for secure aggregation.
    """
    if clique_size <= 0:
        raise ValueError("clique_size must be positive")
    return math.ceil(0.6667 * clique_size)


def find_node_clique(node_id: str, cliques: Sequence[Set[str]]) -> Tuple[int, Set[str]]:
    """
    Find which clique a node belongs to.

    Args:
        node_id: The node identifier.
        cliques: List of clique sets.

    Returns:
        Tuple of (clique_index, clique_members).

    Raises:
        ValueError: If node is not found in any clique.
    """
    for idx, clique in enumerate(cliques):
        if node_id in clique:
            return idx, clique
    raise ValueError(f"Node '{node_id}' not found in any clique")


def elect_clique_aggregator(clique_members: Iterable[str], round_idx: int) -> str:
    """
    Elect aggregator for a clique using round-robin.

    Args:
        clique_members: Set or list of node IDs in the clique.
        round_idx: Current training round index.

    Returns:
        Node ID of the elected aggregator.
    """
    sorted_members = sorted(clique_members)
    if not sorted_members:
        raise ValueError("clique_members cannot be empty")
    return sorted_members[round_idx % len(sorted_members)]
