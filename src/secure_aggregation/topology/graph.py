import math
import os
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
    return _merge_singleton_cliques(cliques, node_distributions, global_distribution)


def _merge_singleton_cliques(
    cliques: List[Set[str]],
    node_distributions: Mapping[str, LabelDist],
    global_distribution: LabelDist,
) -> List[Set[str]]:
    """
    Ensure no clique is left with a single member by merging it into the least-disruptive clique.
    """
    if len(cliques) <= 1:
        return cliques
    idx = 0
    while idx < len(cliques):
        clique = cliques[idx]
        if len(clique) > 1:
            idx += 1
            continue
        # Merge singleton clique into the clique that minimizes skew increase.
        node = next(iter(clique))
        best_target: int | None = None
        best_delta = float("inf")
        for target_idx, target_clique in enumerate(cliques):
            if target_idx == idx:
                continue
            current_skew = compute_skew(target_clique, node_distributions, global_distribution)
            candidate = set(target_clique)
            candidate.add(node)
            new_skew = compute_skew(candidate, node_distributions, global_distribution)
            delta = new_skew - current_skew
            if delta < best_delta:
                best_delta = delta
                best_target = target_idx
        if best_target is None:
            idx += 1
            continue
        cliques[best_target].add(node)
        cliques.pop(idx)
        if len(cliques) <= 1:
            break
    return cliques


def _add_edge(edges: Set[Tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    edge = (min(a, b), max(a, b))
    edges.add(edge)


def build_interclique_edges(
    cliques: Sequence[Set[str]],
    mode: str = "ring_star",
    small_world_c: int = 2,
    ring_star_extra: int = 0,
) -> List[Tuple[int, int]]:
    """
    Construct inter-clique edges between clique indices.

    Args:
        cliques: Collection of node sets.
        mode: Inter-clique topology mode.
        small_world_c: Number of power-of-two offsets when mode == "small_world".
        ring_star_extra: Extra random edges per clique when mode == "ring_star".
    """
    num = len(cliques)
    edges: Set[Tuple[int, int]] = set()
    if num <= 1:
        return []
    if mode not in {"ring", "fractal", "small_world", "fully_connected", "ring_star"}:
        raise ValueError(f"Unknown edge mode '{mode}'")
    if small_world_c <= 0 and mode == "small_world":
        raise ValueError("small_world_c must be positive")
    if ring_star_extra < 0:
        raise ValueError("ring_star_extra must be non-negative")
    # Base connectivity keeps the graph connected.
    for i in range(num):
        _add_edge(edges, i, (i + 1) % num)
    if mode == "ring":
        return sorted(edges)
    if mode == "ring_star":
        # Choose the clique with the most nodes (ties broken by lowest index).
        central_idx = max(range(num), key=lambda idx: (len(cliques[idx]), -idx))
        for i in range(num):
            if i == central_idx:
                continue
            _add_edge(edges, central_idx, i)
        if ring_star_extra > 0:
            rng = random.Random(num)
            _add_ring_star_extra_edges(edges, num, ring_star_extra, rng)
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
    for k in range(small_world_c):
        offset = 2**k
        for i in range(num):
            _add_edge(edges, i, (i + offset) % num)
    return sorted(edges)


def _add_ring_star_extra_edges(
    edges: Set[Tuple[int, int]],
    num_cliques: int,
    extra_per_clique: int,
    rng: random.Random,
) -> None:
    """Add additional random edges for ring_star mode while avoiding duplicates."""
    adjacency: List[Set[int]] = [set() for _ in range(num_cliques)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    for clique_idx in range(num_cliques):
        for _ in range(extra_per_clique):
            candidates = [j for j in range(num_cliques) if j != clique_idx and j not in adjacency[clique_idx]]
            if not candidates:
                break
            target = rng.choice(candidates)
            _add_edge(edges, clique_idx, target)
            adjacency[clique_idx].add(target)
            adjacency[target].add(clique_idx)


def identify_central_clique(
    cliques: Sequence[Set[str]],
    interclique_edges: Sequence[Tuple[int, int]],
) -> Tuple[int | None, List[str]]:
    """
    Determine whether any clique is connected to every other clique (ring-star hub).

    Returns:
        Tuple of (central_clique_index or None, preferred_bridge_nodes).
    """
    clique_degrees: Counter[int] = Counter()
    for a, b in interclique_edges:
        clique_degrees[a] += 1
        clique_degrees[b] += 1
    if not clique_degrees:
        return None, []
    degree_list = [clique_degrees.get(idx, 0) for idx in range(len(cliques))]
    max_degree = max(degree_list)
    # Choose the lowest-index clique among those with max_degree to keep selection deterministic.
    candidate_indices = [idx for idx, degree in enumerate(degree_list) if degree == max_degree]
    if not candidate_indices:
        return None, []
    candidate = min(candidate_indices)
    if max_degree >= max(1, len(cliques) - 1):
        central_clique = sorted(cliques[candidate])
        bridge_target = min(_get_central_bridge_target(), len(central_clique))
        return candidate, central_clique[:bridge_target]
    return None, []


def _get_central_bridge_target() -> int:
    """Return desired number of bridge nodes for a ring-star hub."""
    raw = os.getenv("RING_STAR_NUMBER_OF_CENTRAL_NODES")
    if raw is None:
        return 2
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("RING_STAR_NUMBER_OF_CENTRAL_NODES must be an integer") from exc
    if value <= 0:
        raise ValueError("RING_STAR_NUMBER_OF_CENTRAL_NODES must be positive")
    return value


def _select_regular_node(clique_nodes: Set[str], node_edge_count: Mapping[str, int]) -> str:
    return min(clique_nodes, key=lambda n: (node_edge_count[n], n))


def _select_central_bridge_node(
    clique_nodes: Set[str],
    node_edge_count: Mapping[str, int],
    preferred_nodes: Sequence[str],
    central_limit: int,
    central_served_counts: Dict[str, int],
    central_assignments: Dict[int, str],
    target_clique_idx: int,
) -> str:
    """
    Pick a central bridge node while ensuring coverage and degree limits.

    Each outer clique is pinned to a single central node. New assignments favor
    the node that has served the fewest cliques and is still below the n/2 limit.
    """
    if target_clique_idx in central_assignments:
        assigned = central_assignments[target_clique_idx]
        if assigned in clique_nodes:
            return assigned

    intersection = [node for node in preferred_nodes if node in clique_nodes]
    if not intersection:
        return _select_regular_node(clique_nodes, node_edge_count)

    def _key(node: str) -> Tuple[bool, int, int, str]:
        served = central_served_counts.get(node, 0)
        over_limit = central_limit > 0 and served >= central_limit
        return (over_limit, served, node_edge_count[node], node)

    best = min(intersection, key=_key)
    central_assignments[target_clique_idx] = best
    central_served_counts[best] = central_served_counts.get(best, 0) + 1
    return best


def assign_node_edges(
    cliques: Sequence[Set[str]],
    interclique_edges: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Assign inter-clique edges to specific nodes using load-balanced greedy selection.

    For each inter-clique edge (clique_a, clique_b), picks the node with the lowest
    current edge count from each clique to form the actual node-to-node connection.
    When a clique is connected to every other clique (the ring-star hub), up to
    three nodes inside that clique are preferentially selected to serve as bridge
    nodes so they become the high-connectivity backbone for convergence checks.
    Each hub node is capped at serving all other cliques (n-1) and the algorithm
    adds direct edges so every hub node maintains a one-hop connection to each
    outer clique.

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

    central_idx, central_bridge_nodes = identify_central_clique(cliques, interclique_edges)
    central_limit = (len(cliques) - 1) if central_idx is not None else 0
    central_served_counts: Dict[str, int] = {node: 0 for node in central_bridge_nodes}
    central_assignments: Dict[int, str] = {}

    node_to_node_edges: List[Tuple[str, str]] = []
    adjacency: Dict[str, Set[str]] = defaultdict(set)

    def _record_edge(a: str, b: str) -> None:
        node_to_node_edges.append((a, b))
        adjacency[a].add(b)
        adjacency[b].add(a)
        node_edge_count[a] += 1
        node_edge_count[b] += 1

    for clique_a_idx, clique_b_idx in interclique_edges:
        clique_a = cliques[clique_a_idx]
        clique_b = cliques[clique_b_idx]

        if central_idx is not None and clique_a_idx == central_idx:
            best_a = _select_central_bridge_node(
                clique_a,
                node_edge_count,
                central_bridge_nodes,
                central_limit,
                central_served_counts,
                central_assignments,
                target_clique_idx=clique_b_idx,
            )
        else:
            best_a = _select_regular_node(clique_a, node_edge_count)

        if central_idx is not None and clique_b_idx == central_idx:
            best_b = _select_central_bridge_node(
                clique_b,
                node_edge_count,
                central_bridge_nodes,
                central_limit,
                central_served_counts,
                central_assignments,
                target_clique_idx=clique_a_idx,
            )
        else:
            best_b = _select_regular_node(clique_b, node_edge_count)

        _record_edge(best_a, best_b)

    if central_idx is not None and central_bridge_nodes:
        for clique_idx, clique in enumerate(cliques):
            if clique_idx == central_idx:
                continue
            for central_node in central_bridge_nodes:
                if any(neigh in clique for neigh in adjacency.get(central_node, set())):
                    continue
                target_node = _select_regular_node(clique, node_edge_count)
                central_assignments[clique_idx] = central_node
                central_served_counts[central_node] = central_served_counts.get(central_node, 0) + 1
                _record_edge(central_node, target_node)

    return node_to_node_edges, node_edge_count


def build_full_topology(
    node_labels: Mapping[str, Mapping[str, float] | str],
    clique_size: int,
    iterations: int = 1000,
    edge_mode: str = "small_world",
    small_world_c: int = 2,
    ring_star_extra: int = 0,
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
        ring_star_extra: Additional random edges per clique when using ring_star mode.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of:
        - cliques: List of node sets.
        - intra_edges: List of (node_a, node_b) edges within cliques.
        - inter_edges: List of (node_a, node_b) edges between cliques.
        - node_edge_count: Dict mapping each node to its final edge count.
    """
    cliques = build_d_cliques(node_labels, clique_size, iterations, seed)
    interclique_edges = build_interclique_edges(
        cliques,
        mode=edge_mode,
        small_world_c=small_world_c,
        ring_star_extra=ring_star_extra,
    )

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
    Compute threshold as 2/3 majority of clique size, capped at n-1.

    The cap at n-1 is required because in the unmasking phase, each node
    can only provide b-shares for OTHER survivors (not itself). So the
    maximum collectible b-shares per survivor is n-1.

    Args:
        clique_size: Number of nodes in the clique.

    Returns:
        Minimum number of survivors needed for secure aggregation.
    """
    if clique_size <= 0:
        raise ValueError("clique_size must be positive")
    two_thirds = math.ceil(0.6667 * clique_size)
    return two_thirds


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


def get_bridge_nodes(inter_edges: Sequence[Tuple[str, str]]) -> Set[str]:
    """
    Identify all bridge nodes (nodes with inter-clique connections).

    Args:
        inter_edges: List of (node_a, node_b) inter-clique edges.

    Returns:
        Set of node IDs that are bridge nodes.
    """
    bridge_nodes: Set[str] = set()
    for node_a, node_b in inter_edges:
        bridge_nodes.add(node_a)
        bridge_nodes.add(node_b)
    return bridge_nodes


def get_inter_clique_neighbors(node_id: str, inter_edges: Sequence[Tuple[str, str]]) -> List[str]:
    """
    Get all inter-clique neighbors for a given node.

    Args:
        node_id: The node identifier.
        inter_edges: List of (node_a, node_b) inter-clique edges.

    Returns:
        List of node IDs that are inter-clique neighbors.
    """
    neighbors: List[str] = []
    for a, b in inter_edges:
        if a == node_id:
            neighbors.append(b)
        elif b == node_id:
            neighbors.append(a)
    return neighbors


def is_bridge_node(node_id: str, inter_edges: Sequence[Tuple[str, str]]) -> bool:
    """
    Check if a node is a bridge node.

    Args:
        node_id: The node identifier.
        inter_edges: List of (node_a, node_b) inter-clique edges.

    Returns:
        True if node has inter-clique connections.
    """
    for a, b in inter_edges:
        if a == node_id or b == node_id:
            return True
    return False


def get_clique_bridge_nodes(
    clique_idx: int,
    cliques: Sequence[Set[str]],
    inter_edges: Sequence[Tuple[str, str]],
) -> Set[str]:
    """
    Get bridge nodes within a specific clique.

    Args:
        clique_idx: Index of the clique.
        cliques: List of clique sets.
        inter_edges: List of (node_a, node_b) inter-clique edges.

    Returns:
        Set of bridge node IDs in the specified clique.
    """
    clique_members = cliques[clique_idx]
    all_bridge_nodes = get_bridge_nodes(inter_edges)
    return clique_members & all_bridge_nodes


def get_neighbor_clique_indices(
    clique_idx: int,
    cliques: Sequence[Set[str]],
    inter_edges: Sequence[Tuple[str, str]],
) -> Set[int]:
    """
    Get indices of cliques connected to the given clique.

    Args:
        clique_idx: Index of the clique.
        cliques: List of clique sets.
        inter_edges: List of (node_a, node_b) inter-clique edges.

    Returns:
        Set of clique indices that are neighbors.
    """
    clique_members = cliques[clique_idx]
    node_to_clique = {}
    for idx, clique in enumerate(cliques):
        for node in clique:
            node_to_clique[node] = idx

    neighbor_indices: Set[int] = set()
    for a, b in inter_edges:
        if a in clique_members:
            neighbor_indices.add(node_to_clique[b])
        elif b in clique_members:
            neighbor_indices.add(node_to_clique[a])

    return neighbor_indices


def generate_preliminary_topology(num_nodes: int, clique_size: int) -> Dict:
    """
    Generate a preliminary topology based on node count and clique size.

    This creates a static topology for dashboard/monitoring setup before
    the actual data-aware topology is generated by TTP at runtime.

    Args:
        num_nodes: Total number of nodes.
        clique_size: Target size for each clique.

    Returns:
        Dictionary with topology structure (num_cliques, cliques, inter_edges, edge_counts).
    """
    if clique_size <= 0:
        clique_size = num_nodes

    num_cliques = (num_nodes + clique_size - 1) // clique_size
    cliques: List[List[str]] = []

    for i in range(num_cliques):
        start_idx = i * clique_size
        end_idx = min(start_idx + clique_size, num_nodes)
        clique_nodes = [f"node_{j}" for j in range(start_idx, end_idx)]
        cliques.append(sorted(clique_nodes))

    return {
        "num_cliques": num_cliques,
        "cliques": cliques,
        "inter_edges": [],
        "edge_counts": {},
    }


def compute_node_degrees(cliques: Sequence[Iterable[str]], inter_edges: Sequence[Tuple[str, str]]) -> Dict[str, int]:
    """
    Compute the degree of each node in the topology.

    Args:
        cliques: List of cliques, each containing node IDs.
        inter_edges: List of inter-clique edges.

    Returns:
        Dictionary mapping node_id to its degree (number of connections).
    """
    degrees: Dict[str, int] = {}

    for clique in cliques:
        clique_nodes = list(clique)
        clique_size = len(clique_nodes)
        for node in clique_nodes:
            degrees[node] = clique_size - 1

    for node_a, node_b in inter_edges:
        degrees[node_a] = degrees.get(node_a, 0) + 1
        degrees[node_b] = degrees.get(node_b, 0) + 1

    return degrees


def compute_max_degree(cliques: Sequence[Iterable[str]], inter_edges: Sequence[Tuple[str, str]]) -> int:
    """
    Compute the maximum node degree in the topology (d_max).

    Args:
        cliques: List of cliques, each containing node IDs.
        inter_edges: List of inter-clique edges.

    Returns:
        Maximum degree across all nodes.
    """
    degrees = compute_node_degrees(cliques, inter_edges)
    return max(degrees.values()) if degrees else 0


def compute_average_degree(cliques: Sequence[Iterable[str]], inter_edges: Sequence[Tuple[str, str]]) -> float:
    """
    Compute the average node degree across the topology.

    Args:
        cliques: List of cliques, each containing node IDs.
        inter_edges: List of inter-clique edges.

    Returns:
        Average degree across all nodes.
    """
    degrees = compute_node_degrees(cliques, inter_edges)
    return sum(degrees.values()) / len(degrees) if degrees else 0.0
