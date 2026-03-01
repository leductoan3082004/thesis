import random
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence


def iid_partition(indices: Sequence[int], num_clients: int) -> Dict[str, List[int]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    clients = [f"client_{i}" for i in range(num_clients)]
    assignment: Dict[str, List[int]] = {c: [] for c in clients}
    for idx in indices:
        client = clients[idx % num_clients]
        assignment[client].append(idx)
    return assignment


def _dirichlet_draw(alpha: float, labels: Iterable[int], num_clients: int, seed: int | None = None) -> Dict[str, Dict[int, float]]:
    """
    Generate per-client label proportions using independent Dirichlet draws for each label.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    rng = random.Random(seed)
    proportions: Dict[str, Dict[int, float]] = {f"client_{i}": {} for i in range(num_clients)}
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    for label in label_counts:
        samples = [rng.gammavariate(alpha, 1.0) for _ in range(num_clients)]
        total = sum(samples)
        for i, sample in enumerate(samples):
            proportions[f"client_{i}"][label] = sample / total
    return proportions


def dirichlet_partition(dataset: Sequence[int], labels: Mapping[int, int], num_clients: int, alpha: float, seed: int | None = None) -> Dict[str, List[int]]:
    """
    Non-IID partition using Dirichlet over label distributions.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    proportions = _dirichlet_draw(alpha, labels.values(), num_clients, seed=seed)
    assignments: Dict[str, List[int]] = {f"client_{i}": [] for i in range(num_clients)}
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in labels.items():
        label_to_indices[label].append(idx)
    rng = random.Random(seed)
    for label, idxs in label_to_indices.items():
        rng.shuffle(idxs)
        frac_left = {client: proportions[client][label] for client in assignments}
        total = sum(frac_left.values())
        frac_left = {c: v / total for c, v in frac_left.items()}
        for idx in idxs:
            target = max(frac_left.items(), key=lambda kv: kv[1])[0]
            assignments[target].append(idx)
            frac_left[target] = max(0.0, frac_left[target] - 1.0 / len(idxs))
    return assignments
