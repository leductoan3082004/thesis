"""Partition utilities for federated datasets."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

import numpy as np


def dirichlet_partition(
    dataset: Sequence[int],
    labels: Mapping[int, int],
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Partition dataset indices across clients using a per-class Dirichlet draw.

    Returns a dict keyed as ``client_{i}`` where values are index lists.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    indices = list(dataset)
    if not indices:
        return {f"client_{i}": [] for i in range(num_clients)}

    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx in indices:
        if idx not in labels:
            raise KeyError(f"Missing label for index {idx}")
        by_label[int(labels[idx])].append(idx)

    client_bins: list[list[int]] = [[] for _ in range(num_clients)]
    for label_indices in by_label.values():
        label_arr = np.array(label_indices, dtype=np.int64)
        rng.shuffle(label_arr)
        probs = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        counts = rng.multinomial(len(label_arr), probs)

        start = 0
        for client_idx, count in enumerate(counts):
            if count == 0:
                continue
            end = start + count
            client_bins[client_idx].extend(label_arr[start:end].tolist())
            start = end

    # Ensure each client has at least one sample when dataset is large enough.
    if len(indices) >= num_clients:
        _rebalance_min_one_sample(client_bins, rng)

    for bucket in client_bins:
        rng.shuffle(bucket)

    return {f"client_{i}": client_bins[i] for i in range(num_clients)}


def _rebalance_min_one_sample(client_bins: list[list[int]], rng: np.random.Generator) -> None:
    empties = [idx for idx, bucket in enumerate(client_bins) if not bucket]
    if not empties:
        return

    donors = sorted(
        (idx for idx, bucket in enumerate(client_bins) if len(bucket) > 1),
        key=lambda i: len(client_bins[i]),
        reverse=True,
    )
    donor_ptr = 0
    for empty_idx in empties:
        while donor_ptr < len(donors) and len(client_bins[donors[donor_ptr]]) <= 1:
            donor_ptr += 1
        if donor_ptr >= len(donors):
            break
        donor_idx = donors[donor_ptr]
        take_pos = int(rng.integers(0, len(client_bins[donor_idx])))
        moved = client_bins[donor_idx].pop(take_pos)
        client_bins[empty_idx].append(moved)