"""Simple vector-based model used for tests and baselines."""

from __future__ import annotations

from typing import List, Sequence


class VectorModel:
    """Minimal vector model with apply/update helpers."""

    def __init__(self, weights: Sequence[float] | None = None) -> None:
        self.weights: List[float] = list(weights) if weights is not None else []

    def state_dict(self) -> List[float]:
        return list(self.weights)

    def load_state_dict(self, weights: Sequence[float]) -> None:
        self.weights = list(weights)

    def apply_update(self, update: Sequence[float]) -> None:
        if len(update) != len(self.weights):
            raise ValueError("Update length mismatch")
        self.weights = [w + u for w, u in zip(self.weights, update)]

    def compute_update(self, new_weights: Sequence[float]) -> List[float]:
        if len(new_weights) != len(self.weights):
            raise ValueError("Weight length mismatch")
        return [nw - w for w, nw in zip(self.weights, new_weights)]
