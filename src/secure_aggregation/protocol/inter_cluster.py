"""
Inter-cluster aggregation with adaptive clipping and weighted averaging.

This module implements the cluster-level merge algorithm for combining
intra-cluster models with neighbor cluster models from IPFS.

Reference:
- McMahan et al., "Adaptive Clipping for Private Federated Learning", NeurIPS 2021
- Lian et al., "Can Decentralized Algorithms Outperform Centralized Algorithms?", NeurIPS 2017
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class MergeConfig:
    """Configuration for inter-cluster merge algorithm."""

    window_size: int = 10
    alpha: float = 0.5
    base_gamma: float = 0.2
    initial_c_cluster: float = 1.0


class AdaptiveClipper:
    """
    Maintains a sliding window of delta norms and computes adaptive clipping threshold.

    The threshold is set to the 90th percentile of historical norms, ensuring
    that extreme outliers (potential poisoning) are clipped while allowing
    natural variation.
    """

    def __init__(self, window_size: int = 10, initial_threshold: float = 1.0) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.norm_history: List[float] = []
        self._initial_threshold = initial_threshold

    def update(self, norms: List[float]) -> None:
        """Add new norms to the sliding window."""
        self.norm_history.extend(norms)
        max_history = self.window_size * max(len(norms), 1)
        if len(self.norm_history) > max_history:
            self.norm_history = self.norm_history[-max_history:]

    def get_threshold(self) -> float:
        """Compute adaptive threshold as 90th percentile of historical norms."""
        if not self.norm_history:
            return self._initial_threshold
        return float(np.percentile(self.norm_history, 90))

    def update_and_get_threshold(self, norms: List[float]) -> float:
        """Update history and return current threshold."""
        self.update(norms)
        return self.get_threshold()


def clip_delta(delta: np.ndarray, c_cluster: float) -> np.ndarray:
    """
    Clip delta vector to have maximum L2 norm of c_cluster.

    Args:
        delta: Difference vector between neighbor and local model.
        c_cluster: Maximum allowed L2 norm.

    Returns:
        Clipped delta vector with norm <= c_cluster.
    """
    norm = np.linalg.norm(delta)
    if norm <= c_cluster or norm == 0:
        return delta
    return delta * (c_cluster / norm)


def compute_adaptive_gamma(deltas: List[np.ndarray], alpha: float, base_gamma: float) -> float:
    """
    Compute adaptive step-size based on inter-model disagreement.

    When disagreement is high, gamma is reduced to be more conservative.
    When models agree, gamma is higher for faster mixing.

    Args:
        deltas: List of difference vectors from neighbor models.
        alpha: Sensitivity parameter (higher = more sensitive to disagreement).
        base_gamma: Base step-size (default mixing weight).

    Returns:
        Adaptive gamma value in (0, base_gamma].
    """
    if not deltas:
        return base_gamma
    avg_disagreement = float(np.mean([np.linalg.norm(d) for d in deltas]))
    return base_gamma / (1 + alpha * avg_disagreement)


class InterClusterMerger:
    """
    Merges intra-cluster model with verified neighbor cluster models.

    The merge uses:
    1. Difference computation (delta = neighbor - local)
    2. Adaptive norm-bounding (clip extreme deltas)
    3. Weighted average with uniform weights
    4. Convex combination with adaptive step-size
    """

    def __init__(self, config: Optional[MergeConfig] = None) -> None:
        self.config = config or MergeConfig()
        self.clipper = AdaptiveClipper(
            window_size=self.config.window_size,
            initial_threshold=self.config.initial_c_cluster,
        )

    def merge(
        self,
        theta_local: np.ndarray,
        neighbor_models: List[np.ndarray],
    ) -> np.ndarray:
        """
        Perform inter-cluster merge with adaptive clipping.

        Args:
            theta_local: Intra-cluster model from SAP.
            neighbor_models: List of verified neighbor cluster models.

        Returns:
            Merged model theta_cluster^(t+1).
        """
        if not neighbor_models:
            return theta_local.copy()

        deltas = [theta_j - theta_local for theta_j in neighbor_models]
        norms = [float(np.linalg.norm(d)) for d in deltas]
        c_cluster = self.clipper.update_and_get_threshold(norms)
        clipped_deltas = [clip_delta(d, c_cluster) for d in deltas]
        clipped_models = [theta_local + d for d in clipped_deltas]
        gamma = compute_adaptive_gamma(deltas, self.config.alpha, self.config.base_gamma)

        # Uniform weighted average.
        n_neighbors = len(clipped_models)
        weights = [1.0 / n_neighbors] * n_neighbors
        theta_robust = sum(w * m for w, m in zip(weights, clipped_models))

        theta_final = (1 - gamma) * theta_local + gamma * theta_robust
        return theta_final

    def get_current_threshold(self) -> float:
        """Return current clipping threshold for monitoring."""
        return self.clipper.get_threshold()
