"""Convergence tracking for federated learning with global coordination."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from secure_aggregation.utils import get_logger

logger = get_logger("convergence")


@dataclass
class ConvergenceConfig:
    """Configuration for convergence-driven training.

    Training continues until convergence is achieved. The max_rounds field
    serves only as a safety cap and should not be used to control training duration.
    """

    enabled: bool = True
    max_rounds: int = 999999999
    tol_abs: float = 1e-5
    tol_rel: float = 0.001
    patience: int = 3
    require_neighbor_convergence: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "ConvergenceConfig":
        """Create config from dictionary, using defaults for missing fields."""
        if data is None:
            return cls()
        return cls(
            enabled=data.get("enabled", True),
            max_rounds=data.get("max_rounds", 999999999),
            tol_abs=data.get("tol_abs", 1e-5),
            tol_rel=data.get("tol_rel", 0.001),
            patience=data.get("patience", 3),
            require_neighbor_convergence=data.get("require_neighbor_convergence", True),
        )


@dataclass
class ConvergenceState:
    """Current state of convergence tracking."""

    round_idx: int = 0
    prev_model: Optional[np.ndarray] = None
    delta_norm: float = float("inf")
    convergence_streak: int = 0
    cluster_converged: bool = False
    neighbor_convergence: Dict[str, bool] = field(default_factory=dict)
    should_stop: bool = False
    stop_reason: str = ""


class ConvergenceTracker:
    """
    Tracks convergence state and determines when to stop training.

    Convergence is detected when the model delta stays below tolerance
    for `patience` consecutive rounds. For global convergence across clusters,
    also checks that all neighbor clusters have reported convergence.
    """

    def __init__(self, config: ConvergenceConfig, cluster_id: str) -> None:
        self.config = config
        self.cluster_id = cluster_id
        self.state = ConvergenceState()

    def update(self, current_model: np.ndarray) -> ConvergenceState:
        """
        Update convergence state after receiving new global model.

        Args:
            current_model: The current round's global model (post-merge).

        Returns:
            Updated convergence state with stop decision.
        """
        if not self.config.enabled:
            self.state.round_idx += 1
            return self.state

        if self.state.prev_model is not None:
            delta = current_model - self.state.prev_model
            self.state.delta_norm = float(np.linalg.norm(delta))
            prev_norm = float(np.linalg.norm(self.state.prev_model))
            rel_delta = self.state.delta_norm / (prev_norm + 1e-10)

            local_converged = (
                self.state.delta_norm <= self.config.tol_abs
                or rel_delta <= self.config.tol_rel
            )

            if local_converged:
                self.state.convergence_streak += 1
                logger.info(
                    f"Round {self.state.round_idx}: delta={self.state.delta_norm:.2e}, "
                    f"streak={self.state.convergence_streak}/{self.config.patience}"
                )
            else:
                self.state.convergence_streak = 0
                logger.info(
                    f"Round {self.state.round_idx}: delta={self.state.delta_norm:.2e}, "
                    f"streak reset (above tolerance)"
                )

            self.state.cluster_converged = (
                self.state.convergence_streak >= self.config.patience
            )

            if self.state.cluster_converged:
                neighbors_converged = self._check_neighbors_converged()

                if neighbors_converged:
                    self.state.should_stop = True
                    self.state.stop_reason = "global_convergence"
                    logger.info(
                        f"Global convergence achieved at round {self.state.round_idx}"
                    )
                else:
                    logger.info(
                        f"Cluster converged but waiting for neighbors: "
                        f"{self.state.neighbor_convergence}"
                    )

        self.state.prev_model = current_model.copy()
        self.state.round_idx += 1

        if self.state.round_idx >= self.config.max_rounds:
            self.state.should_stop = True
            self.state.stop_reason = "max_rounds_reached"
            logger.info(f"Max rounds ({self.config.max_rounds}) reached, stopping")

        return self.state

    def _check_neighbors_converged(self) -> bool:
        """Check if all neighbor clusters have converged."""
        if not self.config.require_neighbor_convergence:
            return True

        if not self.state.neighbor_convergence:
            return True

        return all(self.state.neighbor_convergence.values())

    def receive_neighbor_convergence(self, cluster_id: str, converged: bool) -> None:
        """
        Update neighbor convergence status from ECM gossip.

        Args:
            cluster_id: The neighbor cluster's ID.
            converged: Whether the neighbor has converged.
        """
        self.state.neighbor_convergence[cluster_id] = converged
        logger.debug(f"Received convergence status from {cluster_id}: {converged}")

    def reset(self) -> None:
        """Reset state for testing or restarting training."""
        self.state = ConvergenceState()
