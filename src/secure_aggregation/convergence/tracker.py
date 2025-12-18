"""Convergence tracking for federated learning with global coordination."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from secure_aggregation.utils import get_logger

logger = get_logger("convergence")


@dataclass
class ConvergenceSignal:
    """Message sent from a clique to the central convergence checker."""

    cluster_id: str
    round_idx: int
    converged: bool
    destination: Optional[str] = None


@dataclass
class ConvergenceConfig:
    """Configuration for convergence-driven training.

    Training continues until convergence is achieved. The `max_rounds` field now
    represents the number of warmup rounds to run before convergence detection
    (and signaling to bridge/central checker) begins. Set it to 0 to enable
    convergence tracking immediately.
    """

    enabled: bool = True
    max_rounds: int = 0
    tol_abs: float = 1e-5
    tol_rel: float = 0.001
    patience: int = 3
    require_neighbor_convergence: bool = True  # Deprecated: kept for backward compatibility.
    central_checker_id: Optional[str] = None
    signal_timeout: float = 30.0  # Seconds to wait after local convergence before resuming

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "ConvergenceConfig":
        """Create config from dictionary, using defaults for missing fields."""
        if data is None:
            return cls()
        return cls(
            enabled=data.get("enabled", True),
            max_rounds=data.get("max_rounds", 0),
            tol_abs=data.get("tol_abs", 1e-5),
            tol_rel=data.get("tol_rel", 0.001),
            patience=data.get("patience", 3),
            require_neighbor_convergence=data.get("require_neighbor_convergence", True),
            central_checker_id=data.get("central_checker_id"),
            signal_timeout=data.get("signal_timeout", 30.0),
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
    awaiting_global_confirmation: bool = False
    global_stop_round: Optional[int] = None
    last_signal_round: int = -1
    last_signal_status: bool = False


class ConvergenceTracker:
    """
    Tracks convergence state and determines when to stop training.

    Convergence is detected when the model delta stays below tolerance
    for `patience` consecutive rounds. For global convergence across clusters,
    also checks that all neighbor clusters have reported convergence.
    """

    def __init__(
        self,
        config: ConvergenceConfig,
        cluster_id: str,
        signal_sender: Optional[Callable[[ConvergenceSignal], None]] = None,
    ) -> None:
        self.config = config
        self.cluster_id = cluster_id
        self.state = ConvergenceState()
        self._signal_sender = signal_sender

    def update(self, current_model: np.ndarray) -> ConvergenceState:
        """
        Update convergence state after receiving new global model.

        Args:
            current_model: The current round's global model (post-merge).

        Returns:
            Updated convergence state with stop decision.
        """
        warmup_rounds = max(0, self.config.max_rounds)
        tracking_enabled = self.state.round_idx >= warmup_rounds

        if not self.config.enabled:
            self.state.round_idx += 1
            return self.state

        if tracking_enabled and self.state.prev_model is not None:
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

            if self._central_mode_enabled():
                self._report_convergence_status_to_central_checker(local_converged)
            elif self.state.cluster_converged:
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
        elif tracking_enabled:
            if self._central_mode_enabled():
                self._report_convergence_status_to_central_checker(False)

        self.state.prev_model = current_model.copy()
        self.state.round_idx += 1

        if (
            self._central_mode_enabled()
            and self.state.global_stop_round is not None
            and self.state.round_idx >= self.state.global_stop_round
        ):
            self.state.should_stop = True
            self.state.stop_reason = "global_convergence"

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

    def receive_global_convergence(self, round_idx: int) -> None:
        """
        Receive notification from the central checker that global convergence was achieved.
        """
        self.state.global_stop_round = round_idx
        self.state.awaiting_global_confirmation = False
        if self.state.round_idx >= round_idx:
            self.state.should_stop = True
            self.state.stop_reason = "global_convergence"
            logger.info(f"Central checker declared convergence at round {round_idx}")

    def set_signal_sender(
        self, sender: Callable[[ConvergenceSignal], None]
    ) -> None:
        """
        Register a callback used to send convergence signals to the central checker.
        """
        self._signal_sender = sender

    def reset(self) -> None:
        """Reset state for testing or restarting training."""
        self.state = ConvergenceState()

    def _central_mode_enabled(self) -> bool:
        return bool(self.config.central_checker_id)

    def _report_convergence_status_to_central_checker(self, local_converged: bool) -> None:
        """
        Central convergence flow: report status every round and wait for checker decision.
        """
        self._send_signal(local_converged)

        if local_converged:
            if not self.state.awaiting_global_confirmation:
                logger.info(
                    f"Cluster {self.cluster_id} waiting for central checker confirmation"
                )
            self.state.awaiting_global_confirmation = True
        else:
            self.state.awaiting_global_confirmation = False

    def _send_signal(self, converged: bool) -> None:
        """
        Emit a ConvergenceSignal to the configured central checker.
        """
        if self._signal_sender is None:
            logger.debug(
                "Central convergence enabled but no signal sender configured; skipping signal"
            )
            return

        if self.state.last_signal_round == self.state.round_idx and (
            self.state.last_signal_status == converged
        ):
            return

        destination = self.config.central_checker_id
        signal = ConvergenceSignal(
            cluster_id=self.cluster_id,
            round_idx=self.state.round_idx,
            converged=converged,
            destination=destination,
        )
        self._signal_sender(signal)
        self.state.last_signal_round = self.state.round_idx
        self.state.last_signal_status = converged
