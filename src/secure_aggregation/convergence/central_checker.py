"""Central checker aggregates convergence signals and commits decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Optional, Sequence

from secure_aggregation.convergence.central_broadcast import publish_global_convergence
from secure_aggregation.storage.model_store import BlockchainInterface
from secure_aggregation.utils import get_logger

logger = get_logger("central_checker")


@dataclass
class ClusterConvergenceSignal:
    """Snapshot of a single cluster's convergence report."""

    converged: bool
    delta_norm: float
    received_at: float


@dataclass
class CentralChecker:
    """Tracks convergence signals and declares global convergence."""

    blockchain: Optional[BlockchainInterface]
    total_cliques: int
    cluster_ids: Sequence[str]
    on_announcement: Optional[Callable[[int, Optional[str], Dict], None]] = None
    _round_signals: Dict[int, Dict[str, ClusterConvergenceSignal]] = field(default_factory=dict)
    _finalized_rounds: set[int] = field(default_factory=set)

    def record_signal(
        self,
        cluster_id: str,
        round_idx: int,
        converged: bool,
        delta_norm: float = 0.0,
        received_at: Optional[float] = None,
    ) -> None:
        if round_idx in self._finalized_rounds:
            return
        signals = self._round_signals.setdefault(round_idx, {})
        prev = signals.get(cluster_id)
        if prev is not None and prev.converged == converged:
            return
        timestamp = received_at if received_at is not None else datetime.utcnow().timestamp()
        signals[cluster_id] = ClusterConvergenceSignal(converged, delta_norm, timestamp)
        logger.info(
            "Central checker received signal: cluster=%s, round=%s, converged=%s, delta=%.3e",
            cluster_id,
            round_idx,
            converged,
            delta_norm,
        )
        if self._has_all_signals(signals):
            if all(signal.converged for signal in signals.values()):
                self._finalize(round_idx, signals)
            else:
                # Not globally converged; keep waiting for next round.
                self._round_signals.pop(round_idx, None)

    def _has_all_signals(self, signals: Dict[str, ClusterConvergenceSignal]) -> bool:
        if self.cluster_ids:
            return all(cluster_id in signals for cluster_id in self.cluster_ids)
        return len(signals) >= self.total_cliques

    def _finalize(
        self,
        round_idx: int,
        signals: Dict[str, ClusterConvergenceSignal],
    ) -> None:
        self._finalized_rounds.add(round_idx)
        self._round_signals.pop(round_idx, None)
        logger.info("Central checker declaring global convergence at round %d", round_idx)

        metadata = {
            "type": "global_convergence",
            "round": round_idx,
            "declared_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "clusters": {
                cluster_id: {
                    "converged": signal.converged,
                    "delta_norm": signal.delta_norm,
                    "received_at": datetime.utcfromtimestamp(signal.received_at).isoformat(timespec="seconds") + "Z",
                }
                for cluster_id, signal in signals.items()
            },
        }

        data_id: Optional[str] = None
        if self.blockchain is not None:
            data_id = publish_global_convergence(self.blockchain, round_idx, metadata)
        if self.on_announcement:
            self.on_announcement(round_idx, data_id, metadata)
