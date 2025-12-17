"""Central checker aggregates convergence signals and commits decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from secure_aggregation.convergence.central_broadcast import (
    fetch_global_convergence_round,
    publish_global_convergence,
)
from secure_aggregation.storage.model_store import BlockchainInterface
from secure_aggregation.utils import get_logger

logger = get_logger("central_checker")


@dataclass
class CentralChecker:
    """Tracks convergence signals and declares global convergence."""

    blockchain: Optional[BlockchainInterface]
    total_cliques: int
    cluster_ids: Sequence[str]
    _round_signals: Dict[int, Dict[str, bool]] = field(default_factory=dict)
    _finalized_rounds: set[int] = field(default_factory=set)

    def record_signal(self, cluster_id: str, round_idx: int, converged: bool) -> None:
        if round_idx in self._finalized_rounds:
            return
        signals = self._round_signals.setdefault(round_idx, {})
        prev = signals.get(cluster_id)
        if prev is not None and prev == converged:
            return
        signals[cluster_id] = converged
        logger.info(
            f"Central checker received signal: cluster={cluster_id}, round={round_idx}, converged={converged}"
        )
        if self._has_all_signals(signals):
            if all(signals.values()):
                self._finalize(round_idx)
            else:
                # Not globally converged; keep waiting for next round.
                self._round_signals.pop(round_idx, None)

    def _has_all_signals(self, signals: Dict[str, bool]) -> bool:
        if self.cluster_ids:
            return all(cluster_id in signals for cluster_id in self.cluster_ids)
        return len(signals) >= self.total_cliques

    def _finalize(self, round_idx: int) -> None:
        self._finalized_rounds.add(round_idx)
        self._round_signals.pop(round_idx, None)
        logger.info(f"Central checker declaring global convergence at round {round_idx}")
        if self.blockchain is not None:
            existing = fetch_global_convergence_round(self.blockchain)
            if existing is None or existing != round_idx:
                publish_global_convergence(self.blockchain, round_idx)
