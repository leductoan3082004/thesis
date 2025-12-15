"""Coordinator utilities for selecting central checkers and trackers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from secure_aggregation.convergence.central_broadcast import (
    CheckerHealth,
    CentralMetadata,
    fetch_checker_health,
    publish_checker_health,
)
from secure_aggregation.storage.model_store import BlockchainInterface


@dataclass
class CentralCheckerCoordinator:
    """Keeps track of central checker metadata and health announcements."""

    blockchain: Optional[BlockchainInterface]
    metadata: Optional[CentralMetadata] = None

    def update_metadata(self, metadata: Optional[CentralMetadata]) -> None:
        self.metadata = metadata

    def announce_health(self, checker_id: str, round_idx: int, priority: int, alive: bool = True) -> None:
        if self.blockchain is None:
            return
        health = CheckerHealth(
            checker_id=checker_id,
            round_idx=round_idx,
            priority=priority,
            alive=alive,
        )
        publish_checker_health(self.blockchain, health)

    def select_active_checker(self, round_idx: int) -> Optional[str]:
        """Select the checker with the highest priority that is alive."""
        if self.metadata is None or not self.metadata.checker_candidates:
            return None
        candidates = self.metadata.checker_candidates
        # Simple round-robin over candidates based on round index.
        preferred = candidates[round_idx % len(candidates)]
        if self.blockchain is None:
            return preferred

        health_records = fetch_checker_health(self.blockchain)
        if not health_records:
            return preferred
        # Choose the most recent alive checker with the highest priority value.
        alive_records = [h for h in health_records if h.alive]
        if not alive_records:
            return preferred
        best = max(alive_records, key=lambda h: h.priority)
        return best.checker_id
