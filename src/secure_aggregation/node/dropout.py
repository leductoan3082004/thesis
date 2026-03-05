from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Sequence


class DropoutStage(Enum):
    """Points within the SAP flow where a node exits."""

    BEFORE_ROUND0 = "before_round0"
    BEFORE_MASKED_INPUT = "before_masked_input"


@dataclass(frozen=True)
class DropoutPlan:
    """Deterministic per-round mapping of node IDs to dropout stages."""

    round_index: int
    stage_map: Dict[str, DropoutStage]

    def stage_for(self, node_id: str) -> Optional[DropoutStage]:
        return self.stage_map.get(node_id)


class DropoutManager:
    """Generates reproducible dropout schedules across all nodes."""

    def __init__(self, participants: Sequence[str], per_round: int, seed: int = 0) -> None:
        self.participants = sorted({p for p in participants})
        self.per_round = max(0, per_round)
        self.seed = int(seed)
        self._cache_round = -1
        self._cached_plan: Optional[DropoutPlan] = None

    @property
    def enabled(self) -> bool:
        return self.per_round > 0 and bool(self.participants)

    def plan_for_round(self, round_index: int) -> DropoutPlan:
        if not self.enabled:
            return DropoutPlan(round_index=round_index, stage_map={})
        if self._cached_plan and self._cached_plan.round_index == round_index:
            return self._cached_plan
        plan = DropoutPlan(round_index=round_index, stage_map=self._generate_plan(round_index))
        self._cached_plan = plan
        return plan

    def stage_for(self, node_id: str, round_index: int) -> Optional[DropoutStage]:
        plan = self.plan_for_round(round_index)
        return plan.stage_for(node_id)

    def _generate_plan(self, round_index: int) -> Dict[str, DropoutStage]:
        selections = min(self.per_round, len(self.participants))
        if selections <= 0:
            return {}
        rng = random.Random(self._seed_for_round(round_index))
        candidates = list(self.participants)
        rng.shuffle(candidates)
        chosen = candidates[:selections]
        stages = {}
        stage_choices = list(DropoutStage)
        for node_id in chosen:
            stages[node_id] = rng.choice(stage_choices)
        return stages

    def _seed_for_round(self, round_index: int) -> int:
        payload = f"{self.seed}:{round_index}:{len(self.participants)}".encode()
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big")


__all__ = ["DropoutManager", "DropoutPlan", "DropoutStage"]
