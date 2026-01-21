"""Configuration helpers for hierarchical (state-level) aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


class StateAggregationApproach(str, Enum):
    """Enumerates how candidates for the state aggregator are selected."""

    RING_STAR = "ring_star"
    CUSTOM = "custom"


@dataclass
class StateAggregationConfig:
    """
    Runtime configuration for state-level aggregation rounds.

    Attributes:
        enabled: Whether the state layer is active.
        rounds_per_state: Number of clique rounds before a state round fires.
        approach: Candidate selection approach (ring_star promotes central nodes).
        state_id: Identifier used when anchoring state models on-chain.
        collection_timeout_seconds: Max time to wait for ECM coverage.
        digest_timeout_seconds: How long to wait for peer digests per round.
        consensus_timeout_seconds: Overall timeout for digest alignment.
        commit_timeout_seconds: Per-candidate wait before trying to commit.
    """

    enabled: bool = False
    rounds_per_state: int = 0
    approach: StateAggregationApproach = StateAggregationApproach.RING_STAR
    state_id: str = "state_0"
    collection_timeout_seconds: float = 15.0
    digest_timeout_seconds: float = 5.0
    consensus_timeout_seconds: float = 30.0
    commit_timeout_seconds: float = 10.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "StateAggregationConfig":
        """Create a config instance from a mapping."""
        if not data:
            return cls()
        kwargs: dict[str, Any] = {}
        for key in (
            "enabled",
            "rounds_per_state",
            "cluster_rounds",
            "approach",
            "state_id",
            "collection_timeout_seconds",
            "digest_timeout_seconds",
            "consensus_timeout_seconds",
            "commit_timeout_seconds",
        ):
            if key not in data:
                continue
            value = data[key]
            if key == "approach":
                kwargs[key] = StateAggregationApproach(str(value))
            elif key == "enabled":
                kwargs[key] = bool(value)
            elif key in ("rounds_per_state", "cluster_rounds"):
                # Accept both legacy and new field names.
                kwargs["rounds_per_state"] = max(0, int(value))
            elif key.endswith("_seconds"):
                kwargs[key] = max(0.0, float(value))
            else:
                kwargs[key] = str(value)
        return cls(**kwargs)

    def apply_training_defaults(self, rounds_hint: Optional[int]) -> None:
        """
        Derive missing values from the training configuration.

        Args:
            rounds_hint: Value taken from the training config (if any).
        """
        if self.rounds_per_state <= 0 and rounds_hint:
            self.rounds_per_state = max(1, int(rounds_hint))
        if self.rounds_per_state > 0 and not self.enabled:
            self.enabled = True


@dataclass
class NationAggregationConfig:
    """Configuration for scheduling nation-level rounds (built atop state rounds)."""

    enabled: bool = False
    rounds_per_nation: int = 0
    nation_id: str = "nation_0"

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "NationAggregationConfig":
        if not data:
            return cls()
        kwargs: dict[str, Any] = {}
        for key in ("enabled", "rounds_per_nation", "state_rounds", "nation_id"):
            if key not in data:
                continue
            value = data[key]
            if key == "enabled":
                kwargs[key] = bool(value)
            elif key in ("rounds_per_nation", "state_rounds"):
                # Accept new schema naming.
                kwargs["rounds_per_nation"] = max(0, int(value))
            else:
                kwargs[key] = str(value)
        return cls(**kwargs)
