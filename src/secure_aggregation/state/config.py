"""Configuration helpers for hierarchical aggregation levels."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class StateAggregationApproach(str, Enum):
    """Enumerates how candidates for a hierarchy aggregator are selected."""

    RING_STAR = "ring_star"
    CUSTOM = "custom"


@dataclass
class HierarchyLevelConfig:
    """
    Runtime configuration for a hierarchy level (state, nation, etc.).

    Attributes:
        enabled: Whether this level participates in the hierarchy.
        scope_index: Ordering above the cluster layer (1=closest to clusters).
        scope_name: Human-readable name (used in logs and bridge channels).
        scope_id: Identifier used when anchoring models on-chain.
        rounds_per_scope: Number of lower-level rounds before this level fires.
        interval_seconds: Wall-clock interval between rounds (time-based scheduler).
        wait_seconds: Optional pause nodes should honor before pulling this scope's model.
        approach: Candidate election approach for collection-enabled levels.
        collection_timeout_seconds: Max time to wait for ECM coverage.
        digest_timeout_seconds: Legacy knob retained for backward compatibility (no longer used).
        consensus_timeout_seconds: Legacy knob retained for backward compatibility (no longer used).
        commit_timeout_seconds: Per-candidate wait before trying to commit.
        apply_policy: Policy for assimilating upstream checkpoints.
        apply_alpha: Interpolation weight when apply_policy="interpolate".
        apply_layer_mask: Optional layer masks for selective application.
        max_aggregators: Optional bound on concurrent aggregators.
        fanout_per_group: Optional fan-out hint for multi-hop topologies.
        fanout_per_scope: Optional fan-out hint for higher scopes.
        fanout_count: Optional number of fan-out nodes per child scope.
    """

    enabled: bool = False
    scope_index: int = 1
    scope_name: str = "state"
    scope_id: str = "scope_0"
    rounds_per_scope: int = 0
    interval_seconds: float = 0.0
    wait_seconds: float = 0.0
    approach: StateAggregationApproach = StateAggregationApproach.RING_STAR
    collection_timeout_seconds: float = 15.0
    digest_timeout_seconds: float = 5.0
    consensus_timeout_seconds: float = 30.0
    commit_timeout_seconds: float = 10.0
    apply_policy: str = "replace"
    apply_alpha: float = 0.0
    apply_layer_mask: list[str] = field(default_factory=list)
    max_aggregators: Optional[int] = None
    fanout_per_group: Optional[int] = None
    fanout_per_scope: Optional[int] = None
    fanout_count: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "HierarchyLevelConfig":
        """Create a config instance from a mapping."""
        if not data:
            return cls()
        aliases = {
            "state_id": "scope_id",
            "rounds_per_state": "rounds_per_scope",
            "cluster_rounds": "rounds_per_scope",
            "rounds_per_nation": "rounds_per_scope",
            "state_rounds": "rounds_per_scope",
            "rounds_per_parent": "rounds_per_scope",
            "fanout_per_state": "fanout_per_scope",
        }
        kwargs: dict[str, Any] = {}
        for raw_key, value in data.items():
            key = aliases.get(raw_key, raw_key)
            if key == "enabled":
                kwargs[key] = bool(value)
            elif key == "scope_index":
                kwargs[key] = max(1, int(value))
            elif key == "rounds_per_scope":
                kwargs[key] = max(0, int(value))
            elif key == "approach":
                kwargs[key] = StateAggregationApproach(str(value))
            elif key.endswith("_seconds"):
                kwargs[key] = max(0.0, float(value))
            elif key == "apply_alpha":
                kwargs[key] = float(value)
            elif key == "apply_layer_mask":
                if isinstance(value, (list, tuple)):
                    kwargs[key] = [str(v) for v in value]
                else:
                    kwargs[key] = [str(value)]
            elif key in ("max_aggregators", "fanout_per_group", "fanout_per_scope", "fanout_count"):
                kwargs[key] = int(value) if value is not None else None
            else:
                kwargs[key] = str(value)
        return cls(**kwargs)

    def apply_training_defaults(self, rounds_hint: Optional[int]) -> None:
        """
        Derive missing values from the training configuration.

        Args:
            rounds_hint: Value taken from the training config (if any).
        """
        if self.rounds_per_scope <= 0 and rounds_hint:
            self.rounds_per_scope = max(1, int(rounds_hint))
        has_time_interval = self.interval_seconds > 0
        if (self.rounds_per_scope > 0 or has_time_interval) and not self.enabled:
            self.enabled = True

    @property
    def collects_lower_scope(self) -> bool:
        """Return True if this level runs collection/aggregation logic."""
        return bool(self.collection_timeout_seconds > 0)
