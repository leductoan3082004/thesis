"""Hierarchy-specific helpers for node_service."""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from secure_aggregation.node.ecm_buffer import ECM, ECMBuffer
from secure_aggregation.state import (
    HierarchyLevelConfig,
    StateAggregationApproach,
    StateAggregationError,
    StateAggregator,
    StateClusterModel,
)
from secure_aggregation.state.nodes_map import NodesMapMetadata, load_nodes_map
from secure_aggregation.storage.model_store import AnchorScope, ModelAnchor, compute_model_hash, verify_model_hash
from secure_aggregation.utils import get_logger

hierarchy_logger = get_logger("hierarchy")

HierarchyScopeConfig = HierarchyLevelConfig


@dataclass
class ScopeRoundHandler:
    """Describes how to run aggregation rounds for a specific scope."""

    scope_name: str
    config: HierarchyLevelConfig
    trigger_label: str
    round_queue: Deque[Tuple[int, int]]
    rounds_logged: Set[int]
    round_cache: Dict[int, Any]
    round_hashes: Dict[int, str]
    committed_rounds: Set[int]
    is_candidate_fn: Callable[[], bool]
    dispatch_fn: Callable[[int, int], None]
    execute_fn: Callable[[int, int], bool]
    budget_fn: Optional[Callable[[], Optional[int]]] = None


@dataclass
class ScopeRuntime:
    """Mutable state that tracks per-scope aggregation context."""

    scope_name: str
    config: HierarchyLevelConfig
    scope_id: Optional[str]
    round_queue: Deque[Tuple[int, int]]
    rounds_logged: Set[int]
    round_cache: Dict[int, Any]
    round_hashes: Dict[int, str]
    committed_rounds: Set[int]
    candidates: List[str]
    is_candidate: bool
    ecm_buffer: Optional[ECMBuffer]
    aggregator: Optional[StateAggregator]
    last_model_cid: Optional[str]
    last_model_hash: Optional[str]
    last_model_data_id: Optional[str]
    pending_anchor: Optional[ModelAnchor] = None


class RoundRobinPool:
    """Utility to iterate over members in round-robin order."""

    def __init__(self, members: Optional[Sequence[str]] = None) -> None:
        unique: List[str] = []
        seen: Set[str] = set()
        for member in members or []:
            normalized = str(member).strip()
            if normalized and normalized not in seen:
                unique.append(normalized)
                seen.add(normalized)
        self._members: List[str] = unique
        self._cursor: int = 0

    def next(self, count: int = 1) -> List[str]:
        """Return the next `count` members from the pool."""
        if not self._members or count <= 0:
            return []
        selection: List[str] = []
        for _ in range(count):
            selection.append(self._members[self._cursor])
            self._cursor = (self._cursor + 1) % len(self._members)
        return selection

    def all_members(self) -> List[str]:
        return list(self._members)


class HierarchyMixin:
    """Mixin providing hierarchy (state/nation/...) orchestration helpers."""

    def _scope_name(self) -> str:
        config = getattr(self, "scope_config", None)
        name = getattr(config, "scope_name", "") if config else ""
        return str(name or "state")

    def _scope_label_lower(self) -> str:
        return self._scope_name().lower()

    def _scope_label_upper(self) -> str:
        return self._scope_name().upper()

    def _scope_identifier(self) -> Optional[str]:
        config = getattr(self, "scope_config", None)
        if not config:
            return None
        scope_id = getattr(config, "scope_id", None)
        if scope_id:
            return str(scope_id)
        legacy_id = getattr(config, "state_id", None)
        return str(legacy_id) if legacy_id else None

    def _scope_namespace(self, config: Optional[HierarchyLevelConfig] = None) -> str:
        cfg = config or getattr(self, "scope_config", None)
        name = getattr(cfg, "scope_name", "") if cfg else ""
        return (name or "state").lower()

    def _higher_scope_label_lower(self) -> str:
        config = getattr(self, "higher_scope_config", None)
        name = getattr(config, "scope_name", "") if config else ""
        return str(name or "higher").lower()

    def _higher_scope_label_upper(self) -> str:
        return self._higher_scope_label_lower().upper()

    def _register_scope_round_handler(self, handler: ScopeRoundHandler) -> None:
        """Register handler used to run aggregation rounds for a given scope."""
        registry = getattr(self, "_scope_round_handlers", None)
        if registry is None:
            registry = {}
            self._scope_round_handlers = registry
        registry[handler.scope_name.lower()] = handler

    def _get_scope_round_handler(self, scope_name: Optional[str] = None) -> Optional[ScopeRoundHandler]:
        """Fetch the handler used for orchestrating the supplied scope name."""
        registry = getattr(self, "_scope_round_handlers", None) or {}
        key = str(scope_name or self.scope_name or "state").lower()
        return registry.get(key)

    def _load_scope_config(self) -> OrderedDict[str, HierarchyScopeConfig]:
        """Load hierarchy level configurations with shared defaults."""
        system_cfg = self.system_config or {}
        defaults = dict(system_cfg.get("hierarchy_defaults") or {})
        level_data = system_cfg.get("hierarchy_levels")
        level_configs: List[HierarchyScopeConfig] = []
        if isinstance(level_data, list) and level_data:
            for entry in level_data:
                merged = dict(defaults)
                if isinstance(entry, Mapping):
                    merged.update(entry or {})
                config = HierarchyLevelConfig.from_mapping(merged)
                level_configs.append(config)
        else:
            # Fallback to legacy *_aggregation schema if hierarchy_levels is absent.
            sections: List[Dict[str, Any]] = []
            for key, value in system_cfg.items():
                if not key.endswith("_aggregation"):
                    continue
                merged = dict(defaults)
                try:
                    merged.update(value or {})
                except AttributeError:
                    pass
                merged.setdefault("scope_name", key.replace("_aggregation", ""))
                merged.setdefault("scope_index", 1 if key.startswith("state") else 2)
                sections.append(merged)
            if not sections:
                sections.append(dict(defaults))
            for section in sections:
                config = HierarchyLevelConfig.from_mapping(section)
                level_configs.append(config)
        if not level_configs:
            level_configs.append(HierarchyLevelConfig.from_mapping(defaults))
        node_scope_id = getattr(self, "state_id", None)
        ordered_levels: List[HierarchyScopeConfig] = []
        for cfg in level_configs:
            if cfg.scope_index == 1 and node_scope_id:
                cfg.scope_id = str(node_scope_id)
            if not cfg.scope_name.strip():
                cfg.scope_name = f"scope_{cfg.scope_index}"
            cfg.apply_training_defaults(None)
            ordered_levels.append(cfg)
        ordered_levels.sort(key=lambda cfg: (cfg.scope_index, cfg.scope_name))
        scope_configs: "OrderedDict[str, HierarchyScopeConfig]" = OrderedDict(
            (cfg.scope_name.lower(), cfg) for cfg in ordered_levels
        )
        return scope_configs

    def _resolve_nodes_map_path(self) -> Optional[Path]:
        """Determine path to the hierarchical nodes map, if provided."""
        system_cfg = getattr(self, "system_config", None) or {}
        explicit = (
            system_cfg.get("nodes_map_path")
            or system_cfg.get("state_map_path")
            or os.getenv("NODES_MAP_PATH")
            or os.getenv("STATE_MAP_PATH")
        )
        path: Path
        if explicit:
            path = Path(str(explicit))
        else:
            sys_path = getattr(self, "system_config_path", None)
            if sys_path:
                path = Path(sys_path).with_name("nodes-map.json")
            else:
                path = Path("config/nodes-map.json")
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _resolve_topology_path(self) -> Optional[Path]:
        """Resolve the topology description path used for clique membership."""
        config = getattr(self, "inter_cluster_config", None) or {}
        topo = config.get("topology_file") or os.getenv("TOPOLOGY_FILE")
        if topo:
            path = Path(str(topo))
        else:
            path = Path("config/topology.json")
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _scope_order_for_nodes_map(self) -> List[str]:
        """Return scope names ordered from highest to lowest index."""
        configs = getattr(self, "scope_configs", None)
        if not configs:
            configs = self._load_scope_config()
            self.scope_configs = configs
        ordered = sorted((cfg for cfg in configs.values() if cfg.scope_name), key=lambda cfg: cfg.scope_index, reverse=True)
        return [cfg.scope_name.lower() for cfg in ordered]

    def _load_nodes_map_metadata(self) -> NodesMapMetadata:
        """Load the hierarchical nodes map once and cache it."""
        cached = getattr(self, "_nodes_map_metadata", None)
        if cached is not None:
            return cached
        scope_order = self._scope_order_for_nodes_map()
        path = self._resolve_nodes_map_path()
        if not scope_order or not path:
            metadata = NodesMapMetadata(rosters={}, child_map={}, memberships={})
        else:
            try:
                metadata = load_nodes_map(path, scope_order)
            except ValueError as exc:
                hierarchy_logger.warning("Failed to load nodes map from %s: %s", path, exc)
                metadata = NodesMapMetadata(rosters={}, child_map={}, memberships={})
        setattr(self, "_nodes_map_metadata", metadata)
        return metadata

    def _scope_rosters(self, scope_name: Optional[str] = None) -> Mapping[str, List[str]]:
        metadata = self._load_nodes_map_metadata()
        if not scope_name:
            return metadata.rosters
        return metadata.rosters.get(str(scope_name).lower(), {})

    def _scope_children(self, scope_name: str, scope_id: str) -> Mapping[str, List[str]]:
        metadata = self._load_nodes_map_metadata()
        key = (str(scope_name).lower(), str(scope_id))
        return metadata.child_map.get(key, {})

    def _node_scope_memberships(self) -> Mapping[str, Dict[str, str]]:
        metadata = self._load_nodes_map_metadata()
        return metadata.memberships

    def _scope_name_order(self) -> List[str]:
        configs = getattr(self, "scope_configs", None) or {}
        ordered = sorted(configs.values(), key=lambda cfg: (cfg.scope_index, cfg.scope_name))
        return [cfg.scope_name.lower() for cfg in ordered if cfg.scope_name]

    def _lower_scope_name(self, scope_name: str) -> Optional[str]:
        ordered = self._scope_name_order()
        key = str(scope_name or "").lower()
        if key not in ordered:
            return None
        idx = ordered.index(key)
        if idx == 0:
            return "cluster"
        return ordered[idx - 1]

    def _node_scope_membership_map(self) -> Mapping[str, str]:
        """Return cached membership path for this node, keyed by scope name."""
        node_id = getattr(self, "node_id", None)
        if not node_id:
            return {}
        memberships = self._node_scope_memberships()
        mapping = memberships.get(node_id)
        if not mapping:
            mapping = {}
        scope_map = {name.lower(): scope_id for name, scope_id in mapping.items() if scope_id}
        if "cluster" not in scope_map:
            clique_id = getattr(self, "clique_id", None)
            if clique_id is not None and clique_id >= 0:
                scope_map["cluster"] = f"cluster_{clique_id}"
        return scope_map

    def _runtime_for_scope(self, scope_name: Optional[str] = None) -> ScopeRuntime:
        runtimes = getattr(self, "_scope_runtimes", None) or {}
        key = str(scope_name or self.scope_name or "state").lower()
        runtime = runtimes.get(key)
        if runtime is None:
            raise KeyError(f"No runtime registered for scope '{key}'")
        return runtime

    def _runtime_for_scope_label(self, scope_label: Optional[str]) -> ScopeRuntime:
        label = str(scope_label or self.scope_name or "state").lower()
        configs = getattr(self, "scope_configs", None) or {}
        config = configs.get(label)
        if not config:
            return self.scope_runtime
        return self._ensure_scope_runtime(config.scope_name, config)

    def _emit_scope_round_leader_metric(
        self,
        runtime: ScopeRuntime,
        scope_round: int,
        leader_id: Optional[str],
    ) -> None:
        if not leader_id:
            return
        prom = getattr(self, "prom_metrics", None)
        if prom is None:
            return
        scope_id = runtime.scope_id or getattr(runtime.config, "scope_id", None)
        if not scope_id:
            return
        prom.set_scope_round_leader(
            runtime.scope_name,
            scope_id,
            scope_round,
            leader_id,
        )

    def _record_scope_commit_metric(
        self,
        runtime: ScopeRuntime,
        scope_round: int,
        committer_node_id: Optional[str],
    ) -> None:
        if not committer_node_id:
            return
        prom = getattr(self, "prom_metrics", None)
        if prom is None:
            return
        scope_id = runtime.scope_id or getattr(runtime.config, "scope_id", None)
        if not scope_id:
            return
        prom.record_scope_round_commit(
            runtime.scope_name,
            scope_id,
            scope_round,
            committer_node_id,
        )

    @staticmethod
    def _runtime_label_lower(runtime: ScopeRuntime) -> str:
        name = runtime.config.scope_name or runtime.scope_name or "scope"
        return str(name).lower()

    @staticmethod
    def _runtime_label_upper(runtime: ScopeRuntime) -> str:
        return HierarchyMixin._runtime_label_lower(runtime).upper()

    def _runtime_scope_identifier(self, runtime: ScopeRuntime) -> Optional[str]:
        return self._node_scope_identifier_for(runtime.scope_name, runtime.config)

    def _ensure_scope_runtime(self, scope_name: str, config: HierarchyLevelConfig) -> ScopeRuntime:
        runtimes = getattr(self, "_scope_runtimes", None)
        if runtimes is None:
            runtimes = {}
            self._scope_runtimes = runtimes
        key = str(scope_name or "").lower()
        runtime = runtimes.get(key)
        if runtime is None:
            scope_id = self._node_scope_identifier_for(scope_name, config)
            runtime = ScopeRuntime(
                scope_name=scope_name,
                config=config,
                scope_id=scope_id,
                round_queue=deque(),
                rounds_logged=set(),
                round_cache={},
                round_hashes={},
                committed_rounds=set(),
                candidates=[],
                is_candidate=False,
                ecm_buffer=None,
                aggregator=None,
                last_model_cid=None,
                last_model_hash=None,
                last_model_data_id=None,
            )
            runtimes[key] = runtime
        return runtime

    def _iter_scope_runtimes(self) -> List[ScopeRuntime]:
        runtimes = getattr(self, "_scope_runtimes", None) or {}
        return list(runtimes.values())

    @property
    def scope_runtime(self) -> ScopeRuntime:
        return self._runtime_for_scope(self.scope_name)

    @property
    def scope_candidates(self) -> List[str]:
        return self.scope_runtime.candidates

    @scope_candidates.setter
    def scope_candidates(self, value: List[str]) -> None:
        self.scope_runtime.candidates = list(value or [])

    @property
    def is_scope_candidate(self) -> bool:
        return self.scope_runtime.is_candidate

    @is_scope_candidate.setter
    def is_scope_candidate(self, value: bool) -> None:
        self.scope_runtime.is_candidate = bool(value)

    @property
    def scope_ecm_buffer(self) -> Optional[ECMBuffer]:
        return self.scope_runtime.ecm_buffer

    @scope_ecm_buffer.setter
    def scope_ecm_buffer(self, buffer: Optional[ECMBuffer]) -> None:
        self.scope_runtime.ecm_buffer = buffer

    @property
    def scope_aggregator(self) -> Optional[StateAggregator]:
        return self.scope_runtime.aggregator

    @scope_aggregator.setter
    def scope_aggregator(self, aggregator: Optional[StateAggregator]) -> None:
        self.scope_runtime.aggregator = aggregator

    def _node_scope_identifier_for(self, scope_name: str, config: Optional[HierarchyLevelConfig]) -> Optional[str]:
        """Resolve this node's scope identifier for the supplied level."""
        normalized = str(scope_name or "").lower()
        membership = self._node_scope_membership_map()
        scope_id = membership.get(normalized)
        if scope_id:
            return scope_id
        if config and getattr(config, "scope_id", None):
            return str(config.scope_id)
        if normalized == self.scope_name.lower():
            return self._scope_identifier()
        return None

    def _node_participates_in_scope(self, scope_name: str, config: Optional[HierarchyLevelConfig]) -> bool:
        """Return True if this node belongs to the supplied scope."""
        scope_id = self._node_scope_identifier_for(scope_name, config)
        return bool(scope_id)

    def _scope_member_roster(self, scope_name: str, scope_id: Optional[str]) -> List[str]:
        rosters = self._scope_rosters(scope_name) or {}
        key = str(scope_id) if scope_id is not None else None
        if key:
            members = rosters.get(key)
            if members:
                return list(members)
        scope_key = str(scope_name or "").lower()
        if scope_key == "cluster" and key == f"cluster_{getattr(self, 'clique_id', -1)}":
            clique_members = getattr(self, "clique_members", None) or []
            if clique_members:
                return list(clique_members)
        return []

    @staticmethod
    def _round_robin_subset(members: Sequence[str], count: int, seed: int) -> List[str]:
        seen: Set[str] = set()
        unique: List[str] = []
        for member in members:
            normalized = str(member).strip()
            if not normalized or normalized in seen:
                continue
            unique.append(normalized)
            seen.add(normalized)
        if not unique or count <= 0:
            return []
        total = len(unique)
        if count >= total:
            return list(unique)
        seed = max(0, seed - 1)
        start = (seed * max(1, count)) % total
        selection: List[str] = []
        for offset in range(count):
            selection.append(unique[(start + offset) % total])
        return selection

    def _scope_round_leader(
        self,
        scope_name: str,
        scope_id: Optional[str],
        scope_round: int,
        *,
        fallback: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        members = self._scope_member_roster(scope_name, scope_id)
        if not members and fallback:
            members = list(fallback)
        if not members:
            return None
        index = (max(0, scope_round - 1)) % len(members)
        return members[index]

    def _ordered_scope_candidates(self, runtime: ScopeRuntime, scope_round: int) -> List[str]:
        roster = self._scope_member_roster(runtime.scope_name, runtime.scope_id)
        if not roster:
            roster = list(runtime.candidates or [])
        if not roster:
            return []
        leader_index = (max(0, scope_round - 1)) % len(roster)
        return [roster[(leader_index + offset) % len(roster)] for offset in range(len(roster))]

    def _select_healthy_scope_aggregator(
        self,
        runtime: ScopeRuntime,
        scope_round: int,
        timeout: float = 3.0,
    ) -> Optional[Tuple[str, Optional[str]]]:
        """Return the first aggregator whose bridge endpoint is reachable."""
        ordered = self._ordered_scope_candidates(runtime, scope_round)
        if not ordered:
            return None
        label_lower = self._runtime_label_lower(runtime)
        for candidate in ordered:
            if candidate == self.node_id:
                return candidate, None
            address = self._resolve_scope_bridge_address(candidate)
            if not address:
                hierarchy_logger.warning(
                    "Missing bridge address for %s aggregator %s; trying next candidate",
                    label_lower,
                    candidate,
                )
                continue
            if not self._ensure_bridge_client(allow_state_layer=True) or self.bridge_client is None:
                return None
            if self.bridge_client.wait_for_ready(address, timeout=timeout):
                return candidate, address
            hierarchy_logger.warning(
                "%s aggregator %s unreachable; trying next candidate",
                label_lower,
                candidate,
            )
        return None

    def _scope_round_fanout_members(
        self,
        child_scope: str,
        child_scope_id: Optional[str],
        scope_round: int,
        per_child_count: int,
    ) -> List[str]:
        roster = self._scope_member_roster(child_scope, child_scope_id)
        return self._round_robin_subset(roster, per_child_count, scope_round)

    def _resolve_scope_bridge_address(self, node_id: str) -> Optional[str]:
        if not node_id:
            return None
        bridge_map = getattr(self, "central_neighbor_addresses", None) or {}
        address = bridge_map.get(node_id)
        if address:
            return address
        participant_map = getattr(self, "participant_map", None) or {}
        base_address = participant_map.get(node_id)
        if not base_address:
            return None
        try:
            host, port_str = base_address.split(":")
            bridge_port = int(port_str) + 2000
        except ValueError:
            return None
        return f"{host}:{bridge_port}"

    def _is_fanout_node_for_scope_round(
        self,
        runtime: ScopeRuntime,
        scope_round: int,
    ) -> bool:
        config = runtime.config
        configured = getattr(config, "fanout_count", None)
        if configured is None:
            configured = getattr(config, "fanout_per_scope", None)
        if configured is None:
            configured = getattr(config, "fanout_per_group", None)
        try:
            fanout_count = int(configured) if configured is not None else 0
        except (TypeError, ValueError):
            fanout_count = 0
        if fanout_count <= 0:
            return True
        memberships = self._node_scope_membership_map()
        child_scope = self._lower_scope_name(runtime.scope_name)
        if not child_scope:
            return False
        if child_scope == "cluster":
            child_scope_id = f"cluster_{self.clique_id}"
        else:
            child_scope_id = memberships.get(child_scope)
        if not child_scope_id:
            return False
        fanout_members = self._scope_round_fanout_members(
            child_scope,
            child_scope_id,
            scope_round,
            fanout_count,
        )
        return self.node_id in fanout_members

    def _init_scope_role_pools(self) -> None:
        """Initialize round-robin pools for aggregators and fan-out members."""
        metadata = self._load_nodes_map_metadata()
        aggregator_pools: Dict[Tuple[str, str], RoundRobinPool] = {}
        fanout_pools: Dict[Tuple[str, str], RoundRobinPool] = {}
        for scope_name, scope_map in metadata.rosters.items():
            for scope_id, nodes in scope_map.items():
                if not nodes:
                    continue
                key = (scope_name, scope_id)
                aggregator_pools[key] = RoundRobinPool(nodes)
                fanout_pools[key] = RoundRobinPool(nodes)
        self._aggregator_pools = aggregator_pools
        self._fanout_pools = fanout_pools

    def _next_scope_aggregators(self, scope_name: str, scope_id: str, count: int = 1) -> List[str]:
        """Return the next aggregators for the supplied scope using round-robin ordering."""
        pools = getattr(self, "_aggregator_pools", None) or {}
        pool = pools.get((scope_name.lower(), str(scope_id)))
        if not pool:
            return []
        return pool.next(max(1, count))

    def _scope_interval_seconds(self, config: Optional[HierarchyLevelConfig]) -> float:
        if not config:
            return 0.0
        try:
            return float(getattr(config, "interval_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _init_scope_timers(self) -> None:
        """Initialize per-scope round counters and timer deadlines."""
        now = time.time()
        counters: Dict[str, int] = {}
        next_fire: Dict[str, float] = {}
        for name, cfg in (getattr(self, "scope_configs", None) or {}).items():
            counters[name] = 0
            interval = self._scope_interval_seconds(cfg)
            if interval > 0:
                next_fire[name] = now + interval
        self._scope_round_counters = counters
        self._scope_next_fire_at = next_fire

    def _increment_scope_round(self, scope_name: str, fallback: Optional[int] = None) -> int:
        counters = getattr(self, "_scope_round_counters", None)
        if counters is None:
            counters = {}
            self._scope_round_counters = counters
        current = counters.get(scope_name, 0)
        if fallback is not None and fallback > current:
            counters[scope_name] = fallback
            return fallback
        counters[scope_name] = current + 1
        return counters[scope_name]

    def _next_scope_fanout_members(self, scope_name: str, scope_id: str, count: int = 1) -> List[str]:
        """Return fan-out nodes for the supplied scope using round-robin ordering."""
        pools = getattr(self, "_fanout_pools", None) or {}
        pool = pools.get((scope_name.lower(), str(scope_id)))
        if not pool:
            return []
        return pool.next(max(1, count))

    def _queue_scope_wait(self, scope_name: str, config: HierarchyLevelConfig) -> None:
        """Record a wait window so nodes pause before fetching the next scope model."""
        if not self._node_participates_in_scope(scope_name, config):
            return
        wait_seconds = float(getattr(config, "wait_seconds", 0.0) or 0.0)
        scope_key = str(scope_name or "").lower()
        ready = getattr(self, "_ready_scope_fetches", None)
        if ready is None:
            ready = set()
            self._ready_scope_fetches = ready
        if wait_seconds <= 0:
            ready.add(scope_key)
            hierarchy_logger.info(
                "%s wait window is 0s; immediately queued fetch cycle",
                scope_name.upper(),
            )
            return
        queue = getattr(self, "_pending_scope_waits", None)
        if queue is None:
            queue = deque()
            self._pending_scope_waits = queue
        queue.append((scope_key, wait_seconds, time.time()))
        hierarchy_logger.info(
            "Queued %s wait window of %.0fs after scheduler trigger",
            scope_name.upper(),
            wait_seconds,
        )

    def _scope_wait_key(self, runtime: ScopeRuntime) -> str:
        label = getattr(runtime.config, "scope_name", runtime.scope_name) or runtime.scope_name or "scope"
        return str(label).lower()

    def _scope_applied_key(self, runtime: ScopeRuntime) -> Optional[str]:
        scope_id = runtime.scope_id
        if not scope_id:
            return None
        return f"{self._scope_wait_key(runtime)}::{scope_id}"

    def _mark_scope_fetch_ready(
        self,
        runtime: ScopeRuntime,
        anchor_cid: Optional[str],
        anchor: Optional[ModelAnchor] = None,
    ) -> None:
        scope_key = self._scope_wait_key(runtime)
        applied_key = self._scope_applied_key(runtime)
        last_cids = getattr(self, "_scope_last_applied_cids", None) or {}
        if anchor_cid and applied_key and last_cids.get(applied_key) == anchor_cid:
            return
        ready = getattr(self, "_ready_scope_fetches", None)
        if ready is None:
            ready = set()
            self._ready_scope_fetches = ready
        if scope_key in ready:
            return
        ready.add(scope_key)
        runtime.pending_anchor = anchor
        if anchor_cid:
            hierarchy_logger.info(
                "%s anchor cid=%s... observed; fetching latest model without remaining wait window",
                self._runtime_label_upper(runtime),
                anchor_cid[:8],
            )
        else:
            hierarchy_logger.info(
                "%s wait window cleared; fetching latest model without delay",
                self._runtime_label_upper(runtime),
            )

    def _clear_scope_wait(self, runtime: ScopeRuntime) -> None:
        scope_key = self._scope_wait_key(runtime)
        queue = getattr(self, "_pending_scope_waits", None)
        if not queue:
            return
        retained = deque(entry for entry in queue if entry[0] != scope_key)
        queue.clear()
        queue.extend(retained)

    def _pause_for_scope_waits(self) -> bool:
        """Sleep until queued wait windows elapse before fetching high-level models."""
        queue: Optional[Deque[Tuple[str, float, float]]] = getattr(self, "_pending_scope_waits", None)
        if not queue:
            return False
        ready = getattr(self, "_ready_scope_fetches", None)
        if ready is None:
            ready = set()
            self._ready_scope_fetches = ready
        if ready:
            return False
        waited = False
        while queue:
            scope_key, wait_seconds, scheduled_at = queue.popleft()
            elapsed = max(0.0, time.time() - scheduled_at)
            remaining = max(0.0, wait_seconds - elapsed)
            if remaining > 0:
                waited = True
                hierarchy_logger.info(
                    "Waiting %.0fs before pulling latest %s model",
                    remaining,
                    scope_key.upper(),
                )
                time.sleep(remaining)
            else:
                hierarchy_logger.debug(
                    "%s wait window elapsed upstream (extra %.1fs)",
                    scope_key.upper(),
                    elapsed - wait_seconds,
                )
            ready.add(scope_key)
            if ready:
                break
        return waited

    def _apply_ready_scope_models(self) -> bool:
        """Fetch and merge scope models whose wait windows have elapsed."""
        ready: Optional[Set[str]] = getattr(self, "_ready_scope_fetches", None)
        if not ready:
            return False
        scope_configs = getattr(self, "scope_configs", None) or {}
        pending = list(ready)
        ready.clear()
        attempted = False
        for scope_key in pending:
            config = scope_configs.get(scope_key)
            if not config:
                continue
            attempted = True
            if scope_key == self.scope_name.lower():
                self._maybe_apply_scope_model(None)
            else:
                self._maybe_apply_external_scope_model(scope_key, config)
        return attempted

    def _prime_scope_fetches(self) -> None:
        """Ensure nodes pull the latest models for scopes they belong to on startup."""
        scope_configs = getattr(self, "scope_configs", None) or {}
        ready = getattr(self, "_ready_scope_fetches", None)
        if ready is None:
            ready = set()
            self._ready_scope_fetches = ready
        for scope_key, config in scope_configs.items():
            if self._node_participates_in_scope(scope_key, config):
                ready.add(scope_key)

    @staticmethod
    def _trainer_to_canonical_id(trainer_id: str) -> Optional[str]:
        suffix = "".join(ch for ch in str(trainer_id) if ch.isdigit())
        if not suffix:
            return None
        try:
            index = int(suffix) - 1
        except ValueError:
            return None
        if index < 0:
            return None
        return f"node_{index}"

    def _load_state_metadata(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Load state rosters and derive cluster coverage for each state."""
        scope_rosters = self._scope_rosters("state") or {}
        rosters: Dict[str, List[str]] = {scope_id: list(nodes) for scope_id, nodes in scope_rosters.items()}
        cluster_map: Dict[str, List[str]] = {}
        topo_path = self._resolve_topology_path()
        container_to_clique: Dict[str, int] = {}
        if topo_path and topo_path.exists():
            try:
                topo = json.loads(topo_path.read_text())
                cliques = topo.get("cliques") or []
                for idx, members in enumerate(cliques):
                    if not isinstance(members, list):
                        continue
                    for member in members:
                        member_id = str(member)
                        container_to_clique[member_id] = idx
                        canonical = self._trainer_to_canonical_id(member_id)
                        if canonical:
                            container_to_clique.setdefault(canonical, idx)
            except (OSError, json.JSONDecodeError) as exc:
                hierarchy_logger.warning("Failed to parse topology file %s: %s", topo_path, exc)
        for state_id, nodes in rosters.items():
            seen: Set[str] = set()
            clusters: List[str] = []
            for trainer_id in nodes:
                container = self._trainer_to_canonical_id(trainer_id)
                if not container:
                    continue
                clique_idx = container_to_clique.get(container)
                if clique_idx is None:
                    continue
                cluster_id = f"cluster_{clique_idx}"
                if cluster_id not in seen:
                    seen.add(cluster_id)
                    clusters.append(cluster_id)
            if clusters:
                cluster_map[state_id] = clusters
        return rosters, cluster_map

    def _state_metadata(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        rosters = getattr(self, "_state_rosters", None)
        cluster_map = getattr(self, "_state_cluster_map", None)
        if rosters is None or cluster_map is None:
            rosters, cluster_map = self._load_state_metadata()
            setattr(self, "_state_rosters", rosters)
            setattr(self, "_state_cluster_map", cluster_map)
        return rosters, cluster_map

    def _state_roster_for(self, scope_id: Optional[str] = None) -> List[str]:
        rosters, _ = self._state_metadata()
        key = scope_id or getattr(self, "state_id", None) or getattr(self.scope_config, "scope_id", None)
        if not key:
            return []
        return list(rosters.get(str(key), []))

    def _state_cluster_ids(self, scope_id: Optional[str] = None) -> List[str]:
        key = scope_id or getattr(self, "state_id", None) or getattr(self.scope_config, "scope_id", None)
        if not key:
            return []
        scope_key = str(key)
        metadata = getattr(self, "central_metadata", None)
        if metadata:
            scope_clusters = getattr(metadata, "scope_cluster_ids", None) or {}
            clusters = scope_clusters.get(scope_key)
            if clusters:
                return list(clusters)
        _, cluster_map = self._state_metadata()
        return list(cluster_map.get(scope_key, []))

    def _child_scope_ids_for_runtime(self, runtime: ScopeRuntime) -> List[str]:
        child_scope = self._lower_scope_name(runtime.scope_name)
        if not child_scope:
            return []
        scope_id = runtime.scope_id
        parent_label = str(runtime.scope_name or "").lower()
        if child_scope == "cluster":
            ids = self._state_cluster_ids(scope_id)
        else:
            if not scope_id:
                return []
            children = self._scope_children(runtime.scope_name, scope_id)
            ids = children.get(child_scope, [])
        return [f"{parent_label}::{child_id}" for child_id in (ids or [])]

    def _fanout_payload_for_child_scope(self, child_scope: str) -> Tuple[Optional[str], Optional[str]]:
        if child_scope == "cluster":
            return self._last_model_cid, self._last_model_hash
        try:
            child_runtime = self._runtime_for_scope(child_scope)
        except KeyError:
            return None, None
        return child_runtime.last_model_cid, child_runtime.last_model_hash

    def _preferred_scope_candidates(self, limit: Optional[int] = None) -> List[str]:
        """Return candidate aggregator nodes prioritized by state roster."""
        roster = self._state_roster_for()
        metadata_nodes = []
        if self.central_metadata:
            metadata_nodes = list(dict.fromkeys(self.central_metadata.central_nodes))
        if roster:
            roster = list(dict.fromkeys(roster))
            intersect = [node for node in metadata_nodes if node in roster]
            candidates = intersect or roster
        else:
            candidates = metadata_nodes
        if limit and limit > 0:
            candidates = candidates[:limit]
        return candidates

    def _select_scope_roles(
        self,
        scope_configs: Mapping[str, HierarchyScopeConfig],
    ) -> Tuple[str, HierarchyScopeConfig, Optional[str], Optional[HierarchyScopeConfig]]:
        if not scope_configs:
            fallback = HierarchyLevelConfig()
            fallback.apply_training_defaults(None)
            return fallback.scope_name.lower(), fallback, None, None
        ordered = sorted(
            scope_configs.items(),
            key=lambda item: (item[1].scope_index, item[0]),
        )
        scope_name, scope_cfg = ordered[0]
        higher_name: Optional[str] = None
        higher_cfg: Optional[HierarchyScopeConfig] = None
        if len(ordered) > 1:
            higher_name, higher_cfg = ordered[1]
        return scope_name, scope_cfg, higher_name, higher_cfg

    def _get_scope_config_entry(self, scope_name: Optional[str] = None) -> Optional[HierarchyScopeConfig]:
        configs = getattr(self, "scope_configs", None) or {}
        key = scope_name or getattr(self, "scope_name", None)
        if not key:
            return None
        return configs.get(str(key).lower())

    def _configure_scope_layer(self) -> None:
        """Configure aggregator/fan-out roles for all hierarchy scopes."""
        self._configure_scope_runtimes()

    def _configure_scope_runtimes(self) -> None:
        configs = getattr(self, "scope_configs", None) or {}
        for scope_name, config in configs.items():
            runtime = self._ensure_scope_runtime(scope_name, config)
            self._configure_single_scope_runtime(runtime)
        self._update_bridge_hooks()
        self._ensure_bridge_stack(allow_state_layer=True)

    def _configure_single_scope_runtime(self, runtime: ScopeRuntime) -> None:
        config = runtime.config
        runtime.scope_id = self._node_scope_identifier_for(runtime.scope_name, config)
        if not self._runtime_enabled(runtime):
            runtime.candidates = []
            runtime.is_candidate = False
            runtime.ecm_buffer = None
            runtime.aggregator = None
            return
        scope_id = runtime.scope_id
        roster = self._scope_member_roster(runtime.scope_name, scope_id)
        runtime.candidates = roster
        was_candidate = runtime.is_candidate
        runtime.is_candidate = self.node_id in roster
        collects = bool(config.collects_lower_scope)
        if runtime.is_candidate and collects:
            if runtime.ecm_buffer is None:
                freshness = float(
                    max(
                        self.inter_cluster_config.get("freshness_window", 300.0),
                        config.collection_timeout_seconds * 2,
                    )
                )
                runtime.ecm_buffer = ECMBuffer(freshness_window=freshness)
        else:
            runtime.ecm_buffer = None
        if runtime.is_candidate and runtime.aggregator is None and self.ipfs is not None:
            runtime.aggregator = StateAggregator(config, self.ipfs, self.blockchain)
        elif not runtime.is_candidate:
            runtime.aggregator = None
        if (
            runtime.is_candidate
            and not was_candidate
            and runtime.candidates
        ):
            hierarchy_logger.info(
                "%s candidate %s joined aggregator pool with %d candidates",
                (config.scope_name or runtime.scope_name).upper(),
                self.node_id,
                len(runtime.candidates),
            )

    def _runtime_enabled(self, runtime: ScopeRuntime) -> bool:
        interval = self._scope_interval_seconds(runtime.config)
        return bool(runtime.config.enabled and interval > 0)

    def _scope_layer_enabled(self) -> bool:
        return self._runtime_enabled(self.scope_runtime)

    def _higher_scope_enabled(self) -> bool:
        higher_cfg = getattr(self, "higher_scope_config", None)
        if not higher_cfg:
            return False
        higher_runtime = self._ensure_scope_runtime(self.higher_scope_name, higher_cfg)
        return self._runtime_enabled(higher_runtime)

    def _scope_round_budget(self) -> Optional[int]:
        """Estimate how many state rounds can occur over the training horizon."""
        if not self._scope_layer_enabled():
            return None
        interval_seconds = self._scope_interval_seconds(self.scope_config)
        if interval_seconds <= 0:
            return None
        runtime_hint = (
            (self.training_config or {}).get("max_runtime_seconds")
            or (self.training_config or {}).get("target_runtime_seconds")
            or getattr(self, "max_runtime_seconds", None)
        )
        try:
            runtime_seconds = float(runtime_hint) if runtime_hint else None
        except (TypeError, ValueError):
            runtime_seconds = None
        if runtime_seconds and runtime_seconds > 0:
            slots = int(runtime_seconds // max(1.0, interval_seconds))
            return max(1, slots) if slots > 0 else None
        return None

    def _maybe_apply_scope_model(self, round_idx: Optional[int] = None) -> None:
        """Fetch and apply the latest anchored model for this node's scope."""
        self._apply_scope_model_from_anchor(self.scope_name, self.scope_config, is_local_scope=True)

    def _maybe_apply_external_scope_model(self, scope_name: str, config: HierarchyLevelConfig) -> None:
        """Apply anchored models for a higher scope to this node's baseline."""
        self._apply_scope_model_from_anchor(scope_name, config, is_local_scope=False)

    def _record_scope_model_application(
        self,
        runtime: ScopeRuntime,
        scope_round: Optional[int],
        cid: Optional[str],
        hash_val: Optional[str],
        data_id: Optional[str],
        tensor: Optional[np.ndarray],
    ) -> None:
        if cid:
            runtime.last_model_cid = cid
            if runtime.scope_name.lower() == self.scope_name.lower():
                self._last_model_cid = cid
        if hash_val:
            runtime.last_model_hash = hash_val
            if runtime.scope_name.lower() == self.scope_name.lower():
                self._last_model_hash = hash_val
        if data_id:
            runtime.last_model_data_id = data_id
            if runtime.scope_name.lower() == self.scope_name.lower():
                self._last_model_data_id = data_id
        applied_key = self._scope_applied_key(runtime)
        if applied_key:
            last_cids = getattr(self, "_scope_last_applied_cids", None)
            if last_cids is None:
                last_cids = {}
                setattr(self, "_scope_last_applied_cids", last_cids)
            if cid:
                last_cids[applied_key] = cid
            progress = getattr(self, "_scope_last_applied_rounds", None)
            if progress is None:
                progress = {}
                setattr(self, "_scope_last_applied_rounds", progress)
            if scope_round is not None:
                progress[applied_key] = scope_round
        runtime.pending_anchor = None
        if tensor is not None:
            if runtime.scope_name.lower() == self.scope_name.lower():
                self._apply_scope_policy_tensor(tensor)
            else:
                self._apply_scope_policy_tensor_for(runtime.config, tensor, runtime.config.scope_name.upper())
        self._clear_scope_wait(runtime)

    def _apply_scope_model_from_anchor(
        self,
        scope_name: str,
        config: HierarchyLevelConfig,
        *,
        is_local_scope: bool,
    ) -> None:
        if (
            not config
            or not self.blockchain
            or not self.ipfs
            or not self.model
            or not self._node_participates_in_scope(scope_name, config)
        ):
            return
        scope_id = self._node_scope_identifier_for(scope_name, config)
        if not scope_id:
            hierarchy_logger.debug("Scope %s lacks identifier for node %s; skipping fetch", scope_name, getattr(self, "node_id", "unknown"))
            return
        scope_label = str(getattr(config, "scope_name", scope_name) or scope_name)
        runtime = self._ensure_scope_runtime(scope_name, config)
        anchor = runtime.pending_anchor
        runtime.pending_anchor = None
        if anchor is None:
            try:
                anchor = self.blockchain.get_latest_scope_model(scope_label, scope_id)
            except Exception as exc:  # noqa: BLE001
                hierarchy_logger.warning(
                    "Failed to query latest %s model for %s: %s",
                    scope_label,
                    scope_id,
                    exc,
                )
                return
        if anchor is None:
            hierarchy_logger.debug(
                "No %s models available yet for scope_id=%s; will retry on next trigger",
                scope_label.lower(),
                scope_id,
            )
            return
        last_cids = getattr(self, "_scope_last_applied_cids", None)
        if last_cids is None:
            last_cids = {}
            setattr(self, "_scope_last_applied_cids", last_cids)
        scope_key = f"{scope_label.lower()}::{scope_id}"
        if last_cids.get(scope_key) == anchor.cid:
            hierarchy_logger.info(
                "No new %s model available for %s=%s (cid=%s... already applied)",
                scope_label.upper(),
                scope_label.lower(),
                scope_id,
                anchor.cid[:12],
            )
            return
        upstream_model = self.ipfs.get(anchor.cid)
        if upstream_model is None:
            hierarchy_logger.warning(
                "%s latest model unavailable on IPFS (cid=%s...)",
                scope_label.upper(),
                anchor.cid[:12],
            )
            return
        if not verify_model_hash(upstream_model, anchor.hash):
            hierarchy_logger.warning(
                "%s latest model hash mismatch (cid=%s...)",
                scope_label.upper(),
                anchor.cid[:12],
            )
            return
        self._record_scope_model_application(
            runtime,
            anchor.round_num,
            anchor.cid,
            anchor.hash,
            anchor.data_id,
            upstream_model,
        )
        hierarchy_logger.info(
            "Applied %s model round %s (cid=%s...) from latest-scope endpoint",
            scope_label.upper(),
            anchor.round_num if anchor.round_num is not None else "?",
            anchor.cid[:12],
        )

    def _apply_scope_policy_tensor(self, upstream_tensor: np.ndarray) -> None:
        """Apply upstream tensor for the current scope according to the configured policy."""
        self._apply_scope_policy_tensor_for(self.scope_config, upstream_tensor, self._scope_label_upper())

    def _apply_scope_policy_tensor_for(
        self,
        config: HierarchyLevelConfig,
        upstream_tensor: np.ndarray,
        scope_label: Optional[str] = None,
    ) -> None:
        label = scope_label or getattr(config, "scope_name", "scope").upper()
        policy = (getattr(config, "apply_policy", "replace") or "replace").lower()
        alpha = float(getattr(config, "apply_alpha", 0.0) or 0.0)
        if policy == "interpolate":
            hierarchy_logger.info(
                "Applying %s model using '%s' policy (alpha=%.3f)",
                label,
                policy,
                alpha,
            )
        else:
            hierarchy_logger.info(
                "Applying %s model using '%s' policy",
                label,
                policy,
            )
        if policy == "interpolate":
            local_vec = self._export_local_model_vector()
            if local_vec is None:
                hierarchy_logger.warning("Cannot interpolate scope model: local model not available")
                return
            upstream_vec = upstream_tensor.flatten().astype(np.float32)
            alpha = max(0.0, min(alpha, 0.49))
            blended = alpha * upstream_vec + (1.0 - alpha) * local_vec
            reshaped = blended.reshape(upstream_tensor.shape)
            self._apply_model_tensor(reshaped)
        else:
            self._apply_model_tensor(upstream_tensor)
        self._prime_convergence_tracker_state()

    def _export_local_model_vector(self) -> Optional[np.ndarray]:
        exporter = getattr(self, "_export_local_model_vector", None)
        if callable(exporter):
            return exporter()
        return None

    def _schedule_round_for_handler(self, handler: ScopeRoundHandler, trigger_round_idx: int) -> None:
        config = handler.config
        scope_label = getattr(config, "scope_name", handler.scope_name) or handler.scope_name
        interval_seconds = self._scope_interval_seconds(config)
        enabled = bool(getattr(config, "enabled", False) and interval_seconds > 0)
        if not enabled:
            legacy_interval = getattr(config, "rounds_per_scope", 0)
            if legacy_interval > 0:
                hierarchy_logger.warning(
                    "%s has rounds_per_scope=%d but interval_seconds is not configured; skipping legacy scheduling",
                    scope_label.upper(),
                    legacy_interval,
                )
            return
        scope_key = handler.scope_name.lower()
        next_fire_map = getattr(self, "_scope_next_fire_at", None)
        if next_fire_map is None:
            next_fire_map = {}
            self._scope_next_fire_at = next_fire_map
        next_fire = next_fire_map.get(scope_key)
        if not next_fire:
            next_fire_map[scope_key] = time.time() + interval_seconds
            hierarchy_logger.debug(
                "%s timer initialized; first fire at %.0fs interval",
                scope_label.upper(),
                interval_seconds,
            )
            return
        now = time.time()
        did_schedule = False
        while now >= next_fire:
            scope_round = self._increment_scope_round(scope_key)
            if not any(sr == scope_round for sr, _ in handler.round_queue):
                handler.round_queue.append((scope_round, trigger_round_idx))
                self._queue_scope_wait(handler.scope_name, config)
                hierarchy_logger.info(
                    "Scheduling %s round %d via %.1fs interval timer",
                    scope_label.lower(),
                    scope_round,
                    interval_seconds,
                )
            did_schedule = True
            next_fire += interval_seconds
        next_fire_map[scope_key] = next_fire
        if not did_schedule:
            hierarchy_logger.debug(
                "%s timer not due yet (next fire in %.1fs)",
                scope_label.upper(),
                max(0.0, next_fire - now),
            )

    def _maybe_schedule_scope_round(self, round_idx: int) -> None:
        """Check each registered scope handler and schedule rounds when due."""
        handlers = list(getattr(self, "_scope_round_handlers", {}).values())
        if not handlers:
            return
        for handler in handlers:
            self._schedule_round_for_handler(handler, round_idx)

    def _run_next_scope_round(self, scope_name: Optional[str] = None) -> Optional[Tuple[str, int, int]]:
        """Execute the next scheduled round for the supplied scope, if available."""
        handler = self._get_scope_round_handler(scope_name)
        if handler is None:
            hierarchy_logger.debug("No scope handler registered for %s", scope_name or self.scope_name)
            return None
        config = handler.config
        interval_seconds = self._scope_interval_seconds(config)
        if not (getattr(config, "enabled", False) and interval_seconds > 0):
            handler.round_queue.clear()
            return None
        if not handler.round_queue:
            return None
        scope_round, source_round = handler.round_queue.popleft()
        scope_label = str(getattr(config, "scope_name", handler.scope_name) or handler.scope_name)
        source_label = handler.trigger_label or "parent"
        label = f"{scope_label.upper()} Round {scope_round}"
        interval_note = ""
        if interval_seconds > 0:
            interval_note = f" | trigger {scope_label.lower()} round after {interval_seconds:.0f} seconds"
        hierarchy_logger.info("\n" + "=" * 60)
        hierarchy_logger.info(
            "%s (triggered after %s round %d)%s",
            label,
            source_label,
            source_round + 1,
            interval_note,
        )
        hierarchy_logger.info("=" * 60)
        try:
            runtime = self._ensure_scope_runtime(scope_label, config)
        except KeyError:
            runtime = None
        if runtime is not None:
            leader_id = self._scope_round_leader(
                runtime.scope_name,
                runtime.scope_id,
                scope_round,
                fallback=runtime.candidates,
            )
            self._emit_scope_round_leader_metric(runtime, scope_round, leader_id)
        handler.dispatch_fn(scope_round, source_round)
        handler.execute_fn(scope_round, source_round)
        return handler.scope_name, scope_round, source_round

    def _execute_scope_round(
        self,
        scope_round: int,
        cluster_round: int,
        runtime: Optional[ScopeRuntime] = None,
    ) -> bool:
        """Collect child-scope artifacts, merge models, and publish the round."""
        runtime = runtime or self._runtime_for_scope(self.scope_name)
        if not self._runtime_enabled(runtime):
            return False
        config = runtime.config
        label_lower = self._runtime_label_lower(runtime)
        label_upper = label_lower.upper()
        if runtime.scope_id is None:
            hierarchy_logger.debug("Node %s is not part of %s scope; skipping", self.node_id, label_lower)
            return False
        if scope_round not in runtime.rounds_logged:
            total_scope_rounds = self._scope_round_budget()
            label = f"/{total_scope_rounds}" if total_scope_rounds else ""
            runtime.rounds_logged.add(scope_round)
        can_aggregate = runtime.aggregator is not None and runtime.ecm_buffer is not None
        scope_id = runtime.scope_id
        expected_leader = self._scope_round_leader(
            runtime.scope_name,
            scope_id,
            scope_round,
            fallback=runtime.candidates,
        )
        if not can_aggregate:
            return self._wait_for_scope_anchor_observer(
                scope_round,
                runtime=runtime,
                leader_id=expected_leader,
            )
        if expected_leader and self.node_id != expected_leader:
            return self._wait_for_scope_anchor_observer(
                scope_round,
                runtime=runtime,
                leader_id=expected_leader,
            )
        if scope_round in runtime.round_cache:
            return True
        deadline = time.time() + max(1.0, float(config.collection_timeout_seconds))
        snapshot: Dict[str, StateClusterModel] = {}
        missing: List[str] = []
        required_clusters = self._child_scope_ids_for_runtime(runtime)
        child_scope_label = self._lower_scope_name(runtime.scope_name) or "child"
        while time.time() < deadline:
            artifacts = runtime.ecm_buffer.get_fresh_ecms()
            snapshot, missing = runtime.aggregator.build_snapshot(
                artifacts,
                required_clusters,
                None,
            )
            if not missing:
                break
            hierarchy_logger.debug(
                "%s round %d waiting for %s artifacts from: %s",
                label_lower,
                scope_round,
                child_scope_label.upper(),
                ", ".join(sorted(missing)),
            )
            time.sleep(1.0)
        collected = sorted(snapshot.keys())
        hierarchy_logger.info(
            "%s round %d collected %d/%d %s artifacts: %s",
            label_lower,
            scope_round,
            len(collected),
            len(required_clusters),
            child_scope_label.upper(),
            ", ".join(collected) if collected else "none",
        )
        if missing:
            hierarchy_logger.warning(
                "%s round %d missing %s artifacts from: %s",
                label_lower,
                scope_round,
                child_scope_label.upper(),
                ", ".join(sorted(missing)),
            )
            return False
        try:
            models = runtime.aggregator.fetch_models(snapshot, fallback_lookup=self._lookup_lower_scope_anchor)
            merged_model = runtime.aggregator.merge_models(models)
        except StateAggregationError as exc:
            hierarchy_logger.error("%s aggregation failed for round %d: %s", label_upper, scope_round, exc)
            return False
        hierarchy_logger.info(
            "%s round %d merged %d %s artifacts successfully; preparing commit",
            label_lower,
            scope_round,
            len(models),
            child_scope_label.lower(),
        )
        model_hash = compute_model_hash(merged_model)
        runtime.round_cache[scope_round] = merged_model
        runtime.round_hashes[scope_round] = model_hash
        self._try_scope_commit(scope_round, runtime=runtime)
        return True

    def _dispatch_scope_artifacts(
        self,
        scope_round: int,
        cluster_round: int,
        runtime: Optional[ScopeRuntime] = None,
    ) -> None:
        """Have bridge nodes forward their latest child-scope artifact to the elected aggregator."""
        runtime = runtime or self.scope_runtime
        if not self._runtime_enabled(runtime):
            return
        label_lower = self._runtime_label_lower(runtime)
        if runtime.scope_id is None:
            hierarchy_logger.debug("Node %s not a member of %s scope; skipping dispatch", self.node_id, label_lower)
            return
        child_scope = self._lower_scope_name(runtime.scope_name)
        if not child_scope:
            hierarchy_logger.debug(
                "No child scope configured for %s; skipping dispatch",
                label_lower,
            )
            return
        payload_cid, payload_hash = self._fanout_payload_for_child_scope(child_scope)
        if not payload_cid or not payload_hash:
            hierarchy_logger.info(
                "Skipping %s dispatch for round %d: missing latest model reference (cid=%s, hash=%s)",
                label_lower,
                scope_round,
                payload_cid or "N/A",
                payload_hash or "N/A",
            )
            return
        if child_scope == "cluster":
            child_scope_id = f"cluster_{self.clique_id}"
        else:
            membership = self._node_scope_membership_map()
            child_scope_id = membership.get(child_scope)
        if not child_scope_id:
            hierarchy_logger.info(
                "Skipping %s dispatch for round %d: unable to resolve %s membership for node %s",
                label_lower,
                scope_round,
                child_scope,
                self.node_id,
            )
            return
        channel_label = f"{runtime.scope_name.lower()}::{child_scope_id}"
        selection = self._select_healthy_scope_aggregator(runtime, scope_round)
        if not selection:
            hierarchy_logger.warning(
                "%s dispatch skipped for round %d: no healthy aggregators reachable",
                label_lower,
                scope_round,
            )
            return
        aggregator_id, aggregator_address = selection
        if aggregator_id == self.node_id:
            hierarchy_logger.debug(
                "Node %s is %s round %d aggregator; skipping fan-out dispatch",
                self.node_id,
                label_lower.upper(),
                scope_round,
            )
            return
        if not self._is_fanout_node_for_scope_round(runtime, scope_round):
            hierarchy_logger.debug(
                "%s not selected as %s fan-out for round %d; waiting for higher-level model",
                self.node_id,
                label_lower,
                scope_round,
            )
            return
        if not aggregator_address:
            hierarchy_logger.warning(
                "Healthy %s aggregator %s lacks bridge address; skipping dispatch",
                label_lower,
                aggregator_id,
            )
            return
        if not self._ensure_bridge_client(allow_state_layer=True):
            hierarchy_logger.warning(
                "Cannot dispatch %s artifacts for round %d: bridge client unavailable",
                label_lower,
                scope_round,
            )
            return
        accepted = self.bridge_client.broadcast_ecm(
            [aggregator_address],
            channel_label,
            cluster_round,
            payload_cid,
            payload_hash,
        )
        hierarchy_logger.info(
            "Dispatched %s artifacts for round %d to aggregator %s (accepted=%d)",
            label_lower,
            scope_round,
            aggregator_id,
            accepted,
        )

    def _wait_for_scope_anchor_observer(
        self,
        scope_round: int,
        runtime: Optional[ScopeRuntime] = None,
        *,
        leader_id: Optional[str] = None,
    ) -> bool:
        """Pause briefly for another candidate to publish the scope anchor."""
        runtime = runtime or self.scope_runtime
        if not self._runtime_enabled(runtime):
            return False
        label_lower = self._runtime_label_lower(runtime)
        label_upper = self._runtime_label_upper(runtime)
        wait_seconds = float(getattr(runtime.config, "wait_seconds", 0.0) or 0.0)
        if wait_seconds <= 0:
            if leader_id:
                hierarchy_logger.info(
                    "%s round %d aggregator=%s; no wait window configured before checking for anchor",
                    label_upper,
                    scope_round,
                    leader_id,
                )
            else:
                hierarchy_logger.info(
                    "No wait window configured for %s observers; continuing without delay",
                    label_lower,
                )
            return True
        if leader_id:
            hierarchy_logger.info(
                "%s round %d aggregator=%s; waiting up to %.0fs for anchor before continuing",
                label_upper,
                scope_round,
                leader_id,
                wait_seconds,
            )
        else:
            hierarchy_logger.info(
                "Waiting up to %.0fs for %s round %d anchor to appear before continuing",
                wait_seconds,
                label_lower,
                scope_round,
            )
        anchor = self._wait_for_scope_anchor(
            scope_round,
            wait_seconds,
            runtime=runtime,
            expedite_fetch=True,
        )
        if anchor:
            hierarchy_logger.info(
                "Observed %s round %d anchor committed elsewhere (cid=%s...)",
                label_lower,
                scope_round,
                anchor[0][:8],
            )
        else:
            hierarchy_logger.info(
                "%s round %d anchor not observed after %.0fs wait window; proceeding with cluster training",
                label_lower,
                scope_round,
                wait_seconds,
            )
        return True

    def _lookup_lower_scope_anchor(self, cluster_id: str, round_idx: int) -> Optional[Tuple[str, str]]:
        if not self.blockchain:
            return None
        key = cluster_id
        if "::" in key:
            _, key = key.split("::", 1)
        try:
            return self.blockchain.get_anchor(key, round_idx, scope=AnchorScope.CLUSTER)
        except Exception as exc:  # noqa: BLE001
            hierarchy_logger.warning(
                "Failed to fetch anchor for cluster %s round %s: %s",
                cluster_id,
                round_idx,
                exc,
            )
            return None

    def _try_scope_commit(
        self,
        scope_round: int,
        runtime: Optional[ScopeRuntime] = None,
    ) -> None:
        runtime = runtime or self.scope_runtime
        if not self._runtime_enabled(runtime):
            return
        label_lower = self._runtime_label_lower(runtime)
        candidates = runtime.candidates or []
        aggregator = runtime.aggregator
        if not candidates or aggregator is None or scope_round in runtime.committed_rounds:
            return
        model = runtime.round_cache.get(scope_round)
        if model is None:
            hierarchy_logger.debug(
                "%s round %d consensus reached but local model missing; waiting to commit",
                label_lower.upper(),
                scope_round,
            )
            return
        ordered_candidates = self._ordered_scope_candidates(runtime, scope_round) or list(candidates)
        commit_timeout = float(getattr(runtime.config, "commit_timeout_seconds", 0.0) or 0.0)
        for candidate in ordered_candidates:
            if candidate == self.node_id:
                anchor = aggregator.get_anchor(scope_round)
                if anchor:
                    cid, hash_val = anchor
                    self._record_scope_model_application(runtime, scope_round, cid, hash_val, None, model)
                    self._mark_scope_round_committed(scope_round, runtime=runtime)
                    return
                try:
                    cid, hash_val, data_id = aggregator.publish_state_model(model, scope_round)
                except StateAggregationError as exc:
                    hierarchy_logger.error(
                        "%s round %d commit failed on %s: %s",
                        label_lower,
                        scope_round,
                        self.node_id,
                        exc,
                    )
                    continue
                if cid and hash_val:
                    hierarchy_logger.info(
                        "%s round %d committed by %s (cid=%s..., data_id=%s)",
                        label_lower,
                        scope_round,
                        self.node_id,
                        cid[:8],
                        data_id or "N/A",
                    )
                    scope_identifier = runtime.scope_id or getattr(runtime.config, "scope_id", None) or runtime.scope_name
                    pending_anchor = None
                    if scope_identifier:
                        pending_anchor = ModelAnchor(
                            cluster_id=str(scope_identifier),
                            round_num=scope_round,
                            cid=cid,
                            hash=hash_val,
                            data_id=data_id,
                        )
                    self._mark_scope_round_committed(scope_round, runtime=runtime)
                    self._record_scope_commit_metric(runtime, scope_round, self.node_id)
                    self._mark_scope_fetch_ready(runtime, cid, pending_anchor)
                    return
            else:
                anchor = self._wait_for_scope_anchor(scope_round, commit_timeout, runtime=runtime)
                if anchor:
                    hierarchy_logger.info(
                        "%s round %d observed anchor (cid=%s...) by peer",
                        label_lower,
                        scope_round,
                        anchor[0][:8],
                    )
                    self._mark_scope_round_committed(scope_round, runtime=runtime)
                    return
        hierarchy_logger.warning("%s round %d not anchored after iterating all candidates", label_lower, scope_round)

    def _wait_for_scope_anchor(
        self,
        scope_round: int,
        timeout: float,
        runtime: Optional[ScopeRuntime] = None,
        *,
        expedite_fetch: bool = False,
    ) -> Optional[Tuple[str, str]]:
        runtime = runtime or self.scope_runtime
        scope_id = self._runtime_scope_identifier(runtime)
        scope_label = str(getattr(runtime.config, "scope_name", runtime.scope_name) or runtime.scope_name or "state")
        if self.blockchain is None or not scope_id:
            return None
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            try:
                latest_anchor = self.blockchain.get_latest_scope_model(scope_label, scope_id)
            except Exception as exc:  # noqa: BLE001
                hierarchy_logger.warning(
                    "Failed to query latest %s model for scope_id=%s: %s",
                    scope_label,
                    scope_id,
                    exc,
                )
                return None
            if latest_anchor and latest_anchor.cid and latest_anchor.hash:
                latest_round = getattr(latest_anchor, "round_num", None)
                if latest_round is not None and latest_round >= scope_round:
                    if expedite_fetch:
                        self._mark_scope_fetch_ready(runtime, latest_anchor.cid, latest_anchor)
                    return (latest_anchor.cid, latest_anchor.hash)
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(5.0, remaining))
        return None

    def _wait_for_higher_scope_anchor(
        self,
        scope_name: str,
        config: HierarchyLevelConfig,
        scope_round: int,
    ) -> bool:
        """Block until a higher-scope anchor appears or the configured timeout elapses."""
        if not self.blockchain:
            return False
        scope_id = getattr(config, "scope_id", None)
        if not scope_id:
            return False
        scope_namespace = self._scope_namespace(config)
        wait_budget = (
            float(getattr(config, "collection_timeout_seconds", 0.0))
            + float(getattr(config, "consensus_timeout_seconds", 0.0))
            + float(getattr(config, "commit_timeout_seconds", 0.0))
        )
        wait_budget = max(30.0, wait_budget)
        deadline = time.time() + wait_budget
        label = getattr(config, "scope_name", scope_name).upper()
        hierarchy_logger.info(
            "Waiting up to %.0fs for %s round %d anchor",
            wait_budget,
            label,
            scope_round,
        )
        while time.time() < deadline:
            try:
                anchor = self.blockchain.get_anchor(
                    scope_id,
                    scope_round,
                    scope=scope_namespace,
                    suppress_not_found_log=True,
                )
            except Exception as exc:  # noqa: BLE001
                hierarchy_logger.warning(
                    "Failed to poll %s round %d anchor: %s",
                    label,
                    scope_round,
                    exc,
                )
                return False
            if anchor:
                hierarchy_logger.info(
                    "%s round %d anchor observed (cid=%s...)",
                    label,
                    scope_round,
                    anchor[0][:8],
                )
                return True
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(5.0, remaining))
        hierarchy_logger.warning(
            "%s round %d anchor not observed after %.0fs; will continue and retry later",
            label,
            scope_round,
            wait_budget,
        )
        return False

    def _mark_scope_round_committed(
        self,
        scope_round: int,
        runtime: Optional[ScopeRuntime] = None,
    ) -> None:
        runtime = runtime or self.scope_runtime
        runtime.committed_rounds.add(scope_round)
        runtime.round_cache.pop(scope_round, None)
        runtime.round_hashes.pop(scope_round, None)

    def _announce_higher_round(self, higher_round: int, source_scope_round: int) -> None:
        """Placeholder for higher-level aggregation flow."""
        if not self._higher_scope_enabled():
            return
        hierarchy_logger.info(
            "%s round %d triggered after %s round %d",
            self._higher_scope_label_upper(),
            higher_round,
            self._scope_label_lower(),
            source_scope_round,
        )
