"""Hierarchy-specific helpers for node_service."""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
from secure_aggregation.node import ECM, ECMBuffer
from secure_aggregation.state import (
    HierarchyLevelConfig,
    StateAggregationApproach,
    StateAggregationError,
    StateAggregator,
    StateClusterModel,
)
from secure_aggregation.storage.model_store import AnchorScope, compute_model_hash, verify_model_hash
from secure_aggregation.utils import get_logger

hierarchy_logger = get_logger("hierarchy")

HierarchyScopeConfig = HierarchyLevelConfig


@dataclass
class ScopeDigest:
    """Scope-level model digest shared between central candidates."""

    node_id: str
    scope_id: str
    scope_round: int
    cluster_round: int
    model_hash: str
    model_cid: Optional[str]
    received_at: float


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
    digest_records: Dict[int, Dict[str, ScopeDigest]]
    committed_rounds: Set[int]
    is_candidate_fn: Callable[[], bool]
    dispatch_fn: Callable[[int, int], None]
    execute_fn: Callable[[int, int], bool]
    schedule_higher_fn: Optional[Callable[[int], None]] = None
    budget_fn: Optional[Callable[[], Optional[int]]] = None


def build_scope_signal_cid(
    scope_label: str,
    scope_id: Optional[str],
    scope_round: int,
    node_id: str,
) -> str:
    """Create a recognizable CID for scope digest signaling over the bridge."""
    safe_scope = (scope_label or "scope").lower()
    safe_scope_id = scope_id or "unknown"
    return f"signal::{safe_scope}::{safe_scope_id}::{scope_round}::{node_id}"


def parse_scope_digest_signal(ecm: ECM) -> Optional[ScopeDigest]:
    """Convert a bridge ECM carrying a digest signal into a ScopeDigest."""
    if not ecm.is_signal or not ecm.source_cluster:
        return None
    try:
        scope_label, scope_identifier = ecm.source_cluster.split("::", 1)
    except ValueError:
        return None
    if not scope_identifier:
        return None
    scope_round = ecm.round_idx
    if scope_round < 0:
        return None
    cluster_round = scope_round
    origin_node: Optional[str] = None
    if ecm.convergence_data_id:
        try:
            payload = json.loads(ecm.convergence_data_id)
            origin_node = payload.get("node_id") or None
            payload_round = payload.get("cluster_round")
            if payload_round is not None:
                try:
                    cluster_round = int(payload_round)
                except (TypeError, ValueError):
                    pass
        except (TypeError, ValueError, json.JSONDecodeError):
            hierarchy_logger.debug("Malformed scope digest payload: %s", ecm.convergence_data_id)
    if not origin_node and ecm.cid.startswith("signal::"):
        parts = ecm.cid.split("::")
        if len(parts) >= 5:
            origin_node = parts[-1]
    if not origin_node:
        hierarchy_logger.debug("Unable to determine origin node for scope digest cid=%s", ecm.cid)
        return None
    return ScopeDigest(
        node_id=origin_node,
        scope_id=scope_identifier,
        scope_round=scope_round,
        cluster_round=cluster_round,
        model_hash=ecm.hash,
        model_cid=None,
        received_at=ecm.received_at,
    )


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
        """Determine which nodes act as state aggregators based on metadata."""
        if not self.scope_config.enabled:
            if self.scope_ecm_buffer is not None:
                self.scope_ecm_buffer = None
                self._update_bridge_hooks()
                self._ensure_bridge_stack()
            self.scope_candidates = []
            self.is_scope_candidate = False
            return
        if not self.central_metadata:
            return
        if self.scope_config.approach == StateAggregationApproach.RING_STAR:
            candidates = list(self.central_metadata.central_nodes)
        else:
            candidates = list(self.central_metadata.central_nodes)
        candidates = sorted(dict.fromkeys(candidates))
        previous_candidates = self.scope_candidates
        self.scope_candidates = candidates
        was_candidate = self.is_scope_candidate
        self.is_scope_candidate = self.node_id in self.scope_candidates
        if self.is_scope_candidate and self.scope_ecm_buffer is None:
            freshness = float(
                max(
                    self.inter_cluster_config.get("freshness_window", 300.0),
                    self.scope_config.collection_timeout_seconds * 2,
                )
            )
            self.scope_ecm_buffer = ECMBuffer(freshness_window=freshness)
            self._update_bridge_hooks()
            self._ensure_bridge_stack()
        elif not self.is_scope_candidate and self.scope_ecm_buffer is not None:
            self.scope_ecm_buffer = None
            self._update_bridge_hooks()
            self._ensure_bridge_stack()
        if self.is_scope_candidate and self.scope_aggregator is None and self.ipfs is not None:
            self.scope_aggregator = StateAggregator(self.scope_config, self.ipfs, self.blockchain)
        if (
            candidates
            and candidates != previous_candidates
            and self.is_scope_candidate
            and not was_candidate
        ):
            hierarchy_logger.info(
                "%s candidate %s joined aggregator pool with %d candidates",
                self._scope_label_upper(),
                self.node_id,
                len(candidates),
            )

    def _scope_layer_enabled(self) -> bool:
        return bool(self.scope_config.enabled and self.scope_config.rounds_per_scope > 0)

    def _higher_scope_enabled(self) -> bool:
        higher_cfg = getattr(self, "higher_scope_config", None)
        rounds_per = getattr(higher_cfg, "rounds_per_scope", 0) if higher_cfg else 0
        return bool(higher_cfg and higher_cfg.enabled and rounds_per > 0)

    def _scope_round_budget(self) -> Optional[int]:
        """Estimate how many state rounds can occur over the training horizon."""
        if not self._scope_layer_enabled():
            return None
        interval = max(1, self.scope_config.rounds_per_scope)
        cluster_total = self.training_config.get("rounds")
        try:
            cluster_total = int(cluster_total)
        except (TypeError, ValueError):
            cluster_total = None
        if not cluster_total or cluster_total <= 0:
            cluster_total = self.max_training_rounds
        if cluster_total <= 0:
            return None
        return max(1, math.ceil(cluster_total / interval))

    def _maybe_apply_scope_model(self, round_idx: int) -> None:
        """Fetch and apply anchored state models as the new baseline."""
        scope_key = self.scope_name.lower()
        scope_progress = getattr(self, "_scope_last_applied_rounds", None)
        if scope_progress is None:
            scope_progress = {}
            setattr(self, "_scope_last_applied_rounds", scope_progress)
        scope_id = self._scope_identifier()
        scope_namespace = self._scope_namespace()
        if (
            not self._scope_layer_enabled()
            or not self.blockchain
            or not self.ipfs
            or not scope_id
            or not self.model
        ):
            return
        interval = max(1, self.scope_config.rounds_per_scope)
        completed_rounds = (round_idx + 1) // interval
        last_applied = scope_progress.get(scope_key, 0)
        while last_applied < completed_rounds:
            target_round = last_applied + 1
            try:
                anchor = self.blockchain.get_anchor(
                    scope_id,
                    target_round,
                    scope=scope_namespace,
                )
            except Exception as exc:  # noqa: BLE001
                hierarchy_logger.warning(
                    "Failed to fetch state anchor for round %d: %s",
                    target_round,
                    exc,
                )
                return
            if anchor is None:
                hierarchy_logger.debug(
                    "State model round %d not yet available; will retry later",
                    target_round,
                )
                return
            cid, expected_hash = anchor
            scope_label = getattr(self.scope_config, "scope_name", "state")
            hierarchy_logger.info(
                "%s model round %d detected on blockchain (cid=%s..., hash=%s...)",
                scope_label.capitalize(),
                target_round,
                cid[:12],
                expected_hash[:12],
            )
            state_model = self.ipfs.get(cid)
            if state_model is None:
                hierarchy_logger.warning(
                    "State model round %d unavailable on IPFS (cid=%s...)",
                    target_round,
                    cid[:16],
                )
                return
            if not verify_model_hash(state_model, expected_hash):
                hierarchy_logger.warning(
                    "State model round %d hash mismatch (cid=%s...)",
                    target_round,
                    cid[:16],
                )
                return
            self._apply_scope_policy_tensor_for(self.scope_config, state_model, scope_label.upper())
            self._last_model_cid = cid
            self._last_model_hash = expected_hash
            self._last_model_data_id = None
            last_applied = target_round
            scope_progress[scope_key] = last_applied
            hierarchy_logger.info(
                "Applied %s round %d model (cid=%s...); next cluster rounds will start from this baseline",
                scope_label.upper(),
                target_round,
                cid[:12],
            )

    def _maybe_apply_external_scope_model(self, scope_name: str, config: HierarchyLevelConfig) -> None:
        """Apply anchored models for a higher scope to this node's baseline."""
        scope_key = scope_name.lower()
        if not self.blockchain or not self.ipfs or not self.model:
            return
        scope_id = getattr(config, "scope_id", None)
        if not scope_id:
            return
        scope_progress = getattr(self, "_scope_last_applied_rounds", None)
        if scope_progress is None:
            scope_progress = {}
            setattr(self, "_scope_last_applied_rounds", scope_progress)
        last_applied = scope_progress.get(scope_key, 0)
        target_round = last_applied + 1
        scope_namespace = self._scope_namespace(config)
        while True:
            try:
                anchor = self.blockchain.get_anchor(
                    scope_id,
                    target_round,
                    scope=scope_namespace,
                )
            except Exception as exc:  # noqa: BLE001
                hierarchy_logger.warning(
                    "Failed to fetch %s anchor for round %d: %s",
                    scope_name,
                    target_round,
                    exc,
                )
                return
            if anchor is None:
                hierarchy_logger.debug(
                    "%s model round %d not yet available; will retry later",
                    scope_name.upper(),
                    target_round,
                )
                return
            cid, expected_hash = anchor
            scope_label = getattr(config, "scope_name", scope_name)
            hierarchy_logger.info(
                "%s model round %d detected on blockchain (cid=%s..., hash=%s...)",
                scope_label.capitalize(),
                target_round,
                cid[:12],
                expected_hash[:12],
            )
            scope_model = self.ipfs.get(cid)
            if scope_model is None:
                hierarchy_logger.warning(
                    "%s model round %d unavailable on IPFS (cid=%s...)",
                    scope_label,
                    target_round,
                    cid[:16],
                )
                return
            if not verify_model_hash(scope_model, expected_hash):
                hierarchy_logger.warning(
                    "%s model round %d hash mismatch (cid=%s...)",
                    scope_label,
                    target_round,
                    cid[:16],
                )
                return
            self._apply_scope_policy_tensor_for(config, scope_model, scope_label.upper())
            scope_progress[scope_key] = target_round
            hierarchy_logger.info(
                "Applied %s round %d model (cid=%s...); downstream scopes will start from this baseline",
                scope_label.upper(),
                target_round,
                cid[:12],
            )
            target_round += 1

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

    def _maybe_schedule_scope_round(self, round_idx: int) -> None:
        """Trigger scope aggregation when the configured interval elapses."""
        if not self._scope_layer_enabled():
            hierarchy_logger.debug("Skipping %s round scheduling: layer disabled", self._scope_label_lower())
            return
        interval = self.scope_config.rounds_per_scope
        if interval <= 0:
            return
        due = (round_idx + 1) % interval == 0
        hierarchy_logger.debug(
            "%s round scheduler invoked for cluster round %d (interval=%d, due=%s)",
            self._scope_label_upper(),
            round_idx + 1,
            interval,
            due,
        )
        if not due:
            return
        scope_round = (round_idx + 1) // interval
        if any(sr == scope_round for sr, _ in self._scope_round_queue):
            hierarchy_logger.debug(
                "%s round %d already scheduled; skipping duplicate", self._scope_label_upper(), scope_round
            )
            return
        hierarchy_logger.info(
            "Scheduling %s round %d to run after completion of cluster round %d",
            self._scope_label_lower(),
            scope_round,
            round_idx + 1,
        )
        self._scope_round_queue.append((scope_round, round_idx))

    def _run_next_scope_round(self, scope_name: Optional[str] = None) -> Optional[Tuple[str, int, int]]:
        """Execute the next scheduled round for the supplied scope, if available."""
        handler = self._get_scope_round_handler(scope_name)
        if handler is None:
            hierarchy_logger.debug("No scope handler registered for %s", scope_name or self.scope_name)
            return None
        config = handler.config
        rounds_per_scope = getattr(config, "rounds_per_scope", 0)
        if not (getattr(config, "enabled", False) and rounds_per_scope > 0):
            handler.round_queue.clear()
            return None
        if not handler.round_queue:
            return None
        scope_round, source_round = handler.round_queue.popleft()
        total_scope_rounds = handler.budget_fn() if handler.budget_fn else None
        scope_label = getattr(config, "scope_name", handler.scope_name)
        source_label = handler.trigger_label or "parent"
        label = f"{scope_label.upper()} Round {scope_round}/{total_scope_rounds or '?'}"
        hierarchy_logger.info("\n" + "=" * 60)
        hierarchy_logger.info("%s (triggered after %s round %d)", label, source_label, source_round + 1)
        hierarchy_logger.info("=" * 60)
        handler.dispatch_fn(scope_round, source_round)
        completed = handler.execute_fn(scope_round, source_round)
        if completed and handler.is_candidate_fn():
            if handler.schedule_higher_fn:
                handler.schedule_higher_fn(scope_round)
        return handler.scope_name, scope_round, source_round

    def _execute_scope_round(self, scope_round: int, cluster_round: int) -> bool:
        """Collect ECMs, merge models, and broadcast digest for a state round."""
        if not self._scope_layer_enabled():
            return False
        if scope_round not in self._scope_rounds_logged:
            total_scope_rounds = self._scope_round_budget()
            label = f"/{total_scope_rounds}" if total_scope_rounds else ""
            self._scope_rounds_logged.add(scope_round)
        can_aggregate = (
            self.scope_aggregator is not None
            and self.scope_ecm_buffer is not None
            and self.central_metadata is not None
        )
        if not can_aggregate:
            return self._wait_for_scope_anchor_observer(scope_round)
        if scope_round in self._scope_round_cache:
            return True
        deadline = time.time() + max(1.0, float(self.scope_config.collection_timeout_seconds))
        snapshot: Dict[str, StateClusterModel] = {}
        missing: List[str] = []
        while time.time() < deadline:
            ecms = self.scope_ecm_buffer.get_fresh_ecms()
            snapshot, missing = self.scope_aggregator.build_snapshot(
                ecms,
                self.central_metadata.cluster_ids,
                cluster_round,
            )
            if not missing:
                break
            hierarchy_logger.debug(
                "%s round %d waiting for ECMs from clusters: %s",
                self._scope_label_lower(),
                scope_round,
                ", ".join(sorted(missing)),
            )
            time.sleep(1.0)
        if missing:
            hierarchy_logger.warning(
                "%s round %d missing ECMs from clusters: %s",
                self._scope_label_lower(),
                scope_round,
                ", ".join(sorted(missing)),
            )
            return False
        try:
            models = self.scope_aggregator.fetch_models(snapshot, fallback_lookup=self._lookup_lower_scope_anchor)
            merged_model = self.scope_aggregator.merge_models(models)
        except StateAggregationError as exc:
            hierarchy_logger.error("%s aggregation failed for round %d: %s", self._scope_label_upper(), scope_round, exc)
            return False
        model_hash = compute_model_hash(merged_model)
        self._scope_round_cache[scope_round] = merged_model
        self._scope_round_hashes[scope_round] = model_hash
        self._broadcast_scope_digest(scope_round, cluster_round, model_hash)
        local_digest = ScopeDigest(
            node_id=self.node_id,
            scope_id=self._scope_identifier() or "",
            scope_round=scope_round,
            cluster_round=cluster_round,
            model_hash=model_hash,
            model_cid=None,
            received_at=time.time(),
        )
        self._record_scope_digest(local_digest, local=True)
        self._await_scope_digest_consensus(scope_round)
        self._verify_scope_digest_consensus(scope_round, model_hash)
        return True

    def _dispatch_scope_artifacts(self, scope_round: int, cluster_round: int) -> None:
        """Have bridge nodes forward their latest ECM to scope aggregators when a round starts."""
        if not self._scope_layer_enabled():
            return
        target_addresses: Dict[str, str] = {}
        if not self.central_neighbor_addresses:
            hierarchy_logger.debug(
                "No central neighbor addresses available for %s dispatch", self._scope_label_lower()
            )
        else:
            target_addresses.update(self.central_neighbor_addresses)
        if not target_addresses and self.participant_map:
            for node_id in self.central_metadata.central_nodes if self.central_metadata else []:
                addr = self.participant_map.get(node_id)
                if not addr:
                    continue
                try:
                    host, port_str = addr.split(":")
                    target_addresses[node_id] = f"{host}:{int(port_str) + 2000}"
                except ValueError:
                    continue
        if not target_addresses:
            hierarchy_logger.warning(
                "%s dispatch skipped for round %d: no central targets resolved",
                self._scope_label_lower(),
                scope_round,
            )
            return
        target_list = ", ".join(f"{node}@{addr}" for node, addr in sorted(target_addresses.items()))
        hierarchy_logger.info("%s dispatch targets for round %d: %s", self._scope_label_upper(), scope_round, target_list)
        if not self._last_model_cid or not self._last_model_hash:
            hierarchy_logger.info(
                "Skipping %s dispatch for round %d: missing latest model reference (cid=%s, hash=%s)",
                self._scope_label_lower(),
                scope_round,
                self._last_model_cid or "N/A",
                self._last_model_hash or "N/A",
            )
            return
        targets = [
            addr
            for node_id, addr in target_addresses.items()
            if node_id != self.node_id
        ]
        if not targets:
            return
        if not self._ensure_bridge_client(allow_state_layer=True):
            hierarchy_logger.warning(
                "Cannot dispatch %s artifacts for round %d: bridge client unavailable",
                self._scope_label_lower(),
                scope_round,
            )
            return
        accepted = self.bridge_client.broadcast_ecm(
            targets,
            f"cluster_{self.clique_id}",
            cluster_round,
            self._last_model_cid,
            self._last_model_hash,
        )
        hierarchy_logger.info(
            "Dispatched %s artifacts for round %d to %d/%d candidates",
            self._scope_label_lower(),
            scope_round,
            accepted,
            len(targets),
        )

    def _wait_for_scope_anchor_observer(self, scope_round: int) -> bool:
        """Wait for another node to commit the state round before continuing."""
        timeout = (
            float(self.scope_config.collection_timeout_seconds)
            + float(self.scope_config.consensus_timeout_seconds)
            + float(self.scope_config.commit_timeout_seconds)
        )
        hierarchy_logger.info(
            "Waiting up to %.0fs for %s round %d anchor to appear",
            timeout,
            self._scope_label_lower(),
            scope_round,
        )
        anchor = self._wait_for_scope_anchor(scope_round, timeout)
        if anchor:
            hierarchy_logger.info(
                "Observed %s round %d anchor committed elsewhere (cid=%s...)",
                self._scope_label_lower(),
                scope_round,
                anchor[0][:8],
            )
            return True
            hierarchy_logger.warning(
                "%s round %d anchor not observed after waiting %.0fs; continuing cluster training",
                self._scope_label_lower(),
                scope_round,
                timeout,
            )
        return False

    def _lookup_lower_scope_anchor(self, cluster_id: str, round_idx: int) -> Optional[Tuple[str, str]]:
        if not self.blockchain:
            return None
        try:
            return self.blockchain.get_anchor(cluster_id, round_idx, scope=AnchorScope.CLUSTER)
        except Exception as exc:  # noqa: BLE001
            hierarchy_logger.warning(
                "Failed to fetch anchor for cluster %s round %s: %s",
                cluster_id,
                round_idx,
                exc,
            )
            return None

    def _broadcast_scope_digest(self, scope_round: int, cluster_round: int, model_hash: str) -> None:
        """Share this node's scope digest with other central candidates."""
        hierarchy_logger.info(
            "Broadcasting %s digest round=%d to other aggregators",
            self._scope_label_lower(),
            scope_round,
        )
        if not self._scope_layer_enabled():
            return
        if not self.central_neighbor_addresses:
            hierarchy_logger.debug("No central neighbor addresses available for %s digest broadcast", self._scope_label_lower())
            return
        if not self._ensure_bridge_client(allow_state_layer=True):
            hierarchy_logger.warning(
                "Cannot broadcast %s digest for round %d: bridge client unavailable",
                self._scope_label_lower(),
                scope_round,
            )
            return
        payload = json.dumps(
            {
                "cluster_round": cluster_round,
                "node_id": self.node_id,
            }
        )
        scope_id = self._scope_identifier()
        cid = build_scope_signal_cid(self._scope_label_lower(), scope_id, scope_round, self.node_id)
        targets = [addr for node_id, addr in self.central_neighbor_addresses.items() if node_id != self.node_id]
        if not targets:
            return
        channel_label = f"{self._scope_label_lower()}::{scope_id or 'unknown'}"
        accepted = self.bridge_client.broadcast_ecm_with_metadata(
            targets,
            channel_label,
            scope_round,
            cid,
            model_hash,
            metadata=payload,
        )
        detail_targets = targets.copy()
        hierarchy_logger.info(
            "Broadcasted %s digest round=%d hash=%s... to %d/%d aggregators (%s)",
            self._scope_label_lower(),
            scope_round,
            model_hash[:8],
            accepted,
            len(targets),
            ", ".join(detail_targets) if detail_targets else "none",
        )

    def _record_scope_digest(self, digest: ScopeDigest, local: bool = False) -> None:
        if not self._scope_layer_enabled():
            return
        records = self._scope_digest_records.setdefault(digest.scope_round, {})
        prev = records.get(digest.node_id)
        if prev and prev.model_hash == digest.model_hash:
            return
        records[digest.node_id] = digest
        if local:
            hierarchy_logger.info(
                "Recorded local %s digest round=%d hash=%s...",
                self._scope_label_lower(),
                digest.scope_round,
                digest.model_hash[:8],
            )
        else:
            hierarchy_logger.info(
                "Observed %s digest round=%d from %s hash=%s...",
                self._scope_label_lower(),
                digest.scope_round,
                digest.node_id,
                digest.model_hash[:8],
            )
        self._maybe_finalize_scope_round(digest.scope_round)

    def _await_scope_digest_consensus(self, scope_round: int) -> None:
        """Block until digests from all central candidates are observed or timeout elapses."""
        if (
            not self._scope_layer_enabled()
            or not self.scope_candidates
            or len(self.scope_candidates) <= 1
        ):
            return
        timeout = float(self.scope_config.digest_timeout_seconds or 0.0)
        if timeout <= 0:
            return
        deadline = time.time() + timeout
        hierarchy_logger.info(
            "Waiting up to %.0fs for peer %s digests (round %d)",
            timeout,
            self._scope_label_lower(),
            scope_round,
        )
        last_log = 0.0
        while time.time() < deadline:
            self._process_incoming_signals()
            records = self._scope_digest_records.get(scope_round, {})
            if len(records) >= len(self.scope_candidates):
                hierarchy_logger.info(
                    "Received all %d %s digests for round %d",
                    len(records),
                    self._scope_label_lower(),
                    scope_round,
                )
                return
            missing = sorted(set(self.scope_candidates) - set(records.keys()))
            now = time.time()
            if missing and now - last_log >= 3.0:
                hierarchy_logger.debug(
                    "%s round %d waiting for digests from: %s",
                    self._scope_label_lower(),
                    scope_round,
                    ", ".join(missing),
                )
                last_log = now
            time.sleep(1.0)
        remaining = sorted(
            set(self.scope_candidates)
            - set((self._scope_digest_records.get(scope_round) or {}).keys())
        )
        if remaining:
            hierarchy_logger.warning(
                "%s round %d timed out waiting for digests from: %s",
                self._scope_label_lower(),
                scope_round,
                ", ".join(remaining),
            )

    def _verify_scope_digest_consensus(self, scope_round: int, local_hash: str) -> None:
        """Ensure all digests agree with the local hash before committing."""
        records = self._scope_digest_records.get(scope_round, {})
        if not records:
            return
        hashes = {digest.model_hash for digest in records.values()}
        if len(hashes) == 1 and local_hash in hashes:
            hierarchy_logger.info("%s round %d digests are consistent (hash=%s...)", self._scope_label_lower(), scope_round, local_hash[:8])
            return
        hierarchy_logger.warning(
            "%s round %d digest mismatch detected. Local hash=%s..., peers=%s",
            self._scope_label_lower(),
            scope_round,
            local_hash[:8],
            ", ".join(h[:8] for h in sorted(hashes)),
        )

    def _maybe_finalize_scope_round(self, scope_round: int) -> None:
        if (
            not self._scope_layer_enabled()
            or scope_round in self._scope_committed_rounds
            or not self.scope_candidates
        ):
            return
        records = self._scope_digest_records.get(scope_round, {})
        if len(records) < len(self.scope_candidates):
            return
        hashes = {digest.model_hash for digest in records.values()}
        if len(hashes) != 1:
            hierarchy_logger.warning(
                "%s round %d has conflicting digests: %s",
                self._scope_label_lower(),
                scope_round,
                ", ".join(sorted(hashes)),
            )
            return
        hierarchy_logger.info(
            "%s round %d digest consensus reached across %d candidates (hash=%s...)",
            self._scope_label_lower(),
            scope_round,
            len(records),
            next(iter(hashes))[:8],
        )
        self._try_scope_commit(scope_round)

    def _try_scope_commit(self, scope_round: int) -> None:
        if (
            not self._scope_layer_enabled()
            or not self.scope_candidates
            or not self.scope_aggregator
            or scope_round in self._scope_committed_rounds
        ):
            return
        model = self._scope_round_cache.get(scope_round)
        if model is None:
            hierarchy_logger.debug(
                "State round %d consensus reached but local model missing; waiting to commit",
                scope_round,
            )
            return
        leader_index = scope_round % len(self.scope_candidates)
        ordered_candidates = [
            self.scope_candidates[(leader_index + offset) % len(self.scope_candidates)]
            for offset in range(len(self.scope_candidates))
        ]
        for candidate in ordered_candidates:
            if candidate == self.node_id:
                anchor = self.scope_aggregator.get_anchor(scope_round)
                if anchor:
                    self._mark_scope_round_committed(scope_round)
                    return
                try:
                    cid, hash_val, data_id = self.scope_aggregator.publish_state_model(model, scope_round)
                except StateAggregationError as exc:
                    hierarchy_logger.error("%s round %d commit failed on %s: %s", self._scope_label_lower(), scope_round, self.node_id, exc)
                    continue
                if cid and hash_val:
                    hierarchy_logger.info(
                        "%s round %d committed by %s (cid=%s..., data_id=%s)",
                        self._scope_label_lower(),
                        scope_round,
                        self.node_id,
                        cid[:8],
                        data_id or "N/A",
                    )
                    self._mark_scope_round_committed(scope_round)
                    return
            else:
                anchor = self._wait_for_scope_anchor(scope_round, self.scope_config.commit_timeout_seconds)
                if anchor:
                    hierarchy_logger.info(
                        "%s round %d observed anchor (cid=%s...) by peer",
                        self._scope_label_lower(),
                        scope_round,
                        anchor[0][:8],
                    )
                    self._mark_scope_round_committed(scope_round)
                    return
        hierarchy_logger.warning("%s round %d not anchored after iterating all candidates", self._scope_label_lower(), scope_round)

    def _wait_for_scope_anchor(self, scope_round: int, timeout: float) -> Optional[Tuple[str, str]]:
        scope_id = self._scope_identifier()
        scope_namespace = self._scope_namespace()
        if self.scope_aggregator is None and (self.blockchain is None or not scope_id):
            return None
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            anchor: Optional[Tuple[str, str]] = None
            try:
                if self.scope_aggregator is not None:
                    anchor = self.scope_aggregator.get_anchor(
                        scope_round,
                        suppress_not_found_log=True,
                    )
                elif self.blockchain is not None and scope_id:
                    anchor = self.blockchain.get_anchor(
                        scope_id,
                        scope_round,
                        scope=scope_namespace,
                        suppress_not_found_log=True,
                    )
            except Exception as exc:  # noqa: BLE001
                hierarchy_logger.warning(
                    "Failed to poll state anchor round %d: %s",
                    scope_round,
                    exc,
                )
                return None
            if anchor:
                return anchor
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(5.0, remaining))
        return None

    def _mark_scope_round_committed(self, scope_round: int) -> None:
        self._scope_committed_rounds.add(scope_round)
        self._scope_digest_records.pop(scope_round, None)
        self._scope_round_cache.pop(scope_round, None)
        self._scope_round_hashes.pop(scope_round, None)

    def _maybe_schedule_higher_round(self, completed_scope_round: int) -> None:
        """Schedule a higher-level round after enough current-level rounds have finished."""
        if not self._higher_scope_enabled():
            return
        handler = self._get_scope_round_handler(self.higher_scope_name)
        if handler is None:
            return
        interval = getattr(self.higher_scope_config, "rounds_per_scope", 0)
        if interval <= 0:
            return
        if completed_scope_round % interval == 0:
            higher_round = completed_scope_round // interval
            if any(sr == higher_round for sr, _ in handler.round_queue):
                return
            handler.round_queue.append((higher_round, completed_scope_round))
            hierarchy_logger.info(
                "%s layer scheduled round %d after %s round %d (interval=%d)",
                self._higher_scope_label_upper(),
                higher_round,
                self._scope_label_lower(),
                completed_scope_round,
                interval,
            )

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
