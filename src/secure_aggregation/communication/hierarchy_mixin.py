"""Hierarchy-specific helpers for node_service."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from secure_aggregation.communication.bridge_service import STATE_SIGNAL_PREFIX
from secure_aggregation.node import ECM, ECMBuffer
from secure_aggregation.state import (
    NationAggregationConfig,
    StateAggregationApproach,
    StateAggregationConfig,
    StateAggregationError,
    StateAggregator,
    StateClusterModel,
)
from secure_aggregation.storage.model_store import AnchorScope, compute_model_hash, verify_model_hash
from secure_aggregation.utils import get_logger

hierarchy_logger = get_logger("hierarchy")

STATE_SIGNAL_CID_PREFIX = "signal::state::"


@dataclass
class StateDigest:
    """State-level model digest shared between central candidates."""

    node_id: str
    state_id: str
    state_round: int
    cluster_round: int
    model_hash: str
    model_cid: Optional[str]
    received_at: float


def build_state_signal_cid(state_id: Optional[str], state_round: int, node_id: str) -> str:
    """Create a recognizable CID for state digest signaling over the bridge."""
    safe_state_id = state_id or "unknown"
    return f"{STATE_SIGNAL_CID_PREFIX}{safe_state_id}::{state_round}::{node_id}"


def parse_state_digest_signal(ecm: ECM) -> Optional[StateDigest]:
    """Convert a bridge ECM carrying a state digest signal into a StateDigest."""
    if not ecm.is_signal or not ecm.source_cluster:
        return None
    if not ecm.source_cluster.startswith(STATE_SIGNAL_PREFIX):
        return None
    state_id = ecm.source_cluster[len(STATE_SIGNAL_PREFIX) :]
    if not state_id:
        return None
    state_round = ecm.round_idx
    if state_round < 0:
        return None
    cluster_round = state_round
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
            hierarchy_logger.debug("Malformed state digest payload: %s", ecm.convergence_data_id)
    if not origin_node and ecm.cid.startswith(STATE_SIGNAL_CID_PREFIX):
        parts = ecm.cid.split("::")
        if len(parts) >= 5:
            origin_node = parts[-1]
    if not origin_node:
        hierarchy_logger.debug("Unable to determine origin node for state digest cid=%s", ecm.cid)
        return None
    return StateDigest(
        node_id=origin_node,
        state_id=state_id,
        state_round=state_round,
        cluster_round=cluster_round,
        model_hash=ecm.hash,
        model_cid=None,
        received_at=ecm.received_at,
    )


class HierarchyMixin:
    """Mixin providing hierarchy (state/nation/...) orchestration helpers."""

    def _load_state_config(self) -> StateAggregationConfig:
        """Load state aggregation configuration and apply training defaults."""
        system_cfg = self.system_config or {}
        defaults = dict(system_cfg.get("hierarchy_defaults") or {})
        state_section = dict(defaults)
        state_section.update(system_cfg.get("state_aggregation") or {})
        config = StateAggregationConfig.from_mapping(state_section)
        node_scope_id = getattr(self, "state_id", None)
        if node_scope_id:
            config.state_id = str(node_scope_id)
        config.apply_training_defaults(None)
        return config

    def _load_nation_config(self) -> NationAggregationConfig:
        """Load scheduling config for nation-level aggregation rounds."""
        nation_section = (self.system_config or {}).get("nation_aggregation")
        return NationAggregationConfig.from_mapping(nation_section)

    def _configure_scope_layer(self) -> None:
        """Determine which nodes act as state aggregators based on metadata."""
        if not self.state_config.enabled:
            if self.state_ecm_buffer is not None:
                self.state_ecm_buffer = None
                self._update_bridge_hooks()
                self._ensure_bridge_stack()
            self.state_candidates = []
            self.is_state_candidate = False
            return
        if not self.central_metadata:
            return
        if self.state_config.approach == StateAggregationApproach.RING_STAR:
            candidates = list(self.central_metadata.central_nodes)
        else:
            candidates = list(self.central_metadata.central_nodes)
        candidates = sorted(dict.fromkeys(candidates))
        previous_candidates = self.state_candidates
        self.state_candidates = candidates
        was_candidate = self.is_state_candidate
        self.is_state_candidate = self.node_id in self.state_candidates
        if self.is_state_candidate and self.state_ecm_buffer is None:
            freshness = float(
                max(
                    self.inter_cluster_config.get("freshness_window", 300.0),
                    self.state_config.collection_timeout_seconds * 2,
                )
            )
            self.state_ecm_buffer = ECMBuffer(freshness_window=freshness)
            self._update_bridge_hooks()
            self._ensure_bridge_stack()
        elif not self.is_state_candidate and self.state_ecm_buffer is not None:
            self.state_ecm_buffer = None
            self._update_bridge_hooks()
            self._ensure_bridge_stack()
        if self.is_state_candidate and self.state_aggregator is None and self.ipfs is not None:
            self.state_aggregator = StateAggregator(self.state_config, self.ipfs, self.blockchain)
        if (
            candidates
            and candidates != previous_candidates
            and self.is_state_candidate
            and not was_candidate
        ):
            hierarchy_logger.info(
                "Node %s joined state aggregator pool with %d candidates",
                self.node_id,
                len(candidates),
            )

    def _scope_layer_enabled(self) -> bool:
        return bool(self.state_config.enabled and self.state_config.rounds_per_state > 0)

    def _higher_scope_enabled(self) -> bool:
        return bool(self.nation_config.enabled and self.nation_config.rounds_per_nation > 0)

    def _scope_round_budget(self) -> Optional[int]:
        """Estimate how many state rounds can occur over the training horizon."""
        if not self._scope_layer_enabled():
            return None
        interval = max(1, self.state_config.rounds_per_state)
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
        if (
            not self._scope_layer_enabled()
            or not self.blockchain
            or not self.ipfs
            or not self.state_config.state_id
            or not self.model
        ):
            return
        interval = max(1, self.state_config.rounds_per_state)
        completed_rounds = (round_idx + 1) // interval
        while self._last_applied_state_round < completed_rounds:
            target_round = self._last_applied_state_round + 1
            try:
                anchor = self.blockchain.get_anchor(
                    self.state_config.state_id,
                    target_round,
                    scope=AnchorScope.STATE,
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
            hierarchy_logger.info(
                "State model round %d detected on blockchain (cid=%s..., hash=%s...)",
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
            self._apply_model_tensor(state_model)
            self._last_model_cid = cid
            self._last_model_hash = expected_hash
            self._last_model_data_id = None
            self._last_applied_state_round = target_round
            hierarchy_logger.info(
                "Applied STATE ROUND %d model (cid=%s...); next cluster rounds will start from this baseline",
                target_round,
                cid[:12],
            )
            self._prime_convergence_tracker_state()

    def _maybe_schedule_scope_round(self, round_idx: int) -> None:
        """Trigger state aggregation when the configured interval elapses."""
        if not self._scope_layer_enabled():
            hierarchy_logger.debug("Skipping state round scheduling: state layer disabled")
            return
        interval = self.state_config.rounds_per_state
        if interval <= 0:
            return
        due = (round_idx + 1) % interval == 0
        hierarchy_logger.debug(
            "State round scheduler invoked for cluster round %d (interval=%d, due=%s)",
            round_idx + 1,
            interval,
            due,
        )
        if not due:
            return
        state_round = (round_idx + 1) // interval
        if any(sr == state_round for sr, _ in self._state_round_queue):
            hierarchy_logger.debug(
                "State round %d already scheduled; skipping duplicate", state_round
            )
            return
        hierarchy_logger.info(
            "Scheduling state round %d to run after completion of cluster round %d",
            state_round,
            round_idx + 1,
        )
        self._state_round_queue.append((state_round, round_idx))

    def _run_next_scope_round(self) -> Optional[int]:
        """Execute the next scheduled state round if available."""
        if not self._scope_layer_enabled():
            self._state_round_queue.clear()
            return None
        if not self._state_round_queue:
            return None
        state_round, cluster_round = self._state_round_queue.popleft()
        total_state_rounds = self._scope_round_budget()
        label = f"State Round {state_round}/{total_state_rounds or '?'}"
        hierarchy_logger.info("\n" + "=" * 60)
        hierarchy_logger.info("%s (triggered after cluster round %d)", label, cluster_round + 1)
        hierarchy_logger.info("=" * 60)
        self._dispatch_scope_artifacts(state_round, cluster_round)
        completed = self._execute_scope_round(state_round, cluster_round)
        if completed and self.is_state_candidate:
            self._maybe_schedule_higher_round(state_round)
        return cluster_round

    def _execute_scope_round(self, state_round: int, cluster_round: int) -> bool:
        """Collect ECMs, merge models, and broadcast digest for a state round."""
        if not self._scope_layer_enabled():
            return False
        if state_round not in self._state_rounds_logged:
            total_state_rounds = self._scope_round_budget()
            label = f"/{total_state_rounds}" if total_state_rounds else ""
            self._state_rounds_logged.add(state_round)
        can_aggregate = (
            self.state_aggregator is not None
            and self.state_ecm_buffer is not None
            and self.central_metadata is not None
        )
        if not can_aggregate:
            return self._wait_for_scope_anchor_observer(state_round)
        if state_round in self._state_round_cache:
            return True
        deadline = time.time() + max(1.0, float(self.state_config.collection_timeout_seconds))
        snapshot: Dict[str, StateClusterModel] = {}
        missing: List[str] = []
        while time.time() < deadline:
            ecms = self.state_ecm_buffer.get_fresh_ecms()
            snapshot, missing = self.state_aggregator.build_snapshot(
                ecms,
                self.central_metadata.cluster_ids,
                cluster_round,
            )
            if not missing:
                break
            hierarchy_logger.debug(
                "State round %d waiting for ECMs from clusters: %s",
                state_round,
                ", ".join(sorted(missing)),
            )
            time.sleep(1.0)
        if missing:
            hierarchy_logger.warning(
                "State round %d missing ECMs from clusters: %s",
                state_round,
                ", ".join(sorted(missing)),
            )
            return False
        try:
            models = self.state_aggregator.fetch_models(snapshot, fallback_lookup=self._lookup_lower_scope_anchor)
            merged_model = self.state_aggregator.merge_models(models)
        except StateAggregationError as exc:
            hierarchy_logger.error("State aggregation failed for round %d: %s", state_round, exc)
            return False
        model_hash = compute_model_hash(merged_model)
        self._state_round_cache[state_round] = merged_model
        self._state_round_hashes[state_round] = model_hash
        self._broadcast_scope_digest(state_round, cluster_round, model_hash)
        local_digest = StateDigest(
            node_id=self.node_id,
            state_id=self.state_config.state_id,
            state_round=state_round,
            cluster_round=cluster_round,
            model_hash=model_hash,
            model_cid=None,
            received_at=time.time(),
        )
        self._record_scope_digest(local_digest, local=True)
        self._await_scope_digest_consensus(state_round)
        self._verify_scope_digest_consensus(state_round, model_hash)
        return True

    def _dispatch_scope_artifacts(self, state_round: int, cluster_round: int) -> None:
        """Have bridge nodes forward their latest ECM to state aggregators when a round starts."""
        if not self._scope_layer_enabled():
            return
        target_addresses: Dict[str, str] = {}
        if not self.central_neighbor_addresses:
            hierarchy_logger.debug("No central neighbor addresses available for state ECM dispatch")
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
                "State ECM dispatch skipped for round %d: no central targets resolved", state_round
            )
            return
        target_list = ", ".join(f"{node}@{addr}" for node, addr in sorted(target_addresses.items()))
        hierarchy_logger.info("State ECM dispatch targets for round %d: %s", state_round, target_list)
        if not self._last_model_cid or not self._last_model_hash:
            hierarchy_logger.info(
                "Skipping state ECM dispatch for round %d: missing latest model reference (cid=%s, hash=%s)",
                state_round,
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
                "Cannot dispatch state ECM for round %d: bridge client unavailable",
                state_round,
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
            "Dispatched state ECM for state round %d to %d/%d candidates",
            state_round,
            accepted,
            len(targets),
        )

    def _wait_for_scope_anchor_observer(self, state_round: int) -> bool:
        """Wait for another node to commit the state round before continuing."""
        timeout = (
            float(self.state_config.collection_timeout_seconds)
            + float(self.state_config.consensus_timeout_seconds)
            + float(self.state_config.commit_timeout_seconds)
        )
        hierarchy_logger.info(
            "Waiting up to %.0fs for state round %d anchor to appear",
            timeout,
            state_round,
        )
        anchor = self._wait_for_scope_anchor(state_round, timeout)
        if anchor:
            hierarchy_logger.info(
                "Observed state round %d anchor committed elsewhere (cid=%s...)",
                state_round,
                anchor[0][:8],
            )
            return True
        hierarchy_logger.warning(
            "State round %d anchor not observed after waiting %.0fs; continuing cluster training",
            state_round,
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

    def _broadcast_scope_digest(self, state_round: int, cluster_round: int, model_hash: str) -> None:
        """Share this node's state digest with other central candidates."""
        hierarchy_logger.info(
            "Broadcasting state digest round=%d to other central nodes",
            state_round,
        )
        if not self._scope_layer_enabled():
            return
        if not self.central_neighbor_addresses:
            hierarchy_logger.debug("No central neighbor addresses available for state digest broadcast")
            return
        if not self._ensure_bridge_client(allow_state_layer=True):
            hierarchy_logger.warning(
                "Cannot broadcast state digest for round %d: bridge client unavailable",
                state_round,
            )
            return
        payload = json.dumps(
            {
                "cluster_round": cluster_round,
                "node_id": self.node_id,
            }
        )
        cid = build_state_signal_cid(self.state_config.state_id, state_round, self.node_id)
        targets = [addr for node_id, addr in self.central_neighbor_addresses.items() if node_id != self.node_id]
        if not targets:
            return
        accepted = self.bridge_client.broadcast_ecm_with_metadata(
            targets,
            f"state::{self.state_config.state_id}",
            state_round,
            cid,
            model_hash,
            metadata=payload,
        )
        detail_targets = targets.copy()
        hierarchy_logger.info(
            "Broadcasted state digest round=%d hash=%s... to %d/%d central nodes (%s)",
            state_round,
            model_hash[:8],
            accepted,
            len(targets),
            ", ".join(detail_targets) if detail_targets else "none",
        )

    def _record_scope_digest(self, digest: StateDigest, local: bool = False) -> None:
        if not self._scope_layer_enabled():
            return
        records = self._state_digest_records.setdefault(digest.state_round, {})
        prev = records.get(digest.node_id)
        if prev and prev.model_hash == digest.model_hash:
            return
        records[digest.node_id] = digest
        if local:
            hierarchy_logger.info(
                "Recorded local state digest round=%d hash=%s...",
                digest.state_round,
                digest.model_hash[:8],
            )
        else:
            hierarchy_logger.info(
                "Observed state digest round=%d from %s hash=%s...",
                digest.state_round,
                digest.node_id,
                digest.model_hash[:8],
            )
        self._maybe_finalize_scope_round(digest.state_round)

    def _await_scope_digest_consensus(self, state_round: int) -> None:
        """Block until digests from all central candidates are observed or timeout elapses."""
        if (
            not self._scope_layer_enabled()
            or not self.state_candidates
            or len(self.state_candidates) <= 1
        ):
            return
        timeout = float(self.state_config.digest_timeout_seconds or 0.0)
        if timeout <= 0:
            return
        deadline = time.time() + timeout
        hierarchy_logger.info(
            "Waiting up to %.0fs for peer state digests (round %d)",
            timeout,
            state_round,
        )
        last_log = 0.0
        while time.time() < deadline:
            self._process_incoming_signals()
            records = self._state_digest_records.get(state_round, {})
            if len(records) >= len(self.state_candidates):
                hierarchy_logger.info(
                    "Received all %d state digests for round %d",
                    len(records),
                    state_round,
                )
                return
            missing = sorted(set(self.state_candidates) - set(records.keys()))
            now = time.time()
            if missing and now - last_log >= 3.0:
                hierarchy_logger.debug(
                    "State round %d waiting for digests from: %s",
                    state_round,
                    ", ".join(missing),
                )
                last_log = now
            time.sleep(1.0)
        remaining = sorted(
            set(self.state_candidates)
            - set((self._state_digest_records.get(state_round) or {}).keys())
        )
        if remaining:
            hierarchy_logger.warning(
                "State round %d timed out waiting for digests from: %s",
                state_round,
                ", ".join(remaining),
            )

    def _verify_scope_digest_consensus(self, state_round: int, local_hash: str) -> None:
        """Ensure all digests agree with the local hash before committing."""
        records = self._state_digest_records.get(state_round, {})
        if not records:
            return
        hashes = {digest.model_hash for digest in records.values()}
        if len(hashes) == 1 and local_hash in hashes:
            hierarchy_logger.info("State round %d digests are consistent (hash=%s...)", state_round, local_hash[:8])
            return
        hierarchy_logger.warning(
            "State round %d digest mismatch detected. Local hash=%s..., peers=%s",
            state_round,
            local_hash[:8],
            ", ".join(h[:8] for h in sorted(hashes)),
        )

    def _maybe_finalize_scope_round(self, state_round: int) -> None:
        if (
            not self._scope_layer_enabled()
            or state_round in self._state_committed_rounds
            or not self.state_candidates
        ):
            return
        records = self._state_digest_records.get(state_round, {})
        if len(records) < len(self.state_candidates):
            return
        hashes = {digest.model_hash for digest in records.values()}
        if len(hashes) != 1:
            hierarchy_logger.warning(
                "State round %d has conflicting digests: %s",
                state_round,
                ", ".join(sorted(hashes)),
            )
            return
        hierarchy_logger.info(
            "State round %d digest consensus reached across %d candidates (hash=%s...)",
            state_round,
            len(records),
            next(iter(hashes))[:8],
        )
        self._try_scope_commit(state_round)

    def _try_scope_commit(self, state_round: int) -> None:
        if (
            not self._scope_layer_enabled()
            or not self.state_candidates
            or not self.state_aggregator
            or state_round in self._state_committed_rounds
        ):
            return
        model = self._state_round_cache.get(state_round)
        if model is None:
            hierarchy_logger.debug(
                "State round %d consensus reached but local model missing; waiting to commit",
                state_round,
            )
            return
        leader_index = state_round % len(self.state_candidates)
        ordered_candidates = [
            self.state_candidates[(leader_index + offset) % len(self.state_candidates)]
            for offset in range(len(self.state_candidates))
        ]
        for candidate in ordered_candidates:
            if candidate == self.node_id:
                anchor = self.state_aggregator.get_anchor(state_round)
                if anchor:
                    self._mark_scope_round_committed(state_round)
                    return
                try:
                    cid, hash_val, data_id = self.state_aggregator.publish_state_model(model, state_round)
                except StateAggregationError as exc:
                    hierarchy_logger.error("State round %d commit failed on %s: %s", state_round, self.node_id, exc)
                    continue
                if cid and hash_val:
                    hierarchy_logger.info(
                        "State round %d committed by %s (cid=%s..., data_id=%s)",
                        state_round,
                        self.node_id,
                        cid[:8],
                        data_id or "N/A",
                    )
                    self._mark_scope_round_committed(state_round)
                    return
            else:
                anchor = self._wait_for_scope_anchor(state_round, self.state_config.commit_timeout_seconds)
                if anchor:
                    hierarchy_logger.info(
                        "State round %d observed anchor (cid=%s...) by peer",
                        state_round,
                        anchor[0][:8],
                    )
                    self._mark_scope_round_committed(state_round)
                    return
        hierarchy_logger.warning("State round %d not anchored after iterating all candidates", state_round)

    def _wait_for_scope_anchor(self, state_round: int, timeout: float) -> Optional[Tuple[str, str]]:
        if self.state_aggregator is None and (self.blockchain is None or not self.state_config.state_id):
            return None
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            anchor: Optional[Tuple[str, str]] = None
            try:
                if self.state_aggregator is not None:
                    anchor = self.state_aggregator.get_anchor(
                        state_round,
                        suppress_not_found_log=True,
                    )
                elif self.blockchain is not None and self.state_config.state_id:
                    anchor = self.blockchain.get_anchor(
                        self.state_config.state_id,
                        state_round,
                        scope=AnchorScope.STATE,
                        suppress_not_found_log=True,
                    )
            except Exception as exc:  # noqa: BLE001
                hierarchy_logger.warning(
                    "Failed to poll state anchor round %d: %s",
                    state_round,
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

    def _mark_scope_round_committed(self, state_round: int) -> None:
        self._state_committed_rounds.add(state_round)
        self._state_digest_records.pop(state_round, None)
        self._state_round_cache.pop(state_round, None)
        self._state_round_hashes.pop(state_round, None)

    def _maybe_schedule_higher_round(self, completed_state_round: int) -> None:
        """Schedule a nation-level round after enough state rounds have finished."""
        if not self._higher_scope_enabled():
            return
        interval = self.nation_config.rounds_per_nation
        if interval <= 0:
            return
        if completed_state_round % interval == 0:
            nation_round = completed_state_round // interval
            if nation_round not in self._nation_round_to_state_round:
                self._nation_round_to_state_round[nation_round] = completed_state_round
                self._pending_nation_rounds.add(nation_round)
                hierarchy_logger.info(
                    "Nation layer scheduled round %d after state round %d (interval=%d)",
                    nation_round,
                    completed_state_round,
                    interval,
                )
        for nation_round in sorted(self._pending_nation_rounds):
            anchor_state_round = self._nation_round_to_state_round.get(nation_round)
            if anchor_state_round is None:
                continue
            self._announce_higher_round(nation_round, anchor_state_round)
            self._pending_nation_rounds.discard(nation_round)

    def _announce_higher_round(self, nation_round: int, source_state_round: int) -> None:
        """Placeholder for nation-level aggregation flow."""
        if not self._higher_scope_enabled():
            return
        hierarchy_logger.info(
            "Nation round %d triggered after state round %d",
            nation_round,
            source_state_round,
        )
