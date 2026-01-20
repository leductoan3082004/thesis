"""Utilities that implement state-level aggregation mechanics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from secure_aggregation.node import ECM
from secure_aggregation.state.config import StateAggregationConfig
from secure_aggregation.storage.model_store import (
    AnchorScope,
    BlockchainInterface,
    IPFSInterface,
    compute_model_hash,
    verify_model_hash,
)
from secure_aggregation.utils import get_logger

logger = get_logger("state_aggregation")

STATE_SIGNAL_PREFIX = "signal::state::"


class StateAggregationError(RuntimeError):
    """Raised when a state round cannot be completed."""


@dataclass(frozen=True)
class StateClusterModel:
    """Snapshot of a single cluster contribution for a state round."""

    cluster_id: str
    cid: str
    hash: str
    round_idx: int
    received_at: float
    source_node: Optional[str] = None


@dataclass(frozen=True)
class StateDigest:
    """Digest advertised by a central node after merging ECMs."""

    node_id: str
    state_id: str
    state_round: int
    cluster_round: Optional[int]
    model_hash: str
    model_cid: Optional[str]
    received_at: float


def build_state_signal_cid(state_id: str, state_round: int, node_id: str) -> str:
    """Construct the CID prefix used for state digest signals."""
    return f"{STATE_SIGNAL_PREFIX}{state_id}::{state_round}::{node_id}"


def parse_state_digest_signal(ecm: ECM) -> Optional[StateDigest]:
    """Parse a bridge ECM signal into a structured state digest."""
    if not ecm.is_signal or not ecm.cid.startswith(STATE_SIGNAL_PREFIX):
        return None
    parts = ecm.cid.split("::")
    if len(parts) < 5:
        return None
    state_id = parts[2]
    try:
        state_round = int(parts[3])
    except ValueError:
        return None
    node_id = parts[4]
    cluster_round: Optional[int] = None
    model_cid: Optional[str] = None
    if ecm.convergence_data_id:
        try:
            payload = json.loads(ecm.convergence_data_id)
        except json.JSONDecodeError:
            payload = {}
        cluster_round_val = payload.get("cluster_round")
        if cluster_round_val is not None:
            try:
                cluster_round = int(cluster_round_val)
            except (TypeError, ValueError):
                cluster_round = None
        if payload.get("model_cid"):
            model_cid = str(payload["model_cid"])
    return StateDigest(
        node_id=node_id,
        state_id=state_id,
        state_round=state_round,
        cluster_round=cluster_round,
        model_hash=ecm.hash,
        model_cid=model_cid,
        received_at=ecm.received_at,
    )


class StateAggregator:
    """Helper that fetches ECMs, merges them, and publishes the state model."""

    def __init__(
        self,
        config: StateAggregationConfig,
        ipfs: Optional[IPFSInterface],
        blockchain: Optional[BlockchainInterface],
    ) -> None:
        self.config = config
        self.ipfs = ipfs
        self.blockchain = blockchain

    def build_snapshot(
        self,
        ecms: Iterable[ECM],
        required_clusters: Sequence[str],
        target_round: Optional[int],
    ) -> Tuple[Mapping[str, StateClusterModel], List[str]]:
        """
        Deduplicate ECMs per cluster and return the latest contributions.

        Args:
            ecms: Iterable of ECM records from bridge buffers.
            required_clusters: Expected cluster identifiers.
            target_round: Cluster round index that should be represented.
        """
        snapshot: dict[str, StateClusterModel] = {}
        required = set(required_clusters)
        for ecm in ecms:
            if ecm.is_signal:
                continue
            cluster_id = ecm.source_cluster
            if not cluster_id or cluster_id not in required:
                continue
            if target_round is not None and ecm.round_idx not in (-1, target_round):
                continue
            prev = snapshot.get(cluster_id)
            if prev is None or ecm.received_at > prev.received_at:
                snapshot[cluster_id] = StateClusterModel(
                    cluster_id=cluster_id,
                    cid=ecm.cid,
                    hash=ecm.hash,
                    round_idx=ecm.round_idx,
                    received_at=ecm.received_at,
                    source_node=getattr(ecm, "source_node", None),
                )
        missing = [cluster_id for cluster_id in required if cluster_id not in snapshot]
        return snapshot, missing

    def fetch_models(
        self,
        snapshot: Mapping[str, StateClusterModel],
        fallback_lookup: Optional[
            Callable[[str, int], Optional[Tuple[str, str]]]
        ] = None,
    ) -> Mapping[str, np.ndarray]:
        """Fetch and verify models referenced by the snapshot."""
        models: dict[str, np.ndarray] = {}
        for cluster_id, entry in snapshot.items():
            model = self._fetch_from_ipfs(entry.cid, entry.hash)
            if model is None and fallback_lookup is not None:
                alt = fallback_lookup(cluster_id, entry.round_idx)
                if alt:
                    model = self._fetch_from_ipfs(*alt)
            if model is None:
                raise StateAggregationError(
                    f"Missing model for cluster {cluster_id} (cid={entry.cid[:8]}...)"
                )
            models[cluster_id] = model
        return models

    def _fetch_from_ipfs(self, cid: Optional[str], expected_hash: Optional[str]) -> Optional[np.ndarray]:
        if cid is None or not expected_hash or self.ipfs is None:
            return None
        try:
            model = self.ipfs.get(cid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch state model cid=%s: %s", cid[:16], exc)
            return None
        if model is None:
            return None
        if not verify_model_hash(model, expected_hash):
            logger.warning(
                "Hash mismatch while fetching cid=%s (expected %s)",
                cid[:16],
                expected_hash[:16],
            )
            return None
        return model

    def merge_models(self, models: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compute a simple average across all cluster contributions."""
        if not models:
            raise StateAggregationError("Cannot merge zero models at state level")
        stacked = np.stack([arr for _, arr in sorted(models.items())], axis=0)
        merged = np.mean(stacked, axis=0)
        return merged.astype(np.float32)

    def publish_state_model(
        self,
        state_model: np.ndarray,
        state_round: int,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Publish the aggregated state model to IPFS/blockchain.

        Returns:
            Tuple of (cid, hash, data_id).
        """
        if self.ipfs is None:
            raise StateAggregationError("IPFS client not configured for state aggregation")
        logger.info(
            "Publishing state model round=%d params=%d",
            state_round,
            int(state_model.size),
        )
        cid = self.ipfs.add(state_model)
        hash_val = compute_model_hash(state_model)
        data_id: Optional[str] = None
        if self.blockchain is not None:
            try:
                data_id = self.blockchain.anchor(
                    self.config.state_id,
                    state_round,
                    cid,
                    hash_val,
                    scope=AnchorScope.STATE,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to anchor state model round=%s: %s", state_round, exc)
        return cid, hash_val, data_id

    def get_anchor(self, state_round: int) -> Optional[Tuple[str, str]]:
        """Fetch the anchored state model reference if available."""
        if self.blockchain is None:
            return None
        return self.blockchain.get_anchor(self.config.state_id, state_round, scope=AnchorScope.STATE)
