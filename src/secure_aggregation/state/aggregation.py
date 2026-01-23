"""Utilities that implement state-level aggregation mechanics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

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

    def get_anchor(
        self,
        state_round: int,
        *,
        suppress_not_found_log: bool = False,
    ) -> Optional[Tuple[str, str]]:
        """Fetch the anchored state model reference if available."""
        if self.blockchain is None:
            return None
        return self.blockchain.get_anchor(
            self.config.state_id,
            state_round,
            scope=AnchorScope.STATE,
            suppress_not_found_log=suppress_not_found_log,
        )
