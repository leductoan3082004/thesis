"""
Inter-cluster aggregator that extends SAP with cross-cluster model merging.

This module provides:
1. ECM collection from bridge nodes during Round 2
2. Inter-cluster merge after SAP completes
3. Model publishing to IPFS and blockchain anchoring
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from secure_aggregation.node import ECM, ECMBuffer
from secure_aggregation.protocol import InterClusterMerger, MergeConfig
from secure_aggregation.storage.model_store import (
    BlockchainInterface,
    IPFSInterface,
    compute_model_hash,
    verify_model_hash,
)
from secure_aggregation.utils import get_logger

logger = get_logger("inter_cluster_aggregator")


class InterClusterAggregator:
    """
    Aggregator extension that performs inter-cluster merge after SAP.

    After intra-cluster secure aggregation completes, this component:
    1. Collects ECMs from bridge nodes
    2. Fetches and verifies neighbor cluster models from IPFS
    3. Applies adaptive clipping and weighted merge
    4. Publishes merged model to IPFS
    5. Anchors model reference on blockchain
    """

    def __init__(
        self,
        cluster_id: str,
        ipfs: Optional[IPFSInterface] = None,
        blockchain: Optional[BlockchainInterface] = None,
        merge_config: Optional[MergeConfig] = None,
    ) -> None:
        self.cluster_id = cluster_id
        self.ipfs = ipfs
        self.blockchain = blockchain
        self.merge_config = merge_config or MergeConfig()
        self.merger = InterClusterMerger(self.merge_config)
        self.ecm_buffer = ECMBuffer(freshness_window=300.0)
        self.current_round = 0
        self.last_data_id: Optional[str] = None

    def receive_ecms(self, node_id: str, ecms: List[ECM]) -> None:
        """Receive ECMs from a bridge node."""
        for ecm in ecms:
            self.ecm_buffer.add(ecm)
            logger.debug(f"Received ECM from {node_id}: cid={ecm.cid[:8]}...")

    def merge_with_neighbors(
        self,
        intra_cluster_model: np.ndarray,
        max_neighbors: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Merge intra-cluster model with verified neighbor models.

        Args:
            intra_cluster_model: Model from SAP (theta_local).

        Returns:
            Tuple of (merged_model, list of successfully merged neighbor CIDs).
        """
        if self.ipfs is None:
            logger.warning("No IPFS client configured, skipping inter-cluster merge")
            return intra_cluster_model.copy(), []

        unique_ecms = self.ecm_buffer.get_unique_cids()
        if not unique_ecms:
            logger.info("No ECMs received, using intra-cluster model only")
            return intra_cluster_model.copy(), []

        selected_ecms = self._select_neighbors(unique_ecms, max_neighbors)

        logger.info(f"Processing {len(selected_ecms)} unique ECMs from neighbor clusters")

        verified_models: List[np.ndarray] = []
        merged_cids: List[str] = []

        for cid, expected_hash in selected_ecms.items():
            model = None
            try:
                model = self.ipfs.get(cid)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to fetch model with CID %s: %s",
                    cid[:8],
                    exc,
                )
            if model is None:
                logger.warning(f"Failed to fetch model with CID {cid[:8]}...")
                continue

            if not verify_model_hash(model, expected_hash):
                logger.warning(f"Hash mismatch for CID {cid[:8]}..., skipping")
                continue

            verified_models.append(model)
            merged_cids.append(cid)
            logger.debug(f"Verified model CID {cid[:8]}...")

        if not verified_models:
            logger.warning("No valid neighbor models, using intra-cluster model only")
            return intra_cluster_model.copy(), []

        logger.info(f"Merging {len(verified_models)} verified neighbor models")
        merged_model = self.merger.merge(intra_cluster_model, verified_models)

        return merged_model, merged_cids

    def publish_model(
        self,
        model: np.ndarray,
        round_num: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Publish model to IPFS and anchor on blockchain.

        Args:
            model: Model to publish.
            round_num: Current training round.

        Returns:
            Tuple of (cid, hash) or (None, None) if publishing fails.
        """
        if self.ipfs is None:
            logger.warning("No IPFS client configured, skipping publish")
            return None, None

        publish_start = time.monotonic()
        logger.info(
            "Publishing merged model to IPFS: cluster=%s round=%d params=%d",
            self.cluster_id,
            round_num,
            int(model.size),
        )
        cid = self.ipfs.add(model)
        publish_elapsed = time.monotonic() - publish_start
        logger.info(
            "Published model to IPFS: cluster=%s round=%d cid=%s... (%.2fs)",
            self.cluster_id,
            round_num,
            cid[:16],
            publish_elapsed,
        )
        if hasattr(self.ipfs, "provide"):
            try:
                self.ipfs.provide(cid)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to announce CID %s to DHT", cid[:16])
        model_hash = compute_model_hash(model)

        data_id: Optional[str] = None
        if self.blockchain is not None:
            try:
                anchor_start = time.monotonic()
                logger.info(
                    "Anchoring merged model on blockchain: cluster=%s round=%d cid=%s...",
                    self.cluster_id,
                    round_num,
                    cid[:16],
                )
                data_id = self.blockchain.anchor(self.cluster_id, round_num, cid, model_hash)
                logger.info(
                    "Anchored model on blockchain: cluster=%s round=%d data_id=%s (%.2fs)",
                    self.cluster_id,
                    round_num,
                    data_id or "N/A",
                    time.monotonic() - anchor_start,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"~ BLOCKCHAIN ~ BLOCKCHAIN ~ BLOCKCHAIN ~ Failed to anchor "
                    f"cluster={self.cluster_id}, round={round_num}: {exc}"
                )
        self.last_data_id = data_id

        return cid, model_hash

    def get_neighbor_model(
        self,
        neighbor_cluster_id: str,
        round_num: int,
    ) -> Optional[np.ndarray]:
        """
        Fetch verified neighbor cluster model from blockchain/IPFS.

        Args:
            neighbor_cluster_id: ID of neighbor cluster.
            round_num: Training round to fetch.

        Returns:
            Verified model or None if not available/invalid.
        """
        if self.blockchain is None or self.ipfs is None:
            return None

        anchor = self.blockchain.get_anchor(neighbor_cluster_id, round_num)
        if anchor is None:
            logger.debug(f"No anchor for {neighbor_cluster_id} round {round_num}")
            return None

        cid, expected_hash = anchor
        model = self.ipfs.get(cid)
        if model is None:
            logger.warning(f"Failed to fetch model for {neighbor_cluster_id}")
            return None

        if not verify_model_hash(model, expected_hash):
            logger.warning(f"Hash mismatch for {neighbor_cluster_id} model")
            return None

        return model

    def process_round(
        self,
        intra_cluster_model: np.ndarray,
        round_num: int,
    ) -> Tuple[np.ndarray, Optional[str], Optional[str]]:
        """
        Complete inter-cluster processing for a round.

        This is the main entry point after SAP completes:
        1. Merge with neighbor models from ECM buffer
        2. Publish merged model
        3. Clear ECM buffer for next round

        Args:
            intra_cluster_model: Model from SAP.
            round_num: Current training round.

        Returns:
            Tuple of (merged_model, cid, hash).
        """
        self.current_round = round_num

        merged_model, merged_cids = self.merge_with_neighbors(
            intra_cluster_model,
            max_neighbors=self.merge_config.max_neighbors,
        )
        logger.info(
            f"Round {round_num}: merged with {len(merged_cids)} neighbor models, "
            f"clipping threshold={self.merger.get_current_threshold():.4f}"
        )

        cid, model_hash = self.publish_model(merged_model, round_num)
        self.ecm_buffer.clear()

        return merged_model, cid, model_hash

    def reset_for_next_round(self) -> None:
        """Reset state for next aggregation round."""
        self.ecm_buffer.clear()
        self.current_round += 1
    def _select_neighbors(
        self,
        ecms: Dict[str, str],
        max_neighbors: Optional[int],
    ) -> Dict[str, str]:
        """Select subset of ECMs for merging based on historical usage."""
        if not ecms:
            return ecms
        if max_neighbors is None or max_neighbors <= 0 or len(ecms) <= max_neighbors:
            return ecms
        history: Dict[str, int] = getattr(self.merge_config, "neighbor_history", {})
        sorted_neighbors = sorted(
            ecms.keys(),
            key=lambda cid: (history.get(cid, 0), cid),
        )
        selected = sorted_neighbors[:max_neighbors]
        for cid in selected:
            history[cid] = history.get(cid, 0) + 1
        self.merge_config.neighbor_history = history
        return {cid: ecms[cid] for cid in selected}
