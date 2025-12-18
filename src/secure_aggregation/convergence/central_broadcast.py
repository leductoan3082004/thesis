"""Helpers to publish and consume central-clique metadata via the blockchain registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

from secure_aggregation.storage.model_store import BlockchainInterface, ModelAnchor

CENTRAL_METADATA_CLUSTER_ID = "__central_metadata__"
CENTRAL_HEALTH_CLUSTER_ID = "__central_checker_health__"
GLOBAL_CONVERGENCE_CLUSTER_ID = "__global_convergence__"


@dataclass
class CentralMetadata:
    """Topology metadata shared with all cliques."""

    central_clique_idx: int
    central_nodes: list[str]
    checker_candidates: list[str]
    total_cliques: int
    cluster_ids: list[str]
    version: int = 0

    def to_payload(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_payload(cls, payload: str) -> "CentralMetadata":
        data = json.loads(payload)
        return cls(
            central_clique_idx=data["central_clique_idx"],
            central_nodes=list(data["central_nodes"]),
            checker_candidates=list(data["checker_candidates"]),
            total_cliques=data.get("total_cliques", len(data.get("cluster_ids", []))),
            cluster_ids=list(data.get("cluster_ids", [])),
            version=data.get("version", 0),
        )


@dataclass
class CheckerHealth:
    """Health advertisement from a checker candidate."""

    checker_id: str
    round_idx: int
    priority: int
    alive: bool = True

    def to_payload(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_anchor(cls, anchor: ModelAnchor) -> "CheckerHealth":
        return cls(**json.loads(anchor.cid))


def publish_central_metadata(blockchain: BlockchainInterface, metadata: CentralMetadata) -> Optional[str]:
    """Anchor central metadata on the blockchain registry."""
    payload = metadata.to_payload()
    return blockchain.anchor(
        CENTRAL_METADATA_CLUSTER_ID,
        metadata.version,
        cid=payload,
        hash_val=str(metadata.version),
    )


def fetch_central_metadata(blockchain: Optional[BlockchainInterface]) -> Optional[CentralMetadata]:
    """Retrieve the most recent central metadata."""
    if blockchain is None:
        return None
    anchor = blockchain.get_latest_anchor(CENTRAL_METADATA_CLUSTER_ID)
    if anchor is None:
        return None
    return CentralMetadata.from_payload(anchor.cid)


def publish_checker_health(blockchain: BlockchainInterface, health: CheckerHealth) -> None:
    """Anchor checker health information."""
    payload = health.to_payload()
    blockchain.anchor(
        CENTRAL_HEALTH_CLUSTER_ID,
        health.round_idx,
        cid=payload,
        hash_val=str(health.priority),
    )


def fetch_checker_health(blockchain: Optional[BlockchainInterface]) -> list[CheckerHealth]:
    """Fetch all checker health anchors."""
    if blockchain is None:
        return []
    latest = blockchain.get_latest_anchor(CENTRAL_HEALTH_CLUSTER_ID)
    if latest is None:
        return []
    return [CheckerHealth.from_anchor(latest)]


def publish_global_convergence(
    blockchain: BlockchainInterface,
    round_idx: int,
    metadata: dict,
) -> Optional[str]:
    """Commit global convergence metadata and return data identifier."""
    return blockchain.commit_metadata(GLOBAL_CONVERGENCE_CLUSTER_ID, round_idx, metadata)
