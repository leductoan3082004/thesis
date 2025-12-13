"""
Simple HTTP registry service for model anchors.

This acts as a centralized ledger for storing model anchor records,
simulating blockchain functionality. Can be deployed as a Docker
container alongside the IPFS nodes.

Run with: uvicorn secure_aggregation.storage.registry_service:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelAnchor:
    """Anchor record for a cluster model."""

    cluster_id: str
    round_num: int
    cid: str
    hash: str


class AnchorRequest(BaseModel):
    """Request model for creating an anchor."""

    cluster_id: str
    round_num: int
    cid: str
    hash: str


class AnchorResponse(BaseModel):
    """Response model for anchor data."""

    cluster_id: str
    round_num: int
    cid: str
    hash: str


class RegistryStore:
    """Thread-safe storage for model anchors with optional persistence."""

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else None
        self._registry: Dict[str, Dict[int, ModelAnchor]] = {}
        self._lock = threading.Lock()

        if self._storage_path:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            if self._storage_path.exists():
                self._load()
            else:
                self._save()

    def _load(self) -> None:
        """Load registry from file."""
        try:
            data = json.loads(self._storage_path.read_text())
            for cluster_id, rounds in data.items():
                self._registry[cluster_id] = {}
                for round_str, anchor_data in rounds.items():
                    self._registry[cluster_id][int(round_str)] = ModelAnchor(**anchor_data)
            logger.info(f"Loaded {sum(len(r) for r in self._registry.values())} anchors from {self._storage_path}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load registry: {e}")
            self._registry = {}

    def _save(self) -> None:
        """Save registry to file."""
        if not self._storage_path:
            return
        data: Dict[str, Dict[str, Dict]] = {}
        for cluster_id, rounds in self._registry.items():
            data[cluster_id] = {}
            for round_num, anchor in rounds.items():
                data[cluster_id][str(round_num)] = asdict(anchor)
        self._storage_path.write_text(json.dumps(data, indent=2))

    def add_anchor(self, anchor: ModelAnchor) -> None:
        """Add or update an anchor."""
        with self._lock:
            if anchor.cluster_id not in self._registry:
                self._registry[anchor.cluster_id] = {}
            self._registry[anchor.cluster_id][anchor.round_num] = anchor
            self._save()
            logger.info(f"Anchored: cluster={anchor.cluster_id}, round={anchor.round_num}, cid={anchor.cid[:16]}...")

    def get_anchor(self, cluster_id: str, round_num: int) -> Optional[ModelAnchor]:
        """Get anchor for specific cluster and round."""
        with self._lock:
            cluster_rounds = self._registry.get(cluster_id, {})
            return cluster_rounds.get(round_num)

    def get_latest_anchor(self, cluster_id: str) -> Optional[ModelAnchor]:
        """Get most recent anchor for a cluster."""
        with self._lock:
            cluster_rounds = self._registry.get(cluster_id, {})
            if not cluster_rounds:
                return None
            latest_round = max(cluster_rounds.keys())
            return cluster_rounds[latest_round]

    def get_all_anchors(self, cluster_id: str) -> Dict[int, ModelAnchor]:
        """Get all anchors for a cluster."""
        with self._lock:
            return dict(self._registry.get(cluster_id, {}))

    def clear(self) -> None:
        """Clear all anchors."""
        with self._lock:
            self._registry.clear()
            self._save()


# Global store instance - configured via environment variable or default.
import os

STORAGE_PATH = os.environ.get("REGISTRY_STORAGE_PATH")
store = RegistryStore(storage_path=STORAGE_PATH)

app = FastAPI(
    title="Model Anchor Registry",
    description="Centralized registry for federated learning model anchors",
    version="1.0.0",
)


@app.post("/anchors", status_code=201)
async def create_anchor(request: AnchorRequest) -> AnchorResponse:
    """Create or update a model anchor."""
    anchor = ModelAnchor(
        cluster_id=request.cluster_id,
        round_num=request.round_num,
        cid=request.cid,
        hash=request.hash,
    )
    store.add_anchor(anchor)
    return AnchorResponse(**asdict(anchor))


@app.get("/anchors/{cluster_id}/latest")
async def get_latest_anchor(cluster_id: str) -> AnchorResponse:
    """Get most recent anchor for a cluster."""
    anchor = store.get_latest_anchor(cluster_id)
    if anchor is None:
        raise HTTPException(status_code=404, detail="No anchors found for cluster")
    return AnchorResponse(**asdict(anchor))


@app.get("/anchors/{cluster_id}/{round_num}")
async def get_anchor(cluster_id: str, round_num: int) -> AnchorResponse:
    """Get anchor for specific cluster and round."""
    anchor = store.get_anchor(cluster_id, round_num)
    if anchor is None:
        raise HTTPException(status_code=404, detail="Anchor not found")
    return AnchorResponse(**asdict(anchor))


@app.get("/anchors/{cluster_id}")
async def list_anchors(cluster_id: str) -> Dict[int, AnchorResponse]:
    """List all anchors for a cluster."""
    anchors = store.get_all_anchors(cluster_id)
    return {k: AnchorResponse(**asdict(v)) for k, v in anchors.items()}


@app.delete("/anchors", status_code=204)
async def clear_anchors() -> None:
    """Clear all anchors (for testing)."""
    store.clear()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
