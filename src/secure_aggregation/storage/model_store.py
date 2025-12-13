"""
Storage abstractions for IPFS and Blockchain with mock and real implementations.

These interfaces allow seamless switching between mock implementations
(for testing/development) and real implementations (for production).

Real implementations:
- KuboIPFS: Connects to IPFS Kubo HTTP API
- RegistryBlockchain: Uses a simple HTTP registry service
"""

import hashlib
import io
import json
import logging
import pickle
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx
import numpy as np

logger = logging.getLogger(__name__)


def compute_model_hash(model: np.ndarray) -> str:
    """Compute SHA256 hash of model parameters."""
    return hashlib.sha256(model.tobytes()).hexdigest()


def verify_model_hash(model: np.ndarray, expected_hash: str) -> bool:
    """Verify model integrity by comparing hashes."""
    return compute_model_hash(model) == expected_hash


@dataclass
class ModelAnchor:
    """Blockchain anchor record for a cluster model."""

    cluster_id: str
    round_num: int
    cid: str
    hash: str


class IPFSInterface(ABC):
    """Abstract interface for IPFS-like content-addressed storage."""

    @abstractmethod
    def add(self, model: np.ndarray) -> str:
        """Store model and return content identifier (CID)."""
        pass

    @abstractmethod
    def get(self, cid: str) -> Optional[np.ndarray]:
        """Retrieve model by CID. Returns None if not found."""
        pass

    @abstractmethod
    def exists(self, cid: str) -> bool:
        """Check if model exists in storage."""
        pass


class BlockchainInterface(ABC):
    """Abstract interface for blockchain-like model registry."""

    @abstractmethod
    def anchor(self, cluster_id: str, round_num: int, cid: str, hash_val: str) -> None:
        """Record model reference on-chain."""
        pass

    @abstractmethod
    def get_anchor(self, cluster_id: str, round_num: int) -> Optional[Tuple[str, str]]:
        """Retrieve anchored reference (cid, hash) for cluster/round."""
        pass

    @abstractmethod
    def get_latest_anchor(self, cluster_id: str) -> Optional[ModelAnchor]:
        """Get most recent anchor for a cluster."""
        pass


class MockIPFS(IPFSInterface):
    """
    Mock IPFS implementation using shared directory storage.

    All clusters access the same directory, simulating IPFS's global nature.
    CID is derived from the model hash (content-addressed).
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        """
        Initialize mock IPFS.

        Args:
            storage_path: Path to shared storage directory. If None, uses in-memory storage.
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._memory_store: Dict[str, bytes] = {}
        self._store_lock = threading.Lock()

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

    def add(self, model: np.ndarray) -> str:
        """Store model and return CID (which equals the hash)."""
        cid = compute_model_hash(model)
        serialized = pickle.dumps(model)

        if self._storage_path:
            file_path = self._storage_path / cid
            with open(file_path, "wb") as f:
                f.write(serialized)
        else:
            with self._store_lock:
                self._memory_store[cid] = serialized

        return cid

    def get(self, cid: str) -> Optional[np.ndarray]:
        """Retrieve model by CID."""
        if self._storage_path:
            file_path = self._storage_path / cid
            if not file_path.exists():
                return None
            with open(file_path, "rb") as f:
                return pickle.loads(f.read())
        else:
            with self._store_lock:
                serialized = self._memory_store.get(cid)
                if serialized is None:
                    return None
                return pickle.loads(serialized)

    def exists(self, cid: str) -> bool:
        """Check if model exists."""
        if self._storage_path:
            return (self._storage_path / cid).exists()
        else:
            with self._store_lock:
                return cid in self._memory_store

    def clear(self) -> None:
        """Clear all stored models (for testing)."""
        if self._storage_path:
            for f in self._storage_path.iterdir():
                if f.is_file():
                    f.unlink()
        else:
            with self._store_lock:
                self._memory_store.clear()


class MockBlockchain(BlockchainInterface):
    """
    Mock Blockchain implementation using shared JSON file or in-memory storage.

    All clusters access the same registry, simulating blockchain's global nature.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        """
        Initialize mock blockchain.

        Args:
            storage_path: Path to shared JSON file. If None, uses in-memory storage.
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._memory_registry: Dict[str, Dict[int, ModelAnchor]] = {}
        self._registry_lock = threading.Lock()

        if self._storage_path:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._storage_path.exists():
                self._storage_path.write_text("{}")

    def _load_registry(self) -> Dict[str, Dict[int, ModelAnchor]]:
        """Load registry from file. Caller must hold _registry_lock."""
        if not self._storage_path:
            return self._memory_registry

        try:
            data = json.loads(self._storage_path.read_text())
            registry: Dict[str, Dict[int, ModelAnchor]] = {}
            for cluster_id, rounds in data.items():
                registry[cluster_id] = {}
                for round_str, anchor_data in rounds.items():
                    registry[cluster_id][int(round_str)] = ModelAnchor(**anchor_data)
            return registry
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_registry(self, registry: Dict[str, Dict[int, ModelAnchor]]) -> None:
        """Save registry to file. Caller must hold _registry_lock."""
        if not self._storage_path:
            self._memory_registry = registry
            return

        data: Dict[str, Dict[str, Dict]] = {}
        for cluster_id, rounds in registry.items():
            data[cluster_id] = {}
            for round_num, anchor in rounds.items():
                data[cluster_id][str(round_num)] = {
                    "cluster_id": anchor.cluster_id,
                    "round_num": anchor.round_num,
                    "cid": anchor.cid,
                    "hash": anchor.hash,
                }
        self._storage_path.write_text(json.dumps(data, indent=2))

    def anchor(self, cluster_id: str, round_num: int, cid: str, hash_val: str) -> None:
        """Record model reference."""
        anchor = ModelAnchor(cluster_id=cluster_id, round_num=round_num, cid=cid, hash=hash_val)

        if self._storage_path:
            with self._registry_lock:
                registry = self._load_registry()
                if cluster_id not in registry:
                    registry[cluster_id] = {}
                registry[cluster_id][round_num] = anchor
                self._save_registry(registry)
        else:
            with self._registry_lock:
                if cluster_id not in self._memory_registry:
                    self._memory_registry[cluster_id] = {}
                self._memory_registry[cluster_id][round_num] = anchor

    def get_anchor(self, cluster_id: str, round_num: int) -> Optional[Tuple[str, str]]:
        """Retrieve anchored reference (cid, hash)."""
        with self._registry_lock:
            if self._storage_path:
                registry = self._load_registry()
            else:
                registry = self._memory_registry

            cluster_rounds = registry.get(cluster_id, {})
            anchor = cluster_rounds.get(round_num)
            if anchor:
                return (anchor.cid, anchor.hash)
            return None

    def get_latest_anchor(self, cluster_id: str) -> Optional[ModelAnchor]:
        """Get most recent anchor for a cluster."""
        with self._registry_lock:
            if self._storage_path:
                registry = self._load_registry()
            else:
                registry = self._memory_registry

            cluster_rounds = registry.get(cluster_id, {})
            if not cluster_rounds:
                return None

            latest_round = max(cluster_rounds.keys())
            return cluster_rounds[latest_round]

    def clear(self) -> None:
        """Clear all anchors (for testing)."""
        if self._storage_path:
            self._storage_path.write_text("{}")
        else:
            with self._registry_lock:
                self._memory_registry.clear()


class KuboIPFS(IPFSInterface):
    """
    Real IPFS implementation using Kubo HTTP API.

    Connects to an IPFS daemon (typically running on port 5001) and stores
    models as content-addressed data. The returned CID is the actual IPFS
    content identifier.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:5001",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize Kubo IPFS client.

        Args:
            api_url: URL of the IPFS HTTP API (e.g., "http://ipfs-node-1:5001").
            timeout: Request timeout in seconds.
        """
        self._api_url = api_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def add(self, model: np.ndarray) -> str:
        """Store model in IPFS and return CID."""
        serialized = pickle.dumps(model)

        # IPFS API expects multipart form data.
        files = {"file": ("model.pkl", io.BytesIO(serialized), "application/octet-stream")}

        try:
            response = self._client.post(
                f"{self._api_url}/api/v0/add",
                files=files,
                params={"pin": "true"},
            )
            response.raise_for_status()
            result = response.json()
            cid = result["Hash"]
            logger.debug(f"Added model to IPFS with CID: {cid}")
            return cid
        except httpx.HTTPError as e:
            logger.error(f"Failed to add model to IPFS: {e}")
            raise RuntimeError(f"IPFS add failed: {e}") from e

    def get(self, cid: str) -> Optional[np.ndarray]:
        """Retrieve model from IPFS by CID."""
        try:
            response = self._client.post(
                f"{self._api_url}/api/v0/cat",
                params={"arg": cid},
            )
            response.raise_for_status()
            model = pickle.loads(response.content)
            logger.debug(f"Retrieved model from IPFS with CID: {cid}")
            return model
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 500:
                # CID not found or not reachable.
                logger.warning(f"CID not found in IPFS: {cid}")
                return None
            logger.error(f"Failed to get model from IPFS: {e}")
            raise RuntimeError(f"IPFS get failed: {e}") from e
        except httpx.HTTPError as e:
            logger.error(f"Failed to get model from IPFS: {e}")
            raise RuntimeError(f"IPFS get failed: {e}") from e

    def exists(self, cid: str) -> bool:
        """Check if CID exists in IPFS."""
        try:
            # Use stat to check existence without downloading content.
            response = self._client.post(
                f"{self._api_url}/api/v0/files/stat",
                params={"arg": f"/ipfs/{cid}"},
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def provide(self, cid: str) -> None:
        """Announce CID to the DHT so other peers can find it."""
        try:
            self._client.post(
                f"{self._api_url}/api/v0/routing/provide",
                params={"arg": cid},
                timeout=60.0,
            )
            logger.debug(f"Provided CID to DHT: {cid}")
        except httpx.HTTPError as e:
            logger.warning(f"Failed to provide CID to DHT: {e}")

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


class RegistryBlockchain(BlockchainInterface):
    """
    Blockchain implementation using a simple HTTP registry service.

    This provides a centralized registry that acts like a blockchain ledger
    for storing model anchors. In production, this could be replaced with
    a real blockchain (Ethereum, Solana, etc.) or a distributed registry.
    """

    def __init__(
        self,
        registry_url: str = "http://localhost:8000",
        timeout: float = 10.0,
    ) -> None:
        """
        Initialize registry blockchain client.

        Args:
            registry_url: URL of the registry service.
            timeout: Request timeout in seconds.
        """
        self._registry_url = registry_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def anchor(self, cluster_id: str, round_num: int, cid: str, hash_val: str) -> None:
        """Record model reference in registry."""
        try:
            response = self._client.post(
                f"{self._registry_url}/anchors",
                json={
                    "cluster_id": cluster_id,
                    "round_num": round_num,
                    "cid": cid,
                    "hash": hash_val,
                },
            )
            response.raise_for_status()
            logger.debug(f"Anchored model: cluster={cluster_id}, round={round_num}, cid={cid}")
        except httpx.HTTPError as e:
            logger.error(f"Failed to anchor model: {e}")
            raise RuntimeError(f"Registry anchor failed: {e}") from e

    def get_anchor(self, cluster_id: str, round_num: int) -> Optional[Tuple[str, str]]:
        """Retrieve anchored reference (cid, hash) from registry."""
        try:
            response = self._client.get(
                f"{self._registry_url}/anchors/{cluster_id}/{round_num}",
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return (data["cid"], data["hash"])
        except httpx.HTTPError as e:
            logger.error(f"Failed to get anchor: {e}")
            raise RuntimeError(f"Registry get_anchor failed: {e}") from e

    def get_latest_anchor(self, cluster_id: str) -> Optional[ModelAnchor]:
        """Get most recent anchor for a cluster from registry."""
        try:
            response = self._client.get(
                f"{self._registry_url}/anchors/{cluster_id}/latest",
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return ModelAnchor(
                cluster_id=data["cluster_id"],
                round_num=data["round_num"],
                cid=data["cid"],
                hash=data["hash"],
            )
        except httpx.HTTPError as e:
            logger.error(f"Failed to get latest anchor: {e}")
            raise RuntimeError(f"Registry get_latest_anchor failed: {e}") from e

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
