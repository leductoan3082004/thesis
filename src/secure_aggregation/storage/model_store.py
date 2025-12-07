"""
Storage abstractions for IPFS and Blockchain with mock implementations.

These interfaces allow seamless replacement with real IPFS/blockchain
when they become available. The mock implementations use shared storage
(directory or in-memory) to simulate the global nature of these services.
"""

import hashlib
import json
import os
import pickle
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


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
