"""
Storage abstractions for IPFS and Blockchain with mock and real implementations.

These interfaces allow seamless switching between mock implementations
(for testing/development) and real implementations (for production).

Real implementations:
- KuboIPFS: Connects to IPFS Kubo HTTP API
- RegistryBlockchain: Uses a simple HTTP registry service
"""

import base64
import hashlib
import io
import json
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import httpx
import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)

IPFS_LOG_TAG = "~ IPFS ~"
BLOCKCHAIN_LOG_TAG = "~ BLOCKCHAIN ~"


class AnchorScope(str, Enum):
    """Different namespaces supported by the blockchain gateway."""

    CLUSTER = "cluster"
    STATE = "state"
    CONTROL = "control"


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
    data_id: Optional[str] = None
    submitted_at: Optional[str] = None


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
    def anchor(
        self,
        scope_id: str,
        round_num: int,
        cid: str,
        hash_val: str,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[str]:
        """Record model reference on-chain."""
        pass

    @abstractmethod
    def commit_metadata(self, cluster_id: str, round_num: int, metadata: Dict) -> Optional[str]:
        """Commit arbitrary metadata payload and return data identifier."""
        pass

    @abstractmethod
    def get_anchor(
        self,
        scope_id: str,
        round_num: int,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
        suppress_not_found_log: bool = False,
    ) -> Optional[Tuple[str, str]]:
        """Retrieve anchored reference (cid, hash) for cluster/round."""
        pass

    @abstractmethod
    def get_latest_anchor(
        self,
        scope_id: str,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[ModelAnchor]:
        """Get most recent anchor for a cluster."""
        pass

    @abstractmethod
    def remember_anchor(
        self,
        scope_id: str,
        round_num: int,
        data_id: str,
        cid: Optional[str] = None,
        hash_val: Optional[str] = None,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> None:
        """Persist a reference that was anchored elsewhere."""
        pass

    @abstractmethod
    def fetch_data(self, data_id: str) -> Optional[Dict]:
        """Fetch raw data payload by ID."""
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

        logger.info(f"{IPFS_LOG_TAG} Stored model in MockIPFS cid={cid[:16]}...")
        return cid

    def get(self, cid: str) -> Optional[np.ndarray]:
        """Retrieve model by CID."""
        if self._storage_path:
            file_path = self._storage_path / cid
            if not file_path.exists():
                logger.warning(f"{IPFS_LOG_TAG} MockIPFS miss cid={cid[:16]}...")
                return None
            with open(file_path, "rb") as f:
                model = pickle.loads(f.read())
                logger.info(f"{IPFS_LOG_TAG} Loaded model from MockIPFS cid={cid[:16]}...")
                return model
        else:
            with self._store_lock:
                serialized = self._memory_store.get(cid)
                if serialized is None:
                    logger.warning(f"{IPFS_LOG_TAG} MockIPFS miss cid={cid[:16]}...")
                    return None
                model = pickle.loads(serialized)
                logger.info(f"{IPFS_LOG_TAG} Loaded model from MockIPFS cid={cid[:16]}...")
                return model

    def exists(self, cid: str) -> bool:
        """Check if model exists."""
        if self._storage_path:
            exists = (self._storage_path / cid).exists()
        else:
            with self._store_lock:
                exists = cid in self._memory_store
        logger.debug(f"{IPFS_LOG_TAG} MockIPFS exists={exists} cid={cid[:16]}...")
        return exists

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
        self._metadata_store: Dict[str, Dict] = {}

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
                    "data_id": anchor.data_id,
                    "submitted_at": anchor.submitted_at,
                }
        self._storage_path.write_text(json.dumps(data, indent=2))

    def anchor(
        self,
        cluster_id: str,
        round_num: int,
        cid: str,
        hash_val: str,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[str]:
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
        logger.info(
            f"{BLOCKCHAIN_LOG_TAG} MockBlockchain anchor cluster={cluster_id}, round={round_num}, cid={cid[:16]}..."
        )
        return None

    def get_anchor(
        self,
        cluster_id: str,
        round_num: int,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
        suppress_not_found_log: bool = False,
    ) -> Optional[Tuple[str, str]]:
        """Retrieve anchored reference (cid, hash)."""
        with self._registry_lock:
            if self._storage_path:
                registry = self._load_registry()
            else:
                registry = self._memory_registry

            cluster_rounds = registry.get(cluster_id, {})
            anchor = cluster_rounds.get(round_num)
            if anchor:
                logger.info(
                    f"{BLOCKCHAIN_LOG_TAG} MockBlockchain get_anchor cluster={cluster_id}, round={round_num}, cid={anchor.cid[:16]}..."
                )
                return (anchor.cid, anchor.hash)
            if not suppress_not_found_log:
                logger.warning(
                    f"{BLOCKCHAIN_LOG_TAG} MockBlockchain missing anchor cluster={cluster_id}, round={round_num}"
                )
            return None

    def get_latest_anchor(
        self,
        cluster_id: str,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[ModelAnchor]:
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
            anchor = cluster_rounds[latest_round]
            logger.info(
                f"{BLOCKCHAIN_LOG_TAG} MockBlockchain latest cluster={cluster_id}, round={anchor.round_num}, cid={anchor.cid[:16]}..."
            )
            return anchor

    def clear(self) -> None:
        """Clear all anchors (for testing)."""
        if self._storage_path:
            self._storage_path.write_text("{}")
        else:
            with self._registry_lock:
                self._memory_registry.clear()

    def remember_anchor(
        self,
        cluster_id: str,
        round_num: int,
        data_id: str,
        cid: Optional[str] = None,
        hash_val: Optional[str] = None,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> None:
        """Mock registry ignores data_id and stores provided fields."""
        if cid is None or hash_val is None:
            return
        self.anchor(cluster_id, round_num, cid, hash_val)

    def commit_metadata(self, cluster_id: str, round_num: int, metadata: Dict) -> Optional[str]:
        """Store metadata payload locally and return synthetic data_id."""
        data_id = f"{cluster_id}-{round_num}-{int(time.time() * 1000)}"
        record = {
            "data_id": data_id,
            "payload": {
                "cluster_id": cluster_id,
                "round": round_num,
                "metadata": metadata,
            },
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._metadata_store[data_id] = record
        logger.info(
            f"{BLOCKCHAIN_LOG_TAG} MockBlockchain stored metadata cluster={cluster_id}, round={round_num}, data_id={data_id}"
        )
        return data_id

    def fetch_data(self, data_id: str) -> Optional[Dict]:
        """Fetch metadata payload by data_id."""
        record = self._metadata_store.get(data_id)
        if record is None:
            logger.warning(f"{BLOCKCHAIN_LOG_TAG} MockBlockchain missing data_id={data_id}")
        return record


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
        timeout: float = 5.0,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        replica_api_urls: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Initialize Kubo IPFS client.

        Args:
            api_url: URL of the IPFS HTTP API (e.g., "http://ipfs-node-1:5001").
            timeout: Request timeout in seconds.
            max_retries: Number of attempts for read operations before failing.
            retry_delay: Base delay (seconds) between retries (exponential backoff).
        """
        self._api_url = api_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._max_retries = max(1, max_retries)
        self._retry_delay = max(0.1, retry_delay)
        self._replica_clients: List[Tuple[str, httpx.Client]] = []
        if replica_api_urls:
            for replica in replica_api_urls:
                cleaned = str(replica).strip()
                if not cleaned:
                    continue
                cleaned = cleaned.rstrip("/")
                if cleaned == self._api_url:
                    continue
                self._replica_clients.append((cleaned, httpx.Client(timeout=timeout)))

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
            self._replicate_to_peers(serialized, cid)
            logger.info(f"{IPFS_LOG_TAG} Uploaded model to IPFS cid={cid[:16]}...")
            return cid
        except httpx.HTTPError as e:
            logger.error(f"{IPFS_LOG_TAG} Failed to add model to IPFS: {e}")
            raise RuntimeError(f"IPFS add failed: {e}") from e

    def _replicate_to_peers(self, serialized: bytes, cid: str) -> None:
        """Eagerly store the model on additional IPFS daemons to avoid cold fetches."""
        if not self._replica_clients:
            return
        for replica_url, client in self._replica_clients:
            try:
                response = client.post(
                    f"{replica_url}/api/v0/add",
                    files={"file": ("model.pkl", io.BytesIO(serialized), "application/octet-stream")},
                    params={"pin": "true"},
                )
                response.raise_for_status()
                replica_cid = response.json().get("Hash")
                if replica_cid != cid:
                    logger.warning(
                        f"{IPFS_LOG_TAG} Replica {replica_url} returned mismatched cid={replica_cid} "
                        f"(expected {cid[:16]}...)"
                    )
            except httpx.HTTPError as exc:
                logger.warning(f"{IPFS_LOG_TAG} Failed to replicate CID {cid[:16]} to {replica_url}: {exc}")

    def get(self, cid: str) -> Optional[np.ndarray]:
        """Retrieve model from IPFS by CID with retry/backoff."""
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.post(
                    f"{self._api_url}/api/v0/cat",
                    params={"arg": cid},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                model = pickle.loads(response.content)
                logger.info(f"{IPFS_LOG_TAG} Retrieved model from IPFS cid={cid[:16]}...")
                return model
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 500:
                    logger.warning(f"{IPFS_LOG_TAG} CID not found in IPFS: {cid}")
                    return None
                last_error = e
            except httpx.TimeoutException as e:
                last_error = e
            except httpx.HTTPError as e:
                last_error = e

            if attempt < self._max_retries - 1:
                delay = self._retry_delay 
                logger.warning(
                    f"{IPFS_LOG_TAG} Get CID {cid[:16]} attempt {attempt + 1}/{self._max_retries} failed: "
                    f"{last_error}. Retrying in {delay:.1f}s"
                )
                time.sleep(delay)

        logger.error(
            f"{IPFS_LOG_TAG} Failed to get model from IPFS cid={cid[:16]} after {self._max_retries} attempts"
        )
        raise RuntimeError(f"IPFS get failed: {last_error}") from last_error

    def exists(self, cid: str) -> bool:
        """Check if CID exists in IPFS."""
        try:
            # Use stat to check existence without downloading content.
            response = self._client.post(
                f"{self._api_url}/api/v0/files/stat",
                params={"arg": f"/ipfs/{cid}"},
            )
            exists = response.status_code == 200
            logger.debug(f"{IPFS_LOG_TAG} Checked CID existence={exists} cid={cid[:16]}...")
            return exists
        except httpx.HTTPError:
            return False

    def provide(self, cid: str) -> None:
        """Announce CID to the DHT so other peers can find it."""
        self._announce_provider(self._api_url, self._client, cid)
        for replica_url, client in self._replica_clients:
            self._announce_provider(replica_url, client, cid)

    @staticmethod
    def _announce_provider(api_url: str, client: httpx.Client, cid: str) -> None:
        try:
            client.post(
                f"{api_url}/api/v0/routing/provide",
                params={"arg": cid},
                timeout=60.0,
            )
            logger.info(f"{IPFS_LOG_TAG} Provided CID via {api_url} cid={cid[:16]}...")
        except httpx.HTTPError as e:
            logger.warning(f"{IPFS_LOG_TAG} Failed to provide CID via {api_url}: {e}")

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
        for _, client in self._replica_clients:
            client.close()

    def remember_anchor(
        self,
        cluster_id: str,
        round_num: int,
        data_id: str,
        cid: Optional[str] = None,
        hash_val: Optional[str] = None,
    ) -> None:
        """Registry-backed implementation does not support external references."""
        return


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

    def anchor(
        self,
        cluster_id: str,
        round_num: int,
        cid: str,
        hash_val: str,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> None:
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

    def get_anchor(
        self,
        cluster_id: str,
        round_num: int,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[Tuple[str, str]]:
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

    def get_latest_anchor(
        self,
        cluster_id: str,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[ModelAnchor]:
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


class LocalAnchorStore:
    """Persists mapping of cluster anchors to blockchain data IDs."""

    def __init__(self, storage_path: Optional[str]) -> None:
        self._path = Path(storage_path) if storage_path else None
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Dict]] = {}

        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                try:
                    raw = json.loads(self._path.read_text())
                    if isinstance(raw, dict):
                        self._data = raw
                except json.JSONDecodeError:
                    logger.warning("Failed to parse anchor state at %s", self._path)

        self._data.setdefault("clusters", {})
        self._data.setdefault("states", {})

    def _persist(self) -> None:
        if not self._path:
            return
        self._path.write_text(json.dumps(self._data, indent=2))

    def _bucket_for_scope(self, scope: AnchorScope) -> str:
        if scope == AnchorScope.STATE:
            return "states"
        if scope == AnchorScope.CLUSTER:
            return "clusters"
        return f"scopes::{scope.value}"

    def remember(
        self,
        cluster_id: str,
        round_num: int,
        data_id: str,
        cid: Optional[str],
        hash_val: Optional[str],
        submitted_at: Optional[str] = None,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> None:
        with self._lock:
            bucket_name = self._bucket_for_scope(scope)
            clusters = self._data.setdefault(bucket_name, {})
            cluster_entry = clusters.setdefault(cluster_id, {"rounds": {}, "latest_round": -1})
            cluster_entry["rounds"][str(round_num)] = {
                "data_id": data_id,
                "cid": cid,
                "hash": hash_val,
                "submitted_at": submitted_at,
                "entity": scope.value,
            }
            latest = cluster_entry.get("latest_round", -1)
            if round_num >= latest:
                cluster_entry["latest_round"] = round_num
            self._persist()

    def get_round(
        self,
        cluster_id: str,
        round_num: int,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[Dict]:
        with self._lock:
            bucket_name = self._bucket_for_scope(scope)
            cluster_entry = self._data.get(bucket_name, {}).get(cluster_id, {})
            rounds = cluster_entry.get("rounds", {})
            entry = rounds.get(str(round_num))
            if entry is None:
                return None
            return dict(entry)

    def get_latest(
        self,
        cluster_id: str,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[Tuple[int, Dict]]:
        with self._lock:
            bucket_name = self._bucket_for_scope(scope)
            cluster_entry = self._data.get(bucket_name, {}).get(cluster_id)
            if not cluster_entry:
                return None
            latest_round = cluster_entry.get("latest_round")
            if latest_round is None or latest_round < 0:
                return None
            entry = cluster_entry.get("rounds", {}).get(str(latest_round))
            if entry is None:
                return None
            return latest_round, dict(entry)


class EdDSAJWTSigner:
    """Issues JWT tokens signed with an Ed25519 private key."""

    def __init__(
        self,
        private_key_path: str,
        subject: str,
        role: str = "trainer",
        state: str = "system",
        ttl_seconds: int = 24 * 3600,
    ) -> None:
        key_bytes = Path(private_key_path).read_bytes()
        self._private_key = serialization.load_pem_private_key(key_bytes, password=None)
        if not isinstance(self._private_key, ed25519.Ed25519PrivateKey):
            raise ValueError("Private key must be Ed25519")
        self._subject = subject
        self._role = role
        self._state = state
        self._ttl = ttl_seconds

    @staticmethod
    def _b64(value: Dict) -> str:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("utf-8")

    def issue(self) -> str:
        header = {"alg": "EdDSA", "typ": "JWT"}
        payload = {
            "sub": self._subject,
            "role": self._role,
            "state": self._state,
            "exp": int(time.time()) + self._ttl,
        }
        unsigned = f"{self._b64(header)}.{self._b64(payload)}"
        signature = self._private_key.sign(unsigned.encode("utf-8"))
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("utf-8")
        return f"{unsigned}.{sig_b64}"


class GatewayBlockchain(BlockchainInterface):
    """Blockchain client that talks to the Fabric gateway via REST + JWT."""

    def __init__(
        self,
        base_url: str,
        identity: str,
        private_key_path: str,
        *,
        state_path: Optional[str] = None,
        jwt_role: str = "trainer",
        jwt_state: str = "system",
        jwt_ttl_seconds: int = 24 * 3600,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._signer = EdDSAJWTSigner(
            private_key_path=private_key_path,
            subject=identity,
            role=jwt_role,
            state=jwt_state,
            ttl_seconds=jwt_ttl_seconds,
        )
        self._store = LocalAnchorStore(state_path or None)

    def _auth_headers(self) -> Dict[str, str]:
        token = self._signer.issue()
        return {"Authorization": f"Bearer {token}"}

    def _commit_payload(self, payload: Dict) -> Dict:
        response = self._client.post(
            f"{self._base_url}/data/commit",
            headers=self._auth_headers(),
            json={"payload": payload},
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            f"{BLOCKCHAIN_LOG_TAG} Submitted payload to gateway cluster={payload.get('cluster_id')} "
            f"round={payload.get('round')} data_id={result.get('data_id')}"
        )
        return result

    def _commit_cluster_model(self, cluster_id: str, round_num: int, cid: str, hash_val: str) -> Dict:
        response = self._client.post(
            f"{self._base_url}/cluster/models",
            headers=self._auth_headers(),
            json={
                "cluster_id": cluster_id,
                "payload": {
                    "model_hash": hash_val,
                    "cid": cid,
                    "round": round_num,
                },
            },
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            f"{BLOCKCHAIN_LOG_TAG} Submitted cluster model cluster={cluster_id} round={round_num} "
            f"data_id={result.get('data_id')}"
        )
        return result

    def _commit_state_model(self, state_id: str, state_round: int, cid: str, hash_val: str) -> Dict:
        response = self._client.post(
            f"{self._base_url}/state/models",
            headers=self._auth_headers(),
            json={
                "state_id": state_id,
                "payload": {
                    "model_hash": hash_val,
                    "cid": cid,
                    "round": state_round,
                },
            },
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            f"{BLOCKCHAIN_LOG_TAG} Submitted state model state={state_id} round={state_round} "
            f"data_id={result.get('data_id')}"
        )
        return result

    def _fetch_control_data(self, data_id: str) -> Dict:
        try:
            response = self._client.get(
                f"{self._base_url}/data/{data_id}",
                headers=self._auth_headers(),
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"{BLOCKCHAIN_LOG_TAG} Retrieved payload from gateway data_id={data_id}")
            return data
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"{BLOCKCHAIN_LOG_TAG} Gateway returned {exc.response.status_code} for data_id={data_id}"
            )
            raise
        except httpx.HTTPError as exc:
            logger.error(f"{BLOCKCHAIN_LOG_TAG} Failed to fetch data_id={data_id}: {exc}")
            raise

    def _fetch_model_metadata(self, scope: AnchorScope, data_id: str) -> Dict:
        endpoint = "/data"
        if scope == AnchorScope.CLUSTER:
            endpoint = "/cluster/models"
        elif scope == AnchorScope.STATE:
            endpoint = "/state/models"
        try:
            response = self._client.get(
                f"{self._base_url}{endpoint}/{data_id}",
                headers=self._auth_headers(),
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                f"{BLOCKCHAIN_LOG_TAG} Retrieved {scope.value} record from gateway data_id={data_id}"
            )
            return data
        except httpx.HTTPError as exc:
            logger.error(f"{BLOCKCHAIN_LOG_TAG} Failed to fetch {scope.value} data_id={data_id}: {exc}")
            raise

    def anchor(
        self,
        cluster_id: str,
        round_num: int,
        cid: str,
        hash_val: str,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[str]:
        if not isinstance(scope, AnchorScope):
            scope = AnchorScope(scope)
        if scope == AnchorScope.CLUSTER:
            record = self._commit_cluster_model(cluster_id, round_num, cid, hash_val)
        elif scope == AnchorScope.STATE:
            record = self._commit_state_model(cluster_id, round_num, cid, hash_val)
        else:
            payload = {
                "cluster_id": cluster_id,
                "round": round_num,
                "cid": cid,
                "hash": hash_val,
            }
            record = self._commit_payload(payload)
        data_id = record.get("data_id")
        submitted_at = record.get("submitted_at")
        if data_id:
            self._store.remember(
                cluster_id,
                round_num,
                data_id,
                cid,
                hash_val,
                submitted_at,
                scope=scope,
            )
            logger.info(
                f"{BLOCKCHAIN_LOG_TAG} Anchored {scope.value} model scope={cluster_id}, round={round_num}, data_id={data_id}"
            )
        return data_id

    def remember_anchor(
        self,
        cluster_id: str,
        round_num: int,
        data_id: str,
        cid: Optional[str] = None,
        hash_val: Optional[str] = None,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> None:
        self._store.remember(cluster_id, round_num, data_id, cid, hash_val, scope=scope)

    @staticmethod
    def _is_control_cluster(cluster_id: str) -> bool:
        return cluster_id.startswith("__")

    def _resolve_entry(
        self,
        cluster_id: str,
        round_num: int,
        scope: AnchorScope,
    ) -> Optional[ModelAnchor]:
        entry = self._store.get_round(cluster_id, round_num, scope=scope)
        if entry is None:
            return None
        cid = entry.get("cid")
        hash_val = entry.get("hash")
        data_id = entry.get("data_id")
        submitted_at = entry.get("submitted_at")
        entry_scope_value = entry.get("entity")
        if entry_scope_value:
            try:
                scope = AnchorScope(entry_scope_value)
            except ValueError:
                scope = AnchorScope.CLUSTER
        if (cid is None or hash_val is None) and data_id:
            try:
                if scope in (AnchorScope.CLUSTER, AnchorScope.STATE):
                    record = self._fetch_model_metadata(scope, data_id)
                    payload = record.get("payload", {})
                    if scope == AnchorScope.CLUSTER:
                        cid = payload.get("cid") or payload.get("model_cid")
                        hash_val = payload.get("model_hash")
                    else:
                        cid = payload.get("cid") or payload.get("model_cid")
                        hash_val = payload.get("model_hash")
                else:
                    record = self._fetch_control_data(data_id)
                    payload = record.get("payload", {})
                    if isinstance(payload, str):
                        try:
                            payload = json.loads(payload)
                        except json.JSONDecodeError:
                            payload = {}
                    cid = payload.get("cid")
                    hash_val = payload.get("hash")
            except httpx.HTTPError:
                return None
            submitted_at = record.get("submitted_at")
            if cid and hash_val:
                self._store.remember(
                    cluster_id,
                    round_num,
                    data_id,
                    cid,
                    hash_val,
                    submitted_at,
                    scope=scope,
                )
            else:
                logger.warning(
                    f"{BLOCKCHAIN_LOG_TAG} Missing CID/hash when fetching data_id={data_id} scope={cluster_id}"
                )

        if not cid or not hash_val:
            return None
        return ModelAnchor(
            cluster_id=cluster_id,
            round_num=round_num,
            cid=cid,
            hash=hash_val,
            data_id=data_id,
            submitted_at=submitted_at,
        )

    def get_anchor(
        self,
        cluster_id: str,
        round_num: int,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
        suppress_not_found_log: bool = False,
    ) -> Optional[Tuple[str, str]]:
        anchor = self._resolve_entry(cluster_id, round_num, scope)
        if anchor is None:
            if not suppress_not_found_log:
                if scope == AnchorScope.CONTROL or self._is_control_cluster(cluster_id):
                    logger.debug(
                        f"{BLOCKCHAIN_LOG_TAG} No anchor found for control cluster={cluster_id}, round={round_num}"
                    )
                else:
                    logger.warning(
                        f"{BLOCKCHAIN_LOG_TAG} No anchor found for cluster={cluster_id}, round={round_num}"
                    )
            return None
        logger.info(
            f"{BLOCKCHAIN_LOG_TAG} Resolved anchor for cluster={cluster_id}, round={round_num}, cid={anchor.cid[:16]}..."
        )
        return (anchor.cid, anchor.hash)

    def get_latest_anchor(
        self,
        cluster_id: str,
        *,
        scope: AnchorScope = AnchorScope.CLUSTER,
    ) -> Optional[ModelAnchor]:
        latest = self._store.get_latest(cluster_id, scope=scope)
        if latest is None:
            if scope == AnchorScope.CONTROL or self._is_control_cluster(cluster_id):
                logger.debug(f"{BLOCKCHAIN_LOG_TAG} No latest anchor for control cluster={cluster_id}")
            else:
                logger.warning(f"{BLOCKCHAIN_LOG_TAG} No latest anchor for cluster={cluster_id}")
            return None
        round_num, _ = latest
        anchor = self._resolve_entry(cluster_id, round_num, scope)
        if anchor:
            logger.info(
                f"{BLOCKCHAIN_LOG_TAG} Latest anchor cluster={cluster_id}, round={anchor.round_num}, cid={anchor.cid[:16]}..."
            )
        return anchor

    def close(self) -> None:
        self._client.close()

    def commit_metadata(self, cluster_id: str, round_num: int, metadata: Dict) -> Optional[str]:
        payload = {
            "cluster_id": cluster_id,
            "round": round_num,
            "metadata": metadata,
        }
        record = self._commit_payload(payload)
        data_id = record.get("data_id")
        if data_id:
            logger.info(
                f"{BLOCKCHAIN_LOG_TAG} Anchored metadata cluster={cluster_id}, round={round_num}, data_id={data_id}"
            )
        return data_id

    def fetch_data(self, data_id: str) -> Optional[Dict]:
        try:
            return self._fetch_control_data(data_id)
        except httpx.HTTPError:
            return None
