"""
ECM (External Cluster Model) Buffer for storing references from neighbor clusters.

Bridge nodes receive ECMs from neighbor clusters and buffer them until the next
aggregation round when they are sent to the aggregator.
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class ECM:
    """External Cluster Model reference with convergence status."""

    cid: str
    hash: str
    source_cluster: Optional[str] = None
    received_at: float = 0.0
    cluster_converged: bool = False
    cluster_delta_norm: float = 0.0
    round_idx: int = -1
    is_signal: bool = False
    convergence_data_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.received_at == 0.0:
            self.received_at = time.time()


class ECMBuffer:
    """
    Thread-safe buffer for incoming ECMs from neighbor clusters.

    Deduplicates by CID and filters by freshness window.
    """

    def __init__(self, freshness_window: float = 300.0) -> None:
        """
        Initialize ECM buffer.

        Args:
            freshness_window: Maximum age (seconds) for valid ECMs.
        """
        if freshness_window <= 0:
            raise ValueError("freshness_window must be positive")
        self.freshness_window = freshness_window
        self._buffer: Dict[str, ECM] = {}
        self._lock = threading.Lock()

    def add(self, ecm: ECM) -> None:
        """
        Add ECM to buffer, replacing if newer for same CID.

        Args:
            ecm: External cluster model reference to add.
        """
        with self._lock:
            existing = self._buffer.get(ecm.cid)
            if existing is None or ecm.received_at > existing.received_at:
                self._buffer[ecm.cid] = ecm

    def add_from_message(
        self,
        cid: str,
        hash_val: str,
        source_cluster: Optional[str] = None,
    ) -> None:
        """
        Add ECM from raw message data.

        Args:
            cid: IPFS content identifier.
            hash_val: SHA256 hash for verification.
            source_cluster: Optional origin cluster ID.
        """
        ecm = ECM(cid=cid, hash=hash_val, source_cluster=source_cluster)
        self.add(ecm)

    def get_fresh_ecms(self, now: Optional[float] = None) -> List[ECM]:
        """
        Return ECMs within freshness window.

        Args:
            now: Current timestamp (defaults to time.time()).

        Returns:
            List of fresh ECMs.
        """
        if now is None:
            now = time.time()
        cutoff = now - self.freshness_window
        with self._lock:
            return [e for e in self._buffer.values() if e.received_at > cutoff]

    def get_all(self) -> List[ECM]:
        """Return all ECMs regardless of freshness."""
        with self._lock:
            return list(self._buffer.values())

    def get_unique_cids(self) -> Dict[str, str]:
        """
        Return mapping of unique CID -> hash for fresh ECMs.

        Used by aggregator to deduplicate and fetch external models.
        """
        fresh = self.get_fresh_ecms()
        return {ecm.cid: ecm.hash for ecm in fresh if not ecm.is_signal}

    def clear(self) -> None:
        """Clear all ECMs from buffer."""
        with self._lock:
            self._buffer.clear()

    def pop_signal_ecms(self) -> List[ECM]:
        """Remove and return ECMs that represent convergence signals."""
        with self._lock:
            signals = [ecm for ecm in self._buffer.values() if ecm.is_signal]
            for ecm in signals:
                self._buffer.pop(ecm.cid, None)
            return signals

    def remove_stale(self, now: Optional[float] = None) -> int:
        """
        Remove stale ECMs and return count removed.

        Args:
            now: Current timestamp (defaults to time.time()).

        Returns:
            Number of ECMs removed.
        """
        if now is None:
            now = time.time()
        cutoff = now - self.freshness_window
        with self._lock:
            stale_cids = [cid for cid, ecm in self._buffer.items() if ecm.received_at <= cutoff]
            for cid in stale_cids:
                del self._buffer[cid]
            return len(stale_cids)

    def remove_cids(self, cids: Iterable[str]) -> int:
        """Remove ECMs by CID and return count removed."""
        removed = 0
        with self._lock:
            for cid in cids:
                if cid in self._buffer:
                    del self._buffer[cid]
                    removed += 1
        return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
