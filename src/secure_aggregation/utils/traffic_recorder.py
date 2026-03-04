"""CSV-based recorder for node-to-node traffic events."""

from __future__ import annotations

import atexit
import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl unavailable on Windows
    fcntl = None


class TrafficRecorder:
    """Singleton helper that writes message-level traffic metrics to CSV."""

    _instance: Optional["TrafficRecorder"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        node_id: str,
        clique_id: int,
        base_dir: Optional[str] = None,
    ) -> None:
        self.node_id = node_id
        self.clique_id = clique_id
        root = self._resolve_base_dir(base_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.file_path = root / "node-packets.csv"
        self._file_lock = threading.Lock()
        self._ensure_header()
        # Line buffering keeps the CSV flushed without excessive fsyncs.
        self._handle = self.file_path.open("a", newline="", buffering=1)
        self._writer = csv.writer(self._handle)
        atexit.register(self._close_handle)

    def _close_handle(self) -> None:
        try:
            self._handle.close()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_header(self) -> None:
        """Create the shared CSV with header if it does not exist."""
        header = [
            "timestamp",
            "node_id",
            "clique_id",
            "cmd",
            "direction",
            "package_type",
            "packet_bytes",
            "round",
            "peer_node",
            "additional_info",
        ]
        with self.file_path.open("a+", newline="") as handle:
            if fcntl is not None:  # pragma: no cover - Windows lacks fcntl
                fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                writer = csv.writer(handle)
                writer.writerow(header)
                handle.flush()
            if fcntl is not None:  # pragma: no cover - Windows lacks fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)

    @classmethod
    def configure(
        cls,
        node_id: str,
        clique_id: int,
        base_dir: Optional[str] = None,
    ) -> "TrafficRecorder":
        """Create the singleton instance for this node if it does not exist."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(node_id, clique_id, base_dir=base_dir)
            return cls._instance

    @classmethod
    def get_instance(cls) -> Optional["TrafficRecorder"]:
        """Return the configured recorder if available."""
        return cls._instance

    @staticmethod
    def _resolve_base_dir(base_dir: Optional[str]) -> Path:
        if base_dir:
            return Path(base_dir).expanduser()
        # Default to <repo_root>/process-runtime/nodes-message
        return Path(__file__).resolve().parents[3] / "process-runtime" / "nodes-message"

    def record(
        self,
        *,
        cmd: str,
        direction: str,
        package_type: str,
        package_size: int,
        round_idx: Optional[int],
        source: str,
        destination: Optional[str],
        additional_info: str = "",
    ) -> None:
        """Append a single traffic event to the CSV."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        peer = destination or "unknown"
        round_field = "" if round_idx is None else str(round_idx)
        row = [
            timestamp,
            self.node_id,
            str(self.clique_id),
            cmd,
            direction,
            package_type,
            str(int(package_size)),
            round_field,
            peer,
            additional_info,
        ]
        self._write_row(row)

    def _write_row(self, row: list[str]) -> None:
        with self._file_lock:
            try:
                if fcntl is not None:  # pragma: no cover - Windows lacks fcntl
                    fcntl.flock(self._handle, fcntl.LOCK_EX)
                self._writer.writerow(row)
                self._handle.flush()
            finally:
                if fcntl is not None:  # pragma: no cover - Windows lacks fcntl
                    fcntl.flock(self._handle, fcntl.LOCK_UN)

    def record_bytes_exchange(
        self,
        *,
        cmd: str,
        package_type: str,
        round_idx: Optional[int],
        source: str,
        destination: Optional[str],
        request_size: int = 0,
        response_size: int = 0,
        additional_info: str = "",
    ) -> None:
        """Record paired request/response byte counts for a single exchange."""
        if request_size > 0:
            self.record(
                cmd=cmd,
                direction="sent",
                package_type=package_type,
                package_size=request_size,
                round_idx=round_idx,
                source=source,
                destination=destination,
                additional_info=additional_info,
            )
        if response_size > 0:
            self.record(
                cmd=cmd,
                direction="received",
                package_type=package_type,
                package_size=response_size,
                round_idx=round_idx,
                source=source,
                destination=destination,
                additional_info=additional_info,
            )
