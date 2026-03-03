"""CSV-based recorder for node-to-node traffic events."""

from __future__ import annotations

import atexit
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


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
        filename = f"messages-{self.node_id}.csv"
        self.file_path = root / filename
        self._file_lock = threading.Lock()
        new_file = not self.file_path.exists()
        # Line buffering keeps the CSV flushed without excessive fsyncs.
        self._handle = self.file_path.open("a", newline="", buffering=1)
        self._writer = csv.writer(self._handle)
        atexit.register(self._handle.close)
        if new_file:
            self._writer.writerow(
                [
                    "timestamp",
                    "cmd",
                    "direction",
                    "package_type",
                    "package_size",
                    "round",
                    "source",
                    "destination",
                    "additional_info",
                ]
            )

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
        # Default to <repo_root>/process-runner/
        return Path(__file__).resolve().parents[3] / "process-runner"

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
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        dest = destination or "unknown"
        round_field = "" if round_idx is None else str(round_idx)
        row = [
            timestamp,
            cmd,
            direction,
            package_type,
            str(int(package_size)),
            round_field,
            source or "unknown",
            dest,
            additional_info,
        ]
        with self._file_lock:
            self._writer.writerow(row)

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
