"""Background monitor for host resource utilization."""

from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl unavailable on Windows
    fcntl = None

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a runtime dependency but guard defensively.
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional depending on deployment.
    torch = None

from secure_aggregation.utils.prometheus_metrics import PrometheusMetrics


class ResourceUsageRecorder:
    """Append-only CSV recorder for system resource measurements."""

    _file_lock = threading.Lock()
    _file_path: Optional[Path] = None

    @classmethod
    def _default_base_dir(cls) -> Path:
        # Align with TrafficRecorder layout: <repo_root>/process-runtime/nodes-message
        return Path(__file__).resolve().parents[3] / "process-runtime" / "nodes-message"

    @classmethod
    def _ensure_file(cls, base_dir: Optional[str]) -> Path:
        with cls._file_lock:
            if cls._file_path is None:
                root = Path(base_dir).expanduser() if base_dir else cls._default_base_dir()
                root.mkdir(parents=True, exist_ok=True)
                cls._file_path = root / "resource-usage.csv"
                if not cls._file_path.exists():
                    with cls._file_path.open("w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(
                            [
                                "timestamp",
                                "node_id",
                                "clique_id",
                                "cpu_percent",
                                "ram_percent",
                                "gpu_percent",
                            ]
                        )
            return cls._file_path

    @classmethod
    def record(
        cls,
        *,
        node_id: str,
        clique_id: int,
        cpu_percent: float,
        ram_percent: float,
        gpu_percent: float,
        base_dir: Optional[str] = None,
    ) -> None:
        path = cls._ensure_file(base_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        row = [
            timestamp,
            node_id,
            str(clique_id),
            f"{cpu_percent:.4f}",
            f"{ram_percent:.4f}",
            f"{gpu_percent:.4f}",
        ]

        handle = path.open("a", newline="")
        try:
            if fcntl is not None:  # pragma: no cover - Windows lacks fcntl
                fcntl.flock(handle, fcntl.LOCK_EX)
            writer = csv.writer(handle)
            writer.writerow(row)
            if fcntl is not None:  # pragma: no cover - Windows lacks fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


class SystemResourceMonitor:
    """Samples CPU/RAM/GPU usage for CSV + Prometheus export."""

    def __init__(
        self,
        node_id: str,
        clique_id: int,
        metrics: Optional[PrometheusMetrics],
        interval_seconds: float = 5.0,
        csv_base_dir: Optional[str] = None,
        enable_csv: bool = True,
    ) -> None:
        self._node_id = node_id
        self._clique_id = clique_id
        self._metrics = metrics
        self._interval = max(1.0, float(interval_seconds))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._csv_base_dir = csv_base_dir
        self._record_to_csv = enable_csv

    def start(self) -> None:
        """Begin periodic sampling in a daemon thread."""
        if self._thread:
            return
        if psutil is None:
            # Metrics cannot be sampled without psutil; still emit baseline GPU value if possible.
            baseline_gpu = self._get_gpu_percent()
            if self._metrics and baseline_gpu is not None:
                self._metrics.set_gpu_percent(baseline_gpu)
            return
        # Prime CPU percent sampling so the first update has a delta window.
        psutil.cpu_percent(interval=None)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="system-resource-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop periodic sampling."""
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._interval)
        self._thread = None
        self._stop_event.clear()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._interval):
                break
            self._record_sample()

    def _record_sample(self) -> None:
        if psutil is None:
            return
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
            ram_percent = float(psutil.virtual_memory().percent)
        except Exception:
            return

        if self._metrics:
            self._metrics.set_cpu_percent(cpu_percent)
            self._metrics.set_ram_percent(ram_percent)

        gpu_percent = self._get_gpu_percent()
        if gpu_percent is not None and self._metrics:
            self._metrics.set_gpu_percent(gpu_percent)
        if gpu_percent is None:
            gpu_percent = 0.0

        if self._record_to_csv:
            ResourceUsageRecorder.record(
                node_id=self._node_id,
                clique_id=self._clique_id,
                cpu_percent=cpu_percent,
                ram_percent=ram_percent,
                gpu_percent=gpu_percent,
                base_dir=self._csv_base_dir,
            )

    def _get_gpu_percent(self) -> Optional[float]:
        if torch is None:
            return 0.0
        if not torch.cuda.is_available():
            return 0.0

        try:
            device_idx = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_idx)
            total_memory = float(getattr(properties, "total_memory", 0.0))
            if total_memory <= 0:
                return 0.0
            used_memory = float(torch.cuda.memory_allocated(device_idx))
            return (used_memory / total_memory) * 100.0
        except Exception:
            return None
