"""Managed-process registry with PID tracking, graceful shutdown, and port verification."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.runtime.port_allocator import _port_in_use


SHUTDOWN_TIMEOUT = 10

# PID value used for externally-managed processes (e.g. blockchain via manage.sh).
# Must never be sent to os.kill(); guards in _pid_alive/_kill_pid enforce this.
SENTINEL_PID = -1

# Shutdown proceeds in reverse registration order within each tier.
_TIER_ORDER = ["training", "ttp", "monitoring", "infrastructure"]


@dataclass
class ManagedProcess:
    name: str
    pid: int
    pid_file: str
    ports: List[int] = field(default_factory=list)
    log_file: str = ""
    component_type: str = "training"
    started_at: str = ""


class ProcessRegistry:
    """Track and control all managed processes for the secure-aggregation stack."""

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path
        self._processes: OrderedDict[str, ManagedProcess] = OrderedDict()
        self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for entry in data:
            proc = ManagedProcess(**entry)
            self._processes[proc.name] = proc

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(p) for p in self._processes.values()]
        self._path.write_text(json.dumps(payload, indent=2) + "\n")

    # -- public API -----------------------------------------------------------

    def register(self, proc: ManagedProcess) -> None:
        if not proc.started_at:
            proc.started_at = datetime.now(timezone.utc).isoformat()
        pid_path = Path(proc.pid_file)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(f"{proc.pid}\n")
        self._processes[proc.name] = proc
        self._save()

    def deregister(self, name: str) -> None:
        proc = self._processes.pop(name, None)
        if proc:
            Path(proc.pid_file).unlink(missing_ok=True)
        self._save()

    def status_all(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for proc in self._processes.values():
            alive = _process_alive(proc)
            rows.append(
                {
                    "name": proc.name,
                    "pid": proc.pid,
                    "alive": alive,
                    "ports": proc.ports,
                    "type": proc.component_type,
                }
            )
        return rows

    def stop_all(self) -> None:
        """Graceful shutdown in tier order, then verify ports are released."""
        tiers = self._group_by_tier()
        for tier in _TIER_ORDER:
            procs = tiers.get(tier, [])
            if not procs:
                continue
            self._stop_tier(procs)
        still_busy = self._verify_ports_freed()
        if still_busy:
            labels = ", ".join(f"{name}:{port}" for name, port in still_busy)
            print(f"Warning: ports still occupied after shutdown: {labels}")
        self._processes.clear()
        self._save()

    def cleanup_stale(self, purge_logs: bool = False) -> None:
        """Remove stale PID files and optionally purge logs."""
        stale = [name for name, proc in self._processes.items() if not _process_alive(proc)]
        for name in stale:
            proc = self._processes[name]
            Path(proc.pid_file).unlink(missing_ok=True)
            if purge_logs and proc.log_file:
                Path(proc.log_file).unlink(missing_ok=True)
        for name in stale:
            self._processes.pop(name, None)
        self._save()

    def has_stale_on_ports(self, ports: List[int]) -> List[Tuple[str, int]]:
        """Return (name, port) pairs where a stale process occupies a needed port."""
        conflicts: List[Tuple[str, int]] = []
        for proc in self._processes.values():
            if not _process_alive(proc):
                for port in proc.ports:
                    if port in ports and _port_in_use(port):
                        conflicts.append((proc.name, port))
        return conflicts

    def kill_stale(self) -> int:
        """Force-kill any registered processes that are still alive and return count killed."""
        killed = 0
        for proc in list(self._processes.values()):
            if _pid_alive(proc.pid):
                _kill_pid(proc.pid, escalate=True)
                killed += 1
            Path(proc.pid_file).unlink(missing_ok=True)
        self._processes.clear()
        self._save()
        return killed

    @property
    def names(self) -> List[str]:
        return list(self._processes.keys())

    # -- internal helpers -----------------------------------------------------

    def _group_by_tier(self) -> Dict[str, List[ManagedProcess]]:
        tiers: Dict[str, List[ManagedProcess]] = {}
        for proc in self._processes.values():
            tier = proc.component_type if proc.component_type in _TIER_ORDER else "training"
            tiers.setdefault(tier, []).append(proc)
        return tiers

    def _stop_tier(self, procs: List[ManagedProcess]) -> None:
        for proc in reversed(procs):
            if not _process_alive(proc):
                Path(proc.pid_file).unlink(missing_ok=True)
                continue
            _kill_pid(proc.pid, escalate=True)
            Path(proc.pid_file).unlink(missing_ok=True)

    def _verify_ports_freed(self) -> List[Tuple[str, int]]:
        busy: List[Tuple[str, int]] = []
        for proc in self._processes.values():
            for port in proc.ports:
                if _port_in_use(port) or lsof_port_pids(port):
                    busy.append((proc.name, port))
        return busy


# -- standalone helpers -------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A zombie process still responds to kill(pid, 0) but is not a healthy
    # running service.
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        stat = stat_path.read_text()
        fields = stat.split()
        if len(fields) > 2 and fields[2] == "Z":
            return False
    except OSError:
        return False
    return True


def _process_alive(proc: ManagedProcess) -> bool:
    """Check if a managed process is alive.

    Externally-managed processes (SENTINEL_PID) are checked via their
    registered ports since there is no single PID to probe.
    """
    if proc.pid == SENTINEL_PID:
        return bool(proc.ports) and any(_port_in_use(p) for p in proc.ports)
    return _pid_alive(proc.pid)


def _kill_pid(pid: int, *, escalate: bool = False) -> None:
    if pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + SHUTDOWN_TIMEOUT
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.5)
    if escalate:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def lsof_port_pids(port: int) -> List[int]:
    """Return PIDs listening on *port* using lsof (empty list on failure)."""
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        return [int(p) for p in result.stdout.split() if p.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return []