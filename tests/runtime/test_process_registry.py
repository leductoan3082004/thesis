"""Tests for scripts.runtime.process_registry — PID tracking, shutdown tiers, and port verification."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from scripts.runtime.process_registry import (
    ManagedProcess,
    ProcessRegistry,
    SENTINEL_PID,
    _pid_alive,
    _kill_pid,
    lsof_port_pids,
)


@pytest.fixture()
def registry_path(tmp_path):
    return tmp_path / "registry.json"


@pytest.fixture()
def registry(registry_path):
    return ProcessRegistry(registry_path)


def _make_proc(name="test-proc", pid=99999, tmp_path=None, component_type="training", ports=None):
    pid_file = str(tmp_path / f"{name}.pid") if tmp_path else f"/tmp/{name}.pid"
    return ManagedProcess(
        name=name,
        pid=pid,
        pid_file=pid_file,
        ports=ports or [],
        log_file="",
        component_type=component_type,
    )


class TestManagedProcess:
    def test_defaults(self):
        proc = ManagedProcess(name="x", pid=1, pid_file="/tmp/x.pid")
        assert proc.ports == []
        assert proc.log_file == ""
        assert proc.component_type == "training"
        assert proc.started_at == ""


class TestProcessRegistry:
    def test_register_creates_pid_file(self, registry, tmp_path):
        proc = _make_proc(tmp_path=tmp_path)
        registry.register(proc)
        pid_path = tmp_path / "test-proc.pid"
        assert pid_path.exists()
        assert pid_path.read_text().strip() == "99999"

    def test_register_sets_started_at(self, registry, tmp_path):
        proc = _make_proc(tmp_path=tmp_path)
        registry.register(proc)
        assert proc.started_at != ""

    def test_register_preserves_explicit_started_at(self, registry, tmp_path):
        proc = _make_proc(tmp_path=tmp_path)
        proc.started_at = "2024-01-01T00:00:00+00:00"
        registry.register(proc)
        assert proc.started_at == "2024-01-01T00:00:00+00:00"

    def test_deregister_removes_pid_file(self, registry, tmp_path):
        proc = _make_proc(tmp_path=tmp_path)
        registry.register(proc)
        registry.deregister("test-proc")
        assert not (tmp_path / "test-proc.pid").exists()

    def test_deregister_nonexistent_is_safe(self, registry):
        registry.deregister("ghost")

    def test_names_property(self, registry, tmp_path):
        for i in range(3):
            proc = _make_proc(name=f"proc-{i}", tmp_path=tmp_path)
            registry.register(proc)
        assert registry.names == ["proc-0", "proc-1", "proc-2"]

    def test_status_all_reports_dead_for_fake_pid(self, registry, tmp_path):
        proc = _make_proc(pid=2147483647, tmp_path=tmp_path)
        registry.register(proc)
        rows = registry.status_all()
        assert len(rows) == 1
        assert rows[0]["name"] == "test-proc"
        assert rows[0]["alive"] is False

    def test_status_all_reports_alive_for_self(self, registry, tmp_path):
        proc = _make_proc(pid=os.getpid(), tmp_path=tmp_path)
        registry.register(proc)
        rows = registry.status_all()
        assert rows[0]["alive"] is True

    def test_persistence_across_instances(self, registry_path, tmp_path):
        reg1 = ProcessRegistry(registry_path)
        proc = _make_proc(tmp_path=tmp_path)
        reg1.register(proc)

        reg2 = ProcessRegistry(registry_path)
        assert "test-proc" in reg2.names

    def test_persistence_format(self, registry_path, tmp_path):
        reg = ProcessRegistry(registry_path)
        reg.register(_make_proc(tmp_path=tmp_path))
        data = json.loads(registry_path.read_text())
        assert isinstance(data, list)
        assert data[0]["name"] == "test-proc"

    def test_cleanup_stale_removes_dead_processes(self, registry, tmp_path):
        proc = _make_proc(pid=2147483647, tmp_path=tmp_path)
        registry.register(proc)
        registry.cleanup_stale()
        assert registry.names == []

    def test_cleanup_stale_preserves_alive_processes(self, registry, tmp_path):
        proc = _make_proc(pid=os.getpid(), tmp_path=tmp_path)
        registry.register(proc)
        registry.cleanup_stale()
        assert "test-proc" in registry.names

    def test_cleanup_stale_purges_logs(self, registry, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("log content")
        proc = _make_proc(pid=2147483647, tmp_path=tmp_path)
        proc.log_file = str(log_file)
        registry.register(proc)
        registry.cleanup_stale(purge_logs=True)
        assert not log_file.exists()

    def test_has_stale_on_ports(self, registry, tmp_path):
        proc = _make_proc(pid=2147483647, tmp_path=tmp_path, ports=[59999])
        registry.register(proc)
        # Port 59999 is not actually in use, so it should not appear as stale.
        conflicts = registry.has_stale_on_ports([59999])
        assert conflicts == []

    def test_kill_stale_clears_registry(self, registry, tmp_path):
        proc = _make_proc(pid=2147483647, tmp_path=tmp_path)
        registry.register(proc)
        killed = registry.kill_stale()
        assert killed == 0  # PID doesn't exist, so nothing to kill.
        assert registry.names == []

    def test_stop_all_clears_registry(self, registry, tmp_path):
        proc = _make_proc(pid=2147483647, tmp_path=tmp_path)
        registry.register(proc)
        registry.stop_all()
        assert registry.names == []

    def test_sentinel_pid_stop_all_is_safe(self, registry, tmp_path):
        """Sentinel PID entries (e.g. blockchain) must not trigger os.kill(0/−1)."""
        proc = _make_proc(name="blockchain", pid=SENTINEL_PID, tmp_path=tmp_path, ports=[9000])
        registry.register(proc)
        registry.stop_all()
        assert registry.names == []

    def test_sentinel_pid_reported_as_dead(self, registry, tmp_path):
        proc = _make_proc(name="blockchain", pid=SENTINEL_PID, tmp_path=tmp_path)
        registry.register(proc)
        rows = registry.status_all()
        assert rows[0]["alive"] is False

    def test_load_corrupt_registry(self, registry_path):
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text("not valid json")
        reg = ProcessRegistry(registry_path)
        assert reg.names == []


class TestPidAlive:
    def test_self_is_alive(self):
        assert _pid_alive(os.getpid()) is True

    def test_bogus_pid_is_dead(self):
        assert _pid_alive(2147483647) is False

    def test_pid_zero_is_dead(self):
        """PID 0 must never be treated as alive — os.kill(0, 0) targets the process group."""
        assert _pid_alive(0) is False

    def test_sentinel_pid_is_dead(self):
        assert _pid_alive(SENTINEL_PID) is False

    def test_negative_pid_is_dead(self):
        assert _pid_alive(-42) is False


class TestKillPid:
    def test_kill_nonexistent_pid_is_safe(self):
        _kill_pid(2147483647, escalate=False)

    def test_kill_pid_zero_is_noop(self):
        """Sending signals to PID 0 would target the entire process group."""
        _kill_pid(0, escalate=True)

    def test_kill_sentinel_pid_is_noop(self):
        _kill_pid(SENTINEL_PID, escalate=True)

    def test_kill_negative_pid_is_noop(self):
        _kill_pid(-1, escalate=True)

    def test_kill_real_subprocess(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        assert _pid_alive(proc.pid)
        _kill_pid(proc.pid, escalate=True)
        proc.wait(timeout=15)
        assert not _pid_alive(proc.pid)


class TestLsofPortPids:
    def test_returns_empty_for_unused_port(self):
        result = lsof_port_pids(59997)
        assert result == []
