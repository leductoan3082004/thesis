"""Tests for resource monitor CSV recording."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from secure_aggregation.utils import resource_monitor
from secure_aggregation.utils.resource_monitor import (
    ResourceUsageRecorder,
    SystemResourceMonitor,
)


def _workspace_tmp(tmp_path) -> Path:
    """Map pytest tmp paths to a workspace-local directory."""
    base_dir = Path(__file__).resolve().parents[2] / "tmp-tests" / tmp_path.name
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def test_resource_usage_recorder_writes_rows(tmp_path):
    ResourceUsageRecorder._file_path = None
    workspace_dir = _workspace_tmp(tmp_path)
    ResourceUsageRecorder.record(
        node_id="node_a",
        clique_id=2,
        cpu_percent=50.0,
        ram_percent=25.0,
        gpu_percent=0.0,
        base_dir=str(workspace_dir),
    )
    csv_path = workspace_dir / "resource-usage.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().strip().splitlines()
    assert len(rows) == 2
    assert "node_a" in rows[1]
    shutil.rmtree(workspace_dir, ignore_errors=True)


def test_system_resource_monitor_record_sample(monkeypatch, tmp_path):
    ResourceUsageRecorder._file_path = None

    class StubPsutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 10.0

        @staticmethod
        def virtual_memory():
            return SimpleNamespace(percent=20.0)

    monkeypatch.setattr(resource_monitor, "psutil", StubPsutil)
    monkeypatch.setattr(resource_monitor, "torch", None)

    workspace_dir = _workspace_tmp(tmp_path)
    monitor = SystemResourceMonitor(
        node_id="node_b",
        clique_id=1,
        metrics=None,
        interval_seconds=1.0,
        csv_base_dir=str(workspace_dir),
    )
    monitor._record_sample()

    csv_path = workspace_dir / "resource-usage.csv"
    rows = csv_path.read_text().strip().splitlines()
    assert len(rows) == 2
    assert rows[1].split(",")[1] == "node_b"
    assert rows[1].split(",")[2] == "1"
    shutil.rmtree(workspace_dir, ignore_errors=True)
