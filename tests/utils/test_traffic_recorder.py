"""Tests for TrafficRecorder shared packet CSV."""

from __future__ import annotations

import shutil
from pathlib import Path

from secure_aggregation.utils.traffic_recorder import TrafficRecorder


def _workspace_tmp(tmp_path) -> Path:
    base_dir = Path(__file__).resolve().parents[2] / "tmp-tests" / tmp_path.name
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def test_traffic_recorder_creates_shared_csv(tmp_path):
    workspace_dir = _workspace_tmp(tmp_path)
    TrafficRecorder._instance = None
    recorder = TrafficRecorder.configure("node_a", 1, base_dir=str(workspace_dir))
    recorder.record_bytes_exchange(
        cmd="TestRPC",
        package_type="sap",
        round_idx=0,
        source="node_a",
        destination="node_b",
        request_size=128,
        response_size=64,
        additional_info="phase=round0",
    )
    recorder._close_handle()

    TrafficRecorder._instance = None
    recorder_b = TrafficRecorder.configure("node_b", 2, base_dir=str(workspace_dir))
    recorder_b.record_bytes_exchange(
        cmd="TestRPC",
        package_type="bridge",
        round_idx=1,
        source="node_b",
        destination="node_c",
        request_size=32,
        response_size=0,
        additional_info="gossip",
    )
    recorder_b._close_handle()

    csv_path = workspace_dir / "node-packets.csv"
    rows = csv_path.read_text().strip().splitlines()
    # Expect header + 3 records (two for node_a's request/response, one for node_b send)
    assert len(rows) == 4
    assert rows[1].split(",")[1] == "node_a"
    assert rows[-1].split(",")[1] == "node_b"
    shutil.rmtree(workspace_dir, ignore_errors=True)
