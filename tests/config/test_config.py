import json
from pathlib import Path

import pytest

from secure_aggregation.config import MountConfig, NodeConfig, ScenarioConfig, validate_mounts


def test_scenario_defaults_and_validation() -> None:
    data = {
        "name": "demo",
        "participants": ["a", "b", "c"],
        "threshold": 2,
        "clique_size": 2,
        "timeouts": {"advertise_keys": 10},
    }
    scenario = ScenarioConfig.from_dict(data)
    assert scenario.inter_clique_edges == "small_world"
    assert scenario.timeouts.advertise_keys == 10.0
    assert scenario.timeouts.share_keys == 5.0  # default


def test_threshold_validation_fails() -> None:
    data = {"name": "bad", "participants": ["a"], "threshold": 2, "clique_size": 1}
    with pytest.raises(ValueError):
        ScenarioConfig.from_dict(data)


def test_node_hostname_must_match_scenario() -> None:
    scenario = ScenarioConfig.from_dict(
        {
            "name": "demo",
            "participants": ["node1"],
            "threshold": 1,
            "clique_size": 1,
            "service_hostnames": {"node1": "node1-service"},
        }
    )
    with pytest.raises(ValueError):
        NodeConfig.from_dict(
            {"node_id": "node1", "role": "trainer", "host": "other-host", "port": 5000}, scenario
        )


def test_validate_mounts_creates_when_requested(tmp_path: Path) -> None:
    mounts = MountConfig(config_dir="cfg", data_dir="data", logs_dir="logs", checkpoints_dir="chk")
    resolved = validate_mounts(tmp_path, mounts, create=True)
    for path in resolved.values():
        assert path.exists()
        assert path.is_dir()
